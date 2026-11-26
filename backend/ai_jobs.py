"""Local persistent AI work queue; inference never holds the editable project lock."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from .auth import actor

import pymupdf

from bom_extractor.ai_agent import ElectricalDrawingAgent, conflict_pages, merge_independent_observations, save_json, VERSION
from bom_extractor.device_inventory import build_device_inventory, inventory_review_issues
from bom_extractor.independent import analysis_source, schematic_pages, independent_policy
from bom_extractor.vision import VisionRecognizer, TransientVisionError

ACTIVE = {"queued", "running", "reviewing", "merging"}


class StopJob(Exception):
    pass


class AIJobManager:
    def __init__(self, get_store, get_output):
        self.get_store, self.get_output = get_store, get_output
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="electrical-ai")
        self.jobs = {}
        self.guard = threading.RLock()

    def path(self, project, job):
        return Path(self.get_output()) / project / "ai_jobs" / job / "job.json"

    def save(self, job):
        with self.guard:
            job["updated_at"] = time.time()
            repo=self.get_store().repository
            if repo:
                with repo.connection(job.get('requested_by')) as conn:
                    conn.execute("INSERT INTO ai_jobs(id,project_id,input_data_version,drawing_version,requested_by,status,progress,cancel_requested,heartbeat_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,now()) ON CONFLICT(id) DO UPDATE SET status=excluded.status,progress=excluded.progress,cancel_requested=excluded.cancel_requested,heartbeat_at=now(),updated_at=now(),input_data_version=excluded.input_data_version",(job['id'],job['project_id'],job.get('input_data_version',0),job.get('drawing_version',1),job.get('requested_by'),job['status'],json.dumps(job,ensure_ascii=False),job.get('cancel_requested',False)))
            else:
                save_json(self.path(job["project_id"],job["id"]), job)

    def get(self, project, job_id):
        with self.guard:
            job = self.jobs.get((project,job_id))
            if job is None:
                repo=self.get_store().repository
                if repo:
                    with repo.connection(actor.get()['id'] if actor.get() else None) as conn:
                        row=conn.execute('SELECT progress FROM ai_jobs WHERE project_id=%s AND id=%s',(project,job_id)).fetchone()
                    if not row: raise KeyError(job_id)
                    job=row['progress']
                else:
                    path = self.path(project,job_id)
                    if not path.is_file(): raise KeyError(job_id)
                    job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] in ACTIVE:
                    job["status"] = "interrupted"
                    job["message"] = "服务重启中断了任务；可重新运行，已完成页面会使用缓存。"
                    self.save(job)
                self.jobs[(project,job_id)] = job
            return copy.deepcopy(job)

    def start(self, project, pages=None):
        vision = VisionRecognizer()
        if not vision.api_key:
            raise ValueError("请先配置可用的 AI 密钥")
        store = self.get_store()
        with store.lock, self.guard:
            if project not in store.results or project not in store.pdf_paths:
                raise KeyError(project)
            result = store.results[project]
            # Deterministic page classification must not hide a misclassified drawing from AI.
            analysis_source(result,store.pdf_paths[project],Path(self.get_output())/project)
            allowed = schematic_pages(result)
            selected = sorted(allowed if pages is None else set(pages))
            if not selected or not set(selected).issubset(allowed):
                raise ValueError("请选择有效的原理图页码；整页物料表只用于核对，不参与 AI 独立识别")
            for (pid,_),job in self.jobs.items():
                if pid == project and job["status"] in ACTIVE:
                    if job["pages"] != selected:
                        raise ValueError("已有任务正在处理其他页码，请在该任务结束后启动")
                    return copy.deepcopy(job)
            job = {"id":uuid.uuid4().hex,"project_id":project,"status":"queued","message":"等待后台识别",
                   "requested_by":str(actor.get()['id']) if actor.get() else None,
                   "input_data_version":store.data_versions.get(project,0),"drawing_version":1,
                   "pages":selected,"total_pages":len(selected),"completed_pages":0,"reviewed_pages":0,
                   "current_page":None,"api_calls":0,"cache_hits":0,"input_tokens":0,"output_tokens":0,
                   "errors":[],"cancel_requested":False,"model":vision.model,
                   "review_candidates":0,"deferred_review_pages":[],"merge":None,
                   "reasoning_effort":vision.config.get("reasoning_effort","medium"),
                   "review_reasoning_effort":vision.config.get("review_reasoning_effort","high"),
                   "created_at":time.time()}
            self.save(job)
            self.jobs[(project,job["id"])] = job
            result.summary["ai_job_id"] = job["id"]
            try:
                store.sync_and_persist(project)
                job['input_data_version']=store.data_versions.get(project,0)
                self.save(job)
                self.executor.submit(self.run,job,vision)
            except Exception:
                job.update(status="failed", message="后台任务未能启动，请检查项目保存状态后重试")
                self.save(job)
                raise
            return copy.deepcopy(job)

    def cancel(self, project, job_id):
        with self.guard:
            self.get(project,job_id)
            job=self.jobs[(project,job_id)]
            if job["status"] in ("queued","running","reviewing"):
                job["cancel_requested"]=True
                job["message"]="将在当前请求返回后停止；尚未合并的结果不会写入 BOM"
                self.save(job)
            return copy.deepcopy(job)

    def run(self, job, vision):
        store=self.get_store()
        context_token=actor.set({'id':job['requested_by']}) if job.get('requested_by') else None
        folder=Path(self.get_output())/job["project_id"]
        try:
            with store.lock:
                current=store.results[job['project_id']]
                source=analysis_source(current,store.pdf_paths[job['project_id']],folder)
                eligible=schematic_pages(current)
                infos={p.page_num:copy.deepcopy(p) for p in current.pages}
            source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            def limit_reached():
                if job["api_calls"]>=vision.config.get("max_api_calls",80):
                    return "达到单任务 API 调用次数上限"
                if job["input_tokens"]+job["output_tokens"]+40000>vision.config.get("max_total_tokens",1000000):
                    return "达到单任务 Token 预算预留边界"
                return ""
            def before_call():
                if job["cancel_requested"]:
                    raise StopJob("任务已取消")
                limit=limit_reached()
                if limit:
                    raise StopJob(limit)
                job["api_calls"]+=1
                self.save(job)
            agent=ElectricalDrawingAgent(vision,folder/"ai_cache",before_call)
            records=[]
            def account(record):
                if record.get("cache_hit"):
                    job["cache_hits"]+=1
                else:
                    usage=record.get("usage") or {}
                    job["input_tokens"]+=usage.get("input_tokens",usage.get("prompt_tokens",0))
                    job["output_tokens"]+=usage.get("output_tokens",usage.get("completion_tokens",0))
                self.save(job)
            with pymupdf.open(source) as doc:
                job.update(status="running",message="全图盘点位号并记录出现位置")
                self.save(job)
                inventory=build_device_inventory(doc,[infos[n] for n in sorted(eligible)])
                inventory['view_sha256']=source_hash
                save_json(folder/'device_inventory.json',inventory)
                agent.inventory=inventory
                job['inventory_count']=len(inventory['devices'])
                job.update(message="按位号索引逐页核对符号、型号和遗漏设备")
                self.save(job)
                retries=[]
                for n in job["pages"]:
                    if job["cancel_requested"]:
                        raise StopJob("任务已取消")
                    job["current_page"]=n
                    self.save(job)
                    try:
                        record=agent.extract_page(doc[n-1],infos[n],source_hash,effort=job["reasoning_effort"])
                        records.append(record)
                        account(record)
                    except StopJob:
                        raise
                    except Exception as exc:
                        error={"page":n,"stage":"extract","message":str(exc)[:300]}
                        job["errors"].append(error)
                        if isinstance(exc,TransientVisionError):
                            retries.append(error)
                    job["completed_pages"]+=1
                    self.save(job)
                for error in retries:
                    if job["cancel_requested"]:
                        raise StopJob("任务已取消")
                    if limit_reached():
                        break
                    n=error["page"]
                    job.update(current_page=n,message=f"正在重试第 {n} 页")
                    self.save(job)
                    try:
                        record=agent.extract_page(doc[n-1],infos[n],source_hash,effort=job["reasoning_effort"])
                        records.append(record)
                        job["errors"].remove(error)
                        account(record)
                    except StopJob:
                        raise
                    except Exception as exc:
                        error["message"]="重试后仍未完成："+str(exc)[:260]
                conflicts=conflict_pages(records,infos)
                for page, issues in inventory_review_issues(inventory,records,infos).items():
                    conflicts.setdefault(page, []).extend(issues)
                review_candidates=[r for r in records if r.get("checks") or r["page"] in conflicts]
                def review_priority(record):
                    # A review cap must not let early-page terminal questions displace
                    # later repeated-template quantities and cross-page conflicts.
                    template=bool(infos[record['page']].extraction_diagnostics.get('range_annotations')) or any(
                        o.get('quantity_basis')=='repeated_template' for o in record['data']['devices'])
                    return (100 if template else 0)+(80 if record['page'] in conflicts else 0)+len(record.get('checks',[]))
                review_candidates.sort(key=lambda r:(-review_priority(r),r['page']))
                review_limit=vision.config.get("max_high_review_pages",6)
                job.update(status="reviewing",message="提高思考程度复核证据冲突和跨页疑点",review_candidates=len(review_candidates))
                self.save(job)
                for record in review_candidates[:review_limit]:
                    n=record["page"]
                    job["current_page"]=n
                    self.save(job)
                    try:
                        checked=agent.extract_page(doc[n-1],infos[n],source_hash,effort=job["review_reasoning_effort"],
                            prior=record["data"],issues=record.get("checks",[])+conflicts.get(n,[]))
                        account(checked)
                        records[records.index(record)]=checked
                        job["reviewed_pages"]+=1
                    except StopJob:
                        raise
                    except Exception as exc:
                        job["errors"].append({"page":n,"stage":"review","message":str(exc)[:300]})
                    self.save(job)
                job["deferred_review_pages"]=[r["page"] for r in review_candidates[review_limit:]]
            if job["cancel_requested"]:
                raise StopJob("任务已取消")
            if not records:
                raise RuntimeError("没有成功识别的页面；原 BOM 已保留")
            # Keep every unresolved cross-page conflict visible even if high review was capped.
            remaining=conflict_pages(records,infos)
            for page, issues in inventory_review_issues(inventory,records,infos).items():
                remaining.setdefault(page, []).extend(issues)
            for record in records:
                record["checks"]=list(dict.fromkeys(record.get("checks",[])+remaining.get(record["page"],[])))
            with self.guard:
                if job["cancel_requested"]:
                    raise StopJob("任务已取消")
                job.update(status="merging",message="按完整位号归并符号并保护人工成果")
                self.save(job)
            with store.lock:
                if job["project_id"] not in store.results:
                    raise StopJob("项目已删除，任务结果未写入")
                if store.repository and store.data_versions.get(job['project_id']) != job['input_data_version']:
                    raise StopJob('项目已由工程师更新；旧 AI 结果没有覆盖新版本，请重新运行')
                result=store.results[job["project_id"]]
                save_json(folder/"history"/f"before_ai_merge_{job['id']}.json",result.to_dict())
                draft=copy.deepcopy(result)
                job["merge"]=merge_independent_observations(draft,records)
                if independent_policy(draft):
                    snapshot={'policy':draft.summary['independent_analysis']['policy'],'created_at':time.time(),
                              'view_sha256':source_hash,'items':[i.to_dict() for i in draft.independent_bom]}
                    snapshot_path=folder/'independent_snapshot.json'
                    save_json(snapshot_path,snapshot)
                    draft.summary['independent_analysis']['snapshot_sha256']=hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                    draft.summary['independent_analysis']['snapshot_saved_at']=snapshot['created_at']
                draft.summary["ai_last_run"]={k:copy.deepcopy(job[k]) for k in ("id","model","reasoning_effort","review_reasoning_effort","api_calls","cache_hits","input_tokens","output_tokens","merge","pages","errors","deferred_review_pages")}
                draft.summary['ai_last_run'].update(agent_version=VERSION,successful_pages=sorted(r['page'] for r in records),
                    coverage='FULL_DOCUMENT' if {r['page'] for r in records}==eligible else 'PARTIAL_DOCUMENT')
                store.results[job["project_id"]]=draft
                try:
                    store.sync_and_persist(job["project_id"])
                except Exception:
                    if not store.repository or store.data_versions.get(job['project_id']) == job['input_data_version']:
                        store.results[job["project_id"]]=result
                    raise
            job.update(status="partial" if job["errors"] else "completed",current_page=None,
                       message="AI 识别结果已合并；型号、采购范围和数量请在审核清单确认")
        except StopJob as exc:
            job.update(status="cancelled" if job["cancel_requested"] else "interrupted",message=str(exc))
        except Exception as exc:
            job.update(status="failed",message=str(exc)[:300])
        finally:
            self.save(job)
            if store.repository:
                with store.repository.connection(job.get('requested_by')) as conn:
                    conn.execute("INSERT INTO ai_runs(id,job_id,project_id,input_data_version,drawing_version,model,model_config,prompt_version,rules_version,memory_versions,result_validation,usage,error) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(uuid.uuid4(),job['id'],job['project_id'],job.get('input_data_version'),job.get('drawing_version'),job.get('model'),json.dumps({'reasoning_effort':job.get('reasoning_effort'),'review_reasoning_effort':job.get('review_reasoning_effort')}),VERSION,'current','[]',json.dumps({'status':job['status'],'merge':job.get('merge')},ensure_ascii=False),json.dumps({'api_calls':job.get('api_calls'),'input_tokens':job.get('input_tokens'),'output_tokens':job.get('output_tokens')}),job.get('message') if job['status'] in ('failed','interrupted') else None))
            if context_token is not None: actor.reset(context_token)
