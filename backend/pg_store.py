"""PostgreSQL persistence. Business state and revisions commit in one transaction."""
from __future__ import annotations

import hashlib
import json
import os
import mimetypes
import re
from datetime import datetime,timezone
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4,uuid5,NAMESPACE_URL

import psycopg
from psycopg.rows import dict_row
from .auth import request_id


class Conflict(Exception):
    pass


class PgRepository:
    def __init__(self, dsn: str):
        self.dsn = dsn

    @contextmanager
    def connection(self, actor_id=None):
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.transaction():
                conn.execute("SELECT set_config('app.user_id', %s, true)", (str(actor_id or ''),))
                yield conn

    def migrate_schema(self):
        sql = (Path(__file__).parent / 'schema.sql').read_text(encoding='utf-8')
        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql)

    def list_project_snapshots(self, actor_id):
        with self.connection(actor_id) as conn:
            return conn.execute("SELECT id,snapshot,source_path,data_version,owner_id,state FROM projects WHERE state='active' ORDER BY created_at").fetchall()

    def visible_ids(self, actor_id, include_trash=False):
        with self.connection(actor_id) as conn:
            return {r['id'] for r in conn.execute("SELECT id FROM projects WHERE state='active' OR %s", (include_trash,)).fetchall()}

    def version(self, project_id, actor_id):
        with self.connection(actor_id) as conn:
            row = conn.execute("SELECT data_version FROM projects WHERE id=%s", (project_id,)).fetchone()
            if not row: raise KeyError(project_id)
            return row['data_version']

    def save_result(self, result, actor_id, *, expected_version=None, source='ui', summary='保存项目', source_path=None, owner_id=None):
        from bom_extractor.material_scope import exclude_estimated_wires
        exclude_estimated_wires(result)
        data = result.to_dict()
        project_id = result.project_id
        with self.connection(actor_id) as conn:
            row = conn.execute("SELECT data_version,owner_id,current_drawing_version FROM projects WHERE id=%s FOR UPDATE", (project_id,)).fetchone()
            if row:
                version = row['data_version']
                if expected_version is not None and version != expected_version: raise Conflict(f'项目版本已从 {expected_version} 更新至 {version}')
                version += 1
                drawing_version = row['current_drawing_version']
                conn.execute("UPDATE projects SET snapshot=%s,data_version=%s,name=%s,source_path=COALESCE(%s,source_path),updated_at=now() WHERE id=%s", (json.dumps(data,ensure_ascii=False),version,result.project_name,source_path,project_id))
            else:
                version = 1
                drawing_version = 1
                actual_owner = owner_id or actor_id
                conn.execute("INSERT INTO projects(id,owner_id,name,snapshot,source_path) VALUES(%s,%s,%s,%s,%s)", (project_id,actual_owner,result.project_name,json.dumps(data,ensure_ascii=False),source_path))
                conn.execute("INSERT INTO drawing_versions(project_id,version,sha256,page_count,workflow_mode,raw_snapshot) VALUES(%s,1,%s,%s,%s,%s)", (project_id,None,result.page_count,result.summary.get('workflow_mode'),json.dumps(data,ensure_ascii=False)))
            if source_path and Path(source_path).is_file():
                view_file=Path(source_path)
                original_candidate=view_file.parent/f'source.{result.source_format}'
                source_file=original_candidate if original_candidate.is_file() else view_file
                digest=hashlib.sha256(source_file.read_bytes()).hexdigest()
                if not conn.execute("SELECT 1 FROM project_files WHERE project_id=%s AND purpose='source' AND sha256=%s",(project_id,digest)).fetchone():
                    conn.execute("INSERT INTO project_files(id,project_id,drawing_version,purpose,storage_key,original_name,mime,byte_size,sha256) VALUES(%s,%s,%s,'source',%s,%s,%s,%s,%s)",(uuid4(),project_id,drawing_version,str(source_file),result.source_filename or source_file.name,mimetypes.guess_type(source_file.name)[0],source_file.stat().st_size,digest))
                conn.execute('UPDATE drawing_versions SET sha256=COALESCE(sha256,%s) WHERE project_id=%s AND version=%s',(digest,project_id,drawing_version))
                if view_file!=source_file and not conn.execute("SELECT 1 FROM project_files WHERE project_id=%s AND purpose='analysis_view'",(project_id,)).fetchone():
                    conn.execute("INSERT INTO project_files(id,project_id,drawing_version,purpose,storage_key,original_name,mime,byte_size,sha256) VALUES(%s,%s,%s,'analysis_view',%s,%s,%s,%s,%s)",(uuid4(),project_id,drawing_version,str(view_file),view_file.name,'application/pdf',view_file.stat().st_size,hashlib.sha256(view_file.read_bytes()).hexdigest()))
            for purpose,key in (('export_excel','excel_path'),('export_json','json_path')):
                output=result.summary.get(key)
                if not output or not Path(output).is_file() or (purpose=='export_excel' and result.summary.get('export_warning')):continue
                file=Path(output)
                metadata=(str(file),file.name,mimetypes.guess_type(file.name)[0],file.stat().st_size,hashlib.sha256(file.read_bytes()).hexdigest(),version)
                existing=conn.execute('SELECT id FROM project_files WHERE project_id=%s AND purpose=%s',(project_id,purpose)).fetchone()
                if existing:
                    conn.execute('UPDATE project_files SET storage_key=%s,original_name=%s,mime=%s,byte_size=%s,sha256=%s,export_data_version=%s,state=%s WHERE id=%s',(*metadata,'ready',existing['id']))
                else:
                    conn.execute('INSERT INTO project_files(id,project_id,drawing_version,purpose,storage_key,original_name,mime,byte_size,sha256,export_data_version) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',(uuid4(),project_id,drawing_version,purpose,*metadata))
            prior_reviews={r['id']:r for r in conn.execute('SELECT id,status,handled_by,handled_at FROM review_items WHERE project_id=%s',(project_id,)).fetchall()}
            conn.execute("DELETE FROM review_items WHERE project_id=%s", (project_id,))
            conn.execute("DELETE FROM bom_evidence WHERE project_id=%s", (project_id,))
            conn.execute("DELETE FROM bom_items WHERE project_id=%s", (project_id,))
            for collection in ('itemized_bom','filtered_items','independent_bom','declared_parts'):
                for item in getattr(result,collection):
                    raw = item.to_dict()
                    conn.execute("INSERT INTO bom_items(project_id,id,drawing_version,collection,seq_no,tag,name,model,brand,quantity,unit,procurement_scope,source_type,review_state,row_version,payload) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (project_id,item.id,drawing_version,collection,item.seq_no,item.tag,item.name,item.model,item.brand,item.quantity,item.unit,item.procurement_scope,item.source_type,json.dumps({'needs_review':item.needs_review,'review_note':item.review_note},ensure_ascii=False),version,json.dumps(raw,ensure_ascii=False)))
                    for ev in item.evidences:
                        conn.execute("INSERT INTO bom_evidence(id,project_id,item_id,item_collection,drawing_version,page,bbox,norm_bbox,raw_text,verified,payload) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (uuid4(),project_id,item.id,collection,drawing_version,ev.page,json.dumps(ev.bbox),json.dumps(ev.norm_bbox),ev.raw_text,ev.verified,json.dumps(ev.to_dict(),ensure_ascii=False)))
            for review in result.review_items:
                proposed=review.proposed_data or {}
                target_id=proposed.get('target_id') or proposed.get('added_item_id')
                candidates=[(collection,item.id) for collection in ('itemized_bom','filtered_items','independent_bom','declared_parts') for item in getattr(result,collection) if (item.id==target_id if target_id else review.tag in (item.tags or [item.tag]))]
                item_collection,item_id=candidates[0] if len(candidates)==1 else (None,None)
                prior=prior_reviews.get(review.id)
                handled_by=(prior['handled_by'] if prior and prior['status']==review.status else actor_id) if review.status!='PENDING' else None
                handled_at=(prior['handled_at'] if prior and prior['status']==review.status else datetime.now(timezone.utc)) if review.status!='PENDING' else None
                conn.execute("INSERT INTO review_items(project_id,id,item_id,item_collection,issue_type,severity,status,proposed_data,evidence,payload,handled_by,handled_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (project_id,review.id,item_id,item_collection,review.issue_type,review.severity,review.status,json.dumps(review.proposed_data,ensure_ascii=False),json.dumps({'page':review.page,'norm_bbox':review.norm_bbox},ensure_ascii=False),json.dumps(review.to_dict(),ensure_ascii=False),handled_by,handled_at))
                if review.issue_type=='AI_FIELD_CLASSIFICATION' and review.status=='ACCEPTED' and review.proposed_data:
                    memory_id=uuid5(NAMESPACE_URL,f'{project_id}:{review.id}')
                    conn.execute("INSERT INTO ai_memories(id,project_id,kind,content,applicability,source_review_id,confirmed_by,confirmed_data_version,status) VALUES(%s,%s,'field_classification',%s,%s,%s,%s,%s,'active') ON CONFLICT(id) DO UPDATE SET content=excluded.content,source_review_id=excluded.source_review_id,confirmed_by=excluded.confirmed_by,confirmed_data_version=excluded.confirmed_data_version,status='active',version=ai_memories.version+1 WHERE ai_memories.content IS DISTINCT FROM excluded.content OR ai_memories.status<>'active' OR ai_memories.source_review_id IS NULL",(memory_id,project_id,json.dumps(review.proposed_data,ensure_ascii=False),json.dumps({'tag':review.tag,'page':review.page},ensure_ascii=False),review.id,actor_id,version))
                elif review.issue_type=='AI_FIELD_CLASSIFICATION':
                    memory_id=uuid5(NAMESPACE_URL,f'{project_id}:{review.id}')
                    conn.execute("UPDATE ai_memories SET status='inactive',invalidation=%s WHERE id=%s",(json.dumps({'reason':'review_reset_or_ignored','version':version}),memory_id))
            conn.execute("INSERT INTO project_revisions(project_id,version,actor_id,source,summary,snapshot) VALUES(%s,%s,%s,%s,%s,%s)", (project_id,version,actor_id,source,summary,json.dumps(data,ensure_ascii=False)))
            conn.execute("INSERT INTO audit_events(actor_id,project_id,action,request_id,summary) VALUES(%s,%s,%s,%s,%s)", (actor_id,project_id,source,request_id.get(),summary))
            return version

    def migrate_legacy(self, output_dir, admin_id):
        from bom_extractor.models import ExtractionResult
        from bom_extractor.material_scope import exclude_estimated_wires
        from bom_extractor.table_workflow import table_policy
        from bom_extractor.exporter import export_to_excel,export_to_json
        migrated=[]
        for folder in sorted(Path(output_dir).iterdir()):
            if not folder.is_dir(): continue
            state=folder/'project_state.json'
            if not state.exists():
                candidates=sorted(folder.glob('*_提取结果.json'))
                if not candidates: continue
                state=candidates[0]
            result=ExtractionResult.from_dict(json.loads(state.read_text(encoding='utf-8')))
            exclude_estimated_wires(result)
            stem=re.sub(r'[^\w .-]','_',Path(result.pdf_filename).stem) or result.project_id
            suffix='部件表提取清单_未经原理图核对' if table_policy(result) else '标准BOM清单'
            result.summary['excel_path']=str((Path(output_dir)/result.project_id/f'{stem}_{suffix}.xlsx').resolve())
            result.summary['json_path']=str((Path(output_dir)/result.project_id/f'{stem}_提取结果.json').resolve())
            with self.connection(admin_id) as conn:
                if conn.execute('SELECT 1 FROM projects WHERE id=%s',(result.project_id,)).fetchone(): continue
            Path(result.summary['excel_path']).parent.mkdir(parents=True,exist_ok=True)
            export_to_excel(result,Path(result.summary['excel_path']))
            export_to_json(result,Path(result.summary['json_path']))
            pdfs=sorted(folder.glob('*.pdf'))
            source=next((p for p in pdfs if p.name=='source.pdf'),pdfs[0] if pdfs else None)
            if source is None:
                reference=Path(output_dir).parent/'参考文件'/Path(result.pdf_filename).name
                if reference.is_file(): source=reference
            self.save_result(result,admin_id,source='migration',summary='迁移本地项目',source_path=str(source) if source else None,owner_id=admin_id)
            migrated.append(result.project_id)
        return migrated

    def move_state(self, project_id, actor_id, state):
        with self.connection(actor_id) as conn:
            row=conn.execute('SELECT data_version,snapshot FROM projects WHERE id=%s FOR UPDATE',(project_id,)).fetchone()
            if not row: raise KeyError(project_id)
            version=row['data_version']+1
            conn.execute('UPDATE projects SET state=%s,data_version=%s,updated_at=now() WHERE id=%s',(state,version,project_id))
            conn.execute('INSERT INTO project_revisions(project_id,version,actor_id,source,summary,snapshot) VALUES(%s,%s,%s,%s,%s,%s)',(project_id,version,actor_id,state,state,json.dumps(row['snapshot'],ensure_ascii=False)))
            conn.execute('INSERT INTO audit_events(actor_id,project_id,action,request_id,summary) VALUES(%s,%s,%s,%s,%s)',(actor_id,project_id,state,request_id.get(),state))

    def transfer(self, project_id, actor_id, new_owner):
        with self.connection(actor_id) as conn:
            row=conn.execute('SELECT data_version,snapshot FROM projects WHERE id=%s FOR UPDATE',(project_id,)).fetchone()
            if not row: raise KeyError(project_id)
            version=row['data_version']+1
            conn.execute('UPDATE projects SET owner_id=%s,data_version=%s,updated_at=now() WHERE id=%s',(new_owner,version,project_id))
            conn.execute('INSERT INTO project_revisions(project_id,version,actor_id,source,summary,snapshot) VALUES(%s,%s,%s,%s,%s,%s)',(project_id,version,actor_id,'transfer','转交项目',json.dumps(row['snapshot'],ensure_ascii=False)))
            conn.execute('INSERT INTO audit_events(actor_id,project_id,action,request_id,summary) VALUES(%s,%s,%s,%s,%s)',(actor_id,project_id,'transfer',request_id.get(),'转交项目'))

    def revisions(self, project_id, actor_id):
        with self.connection(actor_id) as conn:
            return conn.execute('SELECT version,actor_id,source,summary,created_at FROM project_revisions WHERE project_id=%s ORDER BY version DESC',(project_id,)).fetchall()

    def restore_revision(self, project_id, actor_id, version, expected_version):
        with self.connection(actor_id) as conn:
            row=conn.execute('SELECT snapshot FROM project_revisions WHERE project_id=%s AND version=%s',(project_id,version)).fetchone()
            if not row: raise KeyError(version)
        from bom_extractor.models import ExtractionResult
        result=ExtractionResult.from_dict(row['snapshot'])
        return self.save_result(result,actor_id,expected_version=expected_version,source='restore',summary=f'恢复版本 {version}')
