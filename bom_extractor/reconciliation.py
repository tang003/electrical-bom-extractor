"""Compare two independently persisted sources; never fill the schematic BOM here."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata


def norm(value):
    return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(value or "")).upper()).replace("−","-").replace("–","-")


def tag_key(value):
    value=norm(value)
    if value and not value.startswith(("-","=","+")):
        value="-"+value
    return value


def tag_members(value):
    """Expand reference membership only; the printed quantity is never inferred."""
    value=norm(value)
    result=[]
    prefix=""
    def sequence(base,start_text,end_text):
        start,end=int(start_text),int(end_text)
        if end<start or end-start>200:return None
        width=len(start_text) if start_text.startswith('0') and len(start_text)>1 else 0
        return [tag_key(base+str(n).zfill(width)) for n in range(start,end+1)]
    def same_prefix(left,right):
        if left==right:return True
        local=re.search(r'[A-Z]+$',left)
        return bool(local and not any(c in right for c in '=+') and local[0]==right.lstrip('-'))
    for piece in re.split(r"[,，、/;；]",value):
        if not piece:continue
        repeated=re.fullmatch(r'(.*?)([A-Z]+)(\d+)-([A-Z]+)(\d+)',piece)
        if repeated and repeated[2]==repeated[4]:
            prefix=repeated[1]+repeated[2]
            members=sequence(prefix,repeated[3],repeated[5])
            if members is None:return [tag_key(value)]
            result.extend(members)
            continue
        if prefix and re.fullmatch(r'\d+-\d+',piece):
            members=sequence(prefix,*piece.split('-'))
            if members is None:return [tag_key(value)]
            result.extend(members)
            continue
        if prefix:
            suffix_range = re.fullmatch(r'(\d+-)(\d+)[~～至到](\d+)', piece)
            if suffix_range:
                members = sequence(prefix + suffix_range[1], suffix_range[2], suffix_range[3])
                if members is None:return [tag_key(value)]
                result.extend(members)
                continue
            if re.fullmatch(r'\d+[A-Z]', piece):
                result.append(tag_key(prefix + piece))
                continue
        suffixed = re.fullmatch(r'(.*[A-Z])(\d+[A-Z])', piece)
        if suffixed:
            prefix = suffixed[1]
            result.append(tag_key(piece))
            continue
        letters=re.fullmatch(r'(.+?)(\d+)([A-Z])[~～至到](?:(\d+))?([A-Z])',piece)
        if letters and (not letters[4] or letters[4]==letters[2]):
            start,end=ord(letters[3]),ord(letters[5])
            if end<start:return [tag_key(value)]
            prefix=letters[1]
            result.extend(tag_key(prefix+letters[2]+chr(n)) for n in range(start,end+1))
            continue
        match=re.fullmatch(r"(.+?)(\d+)(?:[~～至到](?:(\D+))?(\d+))?",piece)
        if match and re.search(r'[A-Z]',match[1]):
            prefix=match[1]
            if match[3] and not same_prefix(prefix,match[3]):return [tag_key(value)]
            members=sequence(prefix,match[2],match[4] or match[2])
            if members is None:return [tag_key(value)]
            result.extend(members)
        elif prefix and re.fullmatch(r"\d+(?:[~～至到]\d+)?",piece):
            limits=re.split(r'[~～至到]',piece)
            members=sequence(prefix,limits[0],limits[-1])
            if members is None:return [tag_key(value)]
            result.extend(members)
        else:
            result.append(tag_key(piece))
    return list(dict.fromkeys(result))


def _keys(item):
    values=[item.terminal_strip_tag] if item.terminal_strip_tag else (item.tags or ([item.tag] if item.tag else []))
    members = (k for value in values for k in tag_members(value) if k)
    return list(dict.fromkeys((norm(item.location_code) + k if item.location_code and k.startswith("-") else k)
                              for k in members))


def _local(value):
    from .notation import local_tag
    return local_tag(value)


def model_known(value):
    return bool(value and not re.match(r"^(未提取|图纸未注|待核|待定)",value))


def supply_note(remark):
    """Keep explicit material supply statements verbatim for the buyer."""
    clauses = re.split(r'[，,；;。\n]+', str(remark or ''))
    identified = []
    for raw in clauses:
        clause = raw.strip()
        if not clause or re.search(r'图纸|资料|证明|服务|可提供|待提供|待定', clause):
            continue
        if (re.search(r'(?:甲供|乙供|客供)(?:件|设备|材料)?(?:$|[（(])', clause)
                or re.fullmatch(r'.{1,40}(?:提供|供货|自备|自购)(?:[）)])?', clause)):
            identified.append(clause)
    return '；'.join(identified)


def _fact(item):
    return {"id":item.id,"tag":item.tag,"name":item.name,"model":item.model,"raw_model":item.raw_model,
            "specification":item.specification,"brand":item.brand,"quantity":item.quantity,"unit":item.unit,
            "entity_role":item.entity_role,"parent_tag":item.parent_tag,"procurement_scope":item.procurement_scope,
            "remark":item.remark,"supply_note":supply_note(item.remark),
            "evidences":[e.to_dict() for e in item.evidences]}


def _signature_facts(facts):
    # Preserve previous decisions when only extra descriptive text was exposed.
    return [{k:v for k,v in fact.items() if k!='remark' and (k!='supply_note' or v)} for fact in facts]


# These terms describe an accessory in a parts-list description. A part number
# or a repeat of the same device tag alone is not evidence of this relationship.
_ACCESSORY_NAME = re.compile(
    r"(?:底座|辅助触点(?:模块)?|保护插件|保护罩|防护罩|前连接器|通讯卡|通信卡|扩展卡|安装支架|适配器|附件)$"
    r"|\b(?:relay socket|mounting base|auxiliary contact(?: block)?|protective cover|"
    r"front connector|communication card|mounting bracket|accessory)\b", re.I
)


def _split_accessory_rows(references, devices):
    """Identify a main row and explicitly named accessories for one placed device."""
    if len(devices) != 1 or len(references) < 2:
        return {}
    device = devices[0]
    placements = {norm(r.placement) for r in references}
    order_codes = {norm(r.model) for r in references if model_known(r.model)}
    if '' in placements or len(placements) != 1 or len(order_codes) != len(references):
        return {}
    if any(len(_keys(r)) != 1 for r in references):
        return {}
    main = [r for r in references if not _ACCESSORY_NAME.search(r.name.strip())]
    if len(main) != 1:
        return {}
    main_row = main[0]
    if model_known(device.model):
        if norm(main_row.model) != norm(device.model):
            return {}
    else:
        a, b = norm(main_row.name), norm(device.name)
        if min(len(a), len(b)) < 2 or not (a in b or b in a):
            return {}
    return {r.id: 'main' if r is main_row else 'accessory' for r in references}


def reconcile(result):
    snapshot=result.independent_bom
    references=result.declared_parts
    previous={r['id']:r for r in result.reconciliation}
    index=defaultdict(list);local=defaultdict(set)
    for item in snapshot:
        for key in _keys(item):
            index[key].append(item)
            local[_local(key)].add(key)
    references_by_tag=defaultdict(list)
    for item in references:
        for key in _keys(item):
            references_by_tag[key].append(item)
    used=set();rows=[]
    def append(independent,declared,status,issues,reasons):
        facts=[_fact(i) for i in independent];table=[_fact(i) for i in declared]
        notes = list(dict.fromkeys(f['supply_note'] for f in [*table, *facts] if f['supply_note']))
        if notes:
            issues.append('SUPPLY_SCOPE_REVIEW')
            reasons.append('图纸供货说明：'+'；'.join(notes)+'。请核对哪些物料由本次采购。')
            if status=='MATCH':
                status='DIFFERENCE'
        identity=json.dumps([sorted(i.id for i in independent),sorted(i.id for i in declared)],sort_keys=True)
        rid='reconcile_'+hashlib.sha256(identity.encode()).hexdigest()[:18]
        signature_body=[_signature_facts(facts),_signature_facts(table),status,issues]
        signature=hashlib.sha256(json.dumps(signature_body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        old=previous.get(rid,{})
        stable=old.get('signature')==signature
        rows.append({'id':rid,'signature':signature,'status':status,'issues':issues,'reasons':reasons,
                     'schematic_ids':[i.id for i in independent],'declared_ids':[i.id for i in declared],
                     'schematic':facts,'declared':table,'decision':old.get('decision','PENDING') if stable else 'PENDING',
                     'decision_note':old.get('decision_note','') if stable else '',
                     'decided_at':old.get('decided_at') if stable else None,
                     'previous_decision':old.get('decision') if not stable and old.get('decision') not in (None,'PENDING') else old.get('previous_decision')})
    for declared in references:
        keys=_keys(declared);found={};ambiguous=False;missing=[];component_role=None
        for key in keys:
            exact=index.get(key,[])
            from .notation import COMPONENT_TAG_RE
            if not key.startswith(('=','+')) and COMPONENT_TAG_RE.fullmatch(key):
                options=local.get(_local(key),set())
                if len(options)==1:exact=index[next(iter(options))]
                elif len(options)>1:
                    ambiguous=True
                    exact=[i for k in options for i in index[k]]
            if not exact:missing.append(key)
            siblings=references_by_tag[key]
            if len(siblings)>1 and len(keys)==1 and not ambiguous:
                component_role=_split_accessory_rows(siblings,exact).get(declared.id)
            if len(exact)>1 or (len(siblings)>1 and component_role is None):ambiguous=True
            for item in exact:found[item.id]=item;used.add(item.id)
        matches=list(found.values());issues=[];reasons=[]
        if not keys:
            status='UNLINKED_REFERENCE';reasons.append('原表未给出可对应的完整位号，尚未建立物理物料关联。')
        elif ambiguous:
            status='AMBIGUOUS';reasons.append('同位号对应多个范围、版本或组件，需人工建立对应关系。')
        elif component_role=='accessory':
            status='TABLE_ACCESSORY'
            reasons.append('部件表在同一放置位置另列附件及独立型号。可逐项补入采购明细，核对附件数量。')
        elif not matches:
            status='TABLE_ONLY';reasons.append('原表有记录，独立识别尚未找到原理图证据；可能是漏识别、备品或辅材。')
        else:
            if missing:issues.append('PARTIAL_TAG_COVERAGE');reasons.append('原表部分位号未找到原理图证据：'+'、'.join(missing))
            models={norm(i.model) for i in matches if model_known(i.model)}
            declared_model=norm(declared.model) if model_known(declared.model) else ''
            incomplete=len(models)==0 or any(not model_known(i.model) for i in matches)
            if incomplete or not declared_model:
                issues.append('MODEL_INCOMPLETE');reasons.append('至少一侧型号未确定；原表型号仅可作为待确认补充。')
            elif models!={declared_model}:
                if any('+' in model for model in [declared_model,*models]):
                    issues.append('ASSEMBLY_REVIEW');reasons.append('存在主件、附件或成套组合的表达差异，需确认组件关系。')
                else:issues.append('MODEL_MISMATCH');reasons.append('独立识别型号与原表型号不同。')
            if declared.quantity is None or any(i.quantity is None for i in matches):
                issues.append('QUANTITY_INCOMPLETE');reasons.append('至少一侧数量或长度未建立，未借用另一侧数字。')
            else:
                units={i.unit for i in [declared,*matches]}-{'','待核','待定',None}
                if len(units)>1:
                    issues.append('UNIT_REVIEW');reasons.append('两侧列出不同计量单位，数量未直接比较。')
                elif abs(sum(i.quantity for i in matches)-declared.quantity)>1e-8:
                    issues.append('QUANTITY_MISMATCH');reasons.append('独立物理数量与原表数量不同。')
            if declared.parent_tag or declared.entity_role=='assembly' or any(i.parent_tag or i.entity_role in {'contact','assembly'} for i in matches):
                issues.append('ASSEMBLY_REVIEW');reasons.append('含仅触点、所属组件或成套关系，需核对采购本体。')
            # Parameters and brand are compared only when both sources explicitly state them.
            specs={norm(i.specification) for i in matches if i.specification}
            if declared.specification and specs and specs!={norm(declared.specification)}:
                issues.append('PARAMETER_REVIEW');reasons.append('两侧保留的原文参数不同，请核对电压、电流、极数等条件。')
            brands={norm(i.brand) for i in matches if i.brand and i.brand not in {'待定','国产','未知'}}
            if declared.brand and declared.brand not in {'待定','国产','未知'} and brands and brands!={norm(declared.brand)}:
                issues.append('BRAND_REVIEW');reasons.append('两侧明确标注的制造商不同，请核对。')
            status='MATCH' if not issues else 'DIFFERENCE'
            if status=='MATCH':reasons.append('两侧已提取型号、数量和单位一致；仍需确认采购适用范围。')
        append(matches,[declared],status,list(dict.fromkeys(issues)),reasons)
    for item in snapshot:
        if item.id not in used:
            status='SCHEMATIC_ONLY' if result.summary.get('declared_reference_status')=='loaded' else 'REFERENCE_UNAVAILABLE'
            reason='独立原理图有记录，原表未建立对应；需核查是否漏列、范围不同或设备关系不清。' if status=='SCHEMATIC_ONLY' else '暂无可用图内物料表，保留独立识别结果。'
            append([item],[],status,[],[reason])
    result.reconciliation=rows
    result.summary['reconciliation']={'total':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
                                      'pending':sum(r['decision']=='PENDING' for r in rows),
                                      'independent_count':len(snapshot),'declared_count':len(references)}
    return rows
