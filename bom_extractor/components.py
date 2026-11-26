"""Evidence-backed parts within one device; never infer an accessory quantity."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .notation import extract_model_strings, is_electrical_parameter
from .component_names import resistor_parameters

PART_LABELS = {'main':'本体','socket':'底座','contact_block':'触点模块','lamp_module':'灯模块',
               'accessory':'附件','unknown':'部件待核'}


def norm(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value or '')).upper().replace('–','-').replace('−','-')


class DeviceComponent(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['main','socket','contact_block','lamp_module','accessory','unknown']
    name: str = Field(max_length=240)
    kind_basis: Literal['printed','symbol','inferred','unknown'] = 'inferred'
    kind_span_ids: list[int] = Field(default_factory=list, max_length=60)
    owner_tag: str = Field(max_length=160)
    model_raw: str | None = Field(default=None, max_length=500)
    model_span_ids: list[int] = Field(default_factory=list, max_length=60)
    specification: str = Field(default='', max_length=1000)
    specification_span_ids: list[int] = Field(default_factory=list, max_length=60)
    relationship: Literal['separate_part','integral','unknown'] = 'unknown'
    relationship_basis: Literal['printed','symbol','inferred','unknown'] = 'unknown'
    relationship_span_ids: list[int] = Field(default_factory=list, max_length=60)
    bbox_normalized: list[float] = Field(min_length=4, max_length=4)
    reason: str = Field(default='', max_length=1200)

    @field_validator('bbox_normalized')
    @classmethod
    def valid_box(cls, value):
        if not all(math.isfinite(n) and 0 <= n <= 1 for n in value) or value[0] >= value[2] or value[1] >= value[3]:
            raise ValueError('部件坐标无效')
        return value


def contains(outer, inner, pad=.003):
    return all((inner[0] >= outer[0]-pad, inner[1] >= outer[1]-pad,
                inner[2] <= outer[2]+pad, inner[3] <= outer[3]+pad))


def literal_model(model, text):
    return norm(model) == norm(text) or norm(model) in {norm(v) for v in extract_model_strings(text)}


def model_citation(model, cited):
    if not cited:
        return False
    if all(literal_model(model,s['text']) for s in cited):
        return True
    # Exporters can split a printed order code across adjacent spans. Independent
    # complete order codes may not be joined into an invented kit identifier.
    for left,right in zip(cited,cited[1:]):
        a,b=left['norm_bbox'],right['norm_bbox']
        height=max(a[3]-a[1],b[3]-b[1])
        same=abs(a[1]-b[1])<=height*.5 and -height*.5<=b[0]-a[2]<=height*2
        wrapped=abs(a[0]-b[0])<=height*2 and -height*.2<=b[1]-a[3]<=height
        if not (same or wrapped): return False
        if extract_model_strings(left['text']) and extract_model_strings(right['text']) and not (
                left['text'].rstrip().endswith(('-', '/', '+', '.')) or right['text'].lstrip().startswith(('-', '/', '+', '.'))):
            return False
    return norm(''.join(s['text'] for s in cited)) == norm(model)


def validate_components(parts, owner_tag, spans, page, region, neighbors=()):
    """Every part has its own local literal evidence and the exact full owner tag."""
    spans = {s['id']:s for s in spans}
    result = []
    main_models = set()
    used_models = {}
    for raw in parts:
        part = DeviceComponent.model_validate(raw)
        if norm(part.owner_tag) != norm(owner_tag):
            raise ValueError('部件位号与目标位号不一致')
        if not contains(region, part.bbox_normalized):
            raise ValueError('部件坐标超出所属器件范围')
        evidence = []
        for role, ids in (('model',part.model_span_ids),('rating',part.specification_span_ids),('kind',part.kind_span_ids),
                          ('relationship',part.relationship_span_ids)):
            if len(ids) != len(set(ids)) or any(i not in spans for i in ids):
                raise ValueError('部件引用不存在或重复的原文编号')
            for i in ids:
                span = spans[i]
                box = span['norm_bbox']
                # The model often boxes just the symbol while the printed order
                # code sits below it. The trusted owner region and literal span
                # coordinates bound evidence; an AI-drawn sub-box must not veto
                # otherwise valid text from that same device.
                if span.get('page', page) != page or not contains(region,box):
                    raise ValueError('部件引用超出本页器件/部件范围')
                if any(norm(n['tag']) != norm(owner_tag) and n.get('page',page) == page and
                       contains(n['box'],box,0) for n in neighbors):
                    raise ValueError('部件引用了邻近器件的原文')
                if role == 'model':
                    identity = (page,i)
                    previous = used_models.get(identity)
                    current = (part.kind,norm(part.model_raw))
                    if previous is not None and previous != current:
                        raise ValueError('同一型号原文被分配给不同部件角色')
                    used_models[identity] = current
                evidence.append({'page':page,'raw_text':span['text'],'norm_bbox':box,
                                 'role':role,'span_id':i})
        if part.model_raw:
            cited = [spans[i] for i in part.model_span_ids]
            if (not cited or is_electrical_parameter(norm(part.model_raw)) or resistor_parameters(part.model_raw)
                    or not model_citation(part.model_raw,cited)):
                raise ValueError('部件型号缺少逐字原文证据，参数不能作为型号')
        elif part.model_span_ids:
            raise ValueError('部件型号为空但仍引用型号原文')
        if part.specification:
            if not part.specification_span_ids or norm(part.specification) not in norm(''.join(spans[i]['text'] for i in part.specification_span_ids)):
                raise ValueError('部件参数缺少原文证据')
        elif part.specification_span_ids:
            raise ValueError('部件参数为空但仍引用参数原文')
        if part.kind_basis == 'printed' and (not part.kind_span_ids or
                norm(part.name) not in norm(''.join(spans[i]['text'] for i in part.kind_span_ids))):
            # A literal catalogue code can be valid while the type/name is only
            # inferred. Keep that distinction instead of treating it as proof.
            part.kind_basis = 'inferred'
            part.kind_span_ids = []
            part.reason = (part.reason + '；' if part.reason else '') + '部件名称没有明文支持，类型按推断保留待核。'
            evidence = [ev for ev in evidence if ev['role'] != 'kind']
        if part.kind_basis != 'printed' and part.kind_span_ids:
            raise ValueError('推断部件类型不能伪装成明文名称')
        if part.relationship_basis == 'printed' and not part.relationship_span_ids:
            raise ValueError('部件关系声称为原图明文但没有引用')
        if part.relationship_basis == 'printed' and all(
                extract_model_strings(spans[i]['text']) and
                norm(spans[i]['text']) in {norm(code) for code in extract_model_strings(spans[i]['text'])}
                for i in part.relationship_span_ids):
            part.relationship_basis = 'inferred'
            part.relationship_span_ids = []
            part.reason = (part.reason + '；' if part.reason else '') + '订货号只证明型号，部件关系按推断保留待核。'
            evidence = [ev for ev in evidence if ev['role'] != 'relationship']
        # Use verified field positions to include labels outside the symbol box.
        # Relationship notes can refer to another part and do not enlarge it.
        boxes = [part.bbox_normalized] + [ev['norm_bbox'] for ev in evidence if ev['role'] != 'relationship']
        part.bbox_normalized = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                max(b[2] for b in boxes), max(b[3] for b in boxes)]
        if part.kind == 'main' and part.model_raw:
            main_models.add(norm(part.model_raw))
        key = hashlib.sha256(json.dumps([norm(owner_tag),part.kind,norm(part.model_raw),part.relationship],ensure_ascii=False).encode()).hexdigest()[:20]
        result.append({'id':key,'owner_tag':owner_tag,'kind':part.kind,'name':part.name,
                       'model':part.model_raw or '', 'specification':part.specification,
                       'kind_basis':part.kind_basis,
                       'relationship':part.relationship,'relationship_basis':part.relationship_basis,
                       'bbox_normalized':part.bbox_normalized,
                       'reason':part.reason,'evidences':evidence,'status':'PENDING',
                       'quantity':None,'unit':'个'})
    if len(main_models) > 1:
        raise ValueError('同一器件存在不同本体型号，需保留型号冲突')
    return result


def merge_parts(existing, incoming, confirmed=False):
    result = copy.deepcopy(existing)
    if confirmed:
        main_models = {norm(p.get('model')) for p in incoming if p['kind']=='main' and p.get('model')}
        for old in result:
            if main_models and old.get('kind')=='main' and norm(old.get('model')) not in main_models:
                old['status']='SUPERSEDED'
    indexed = {part['id']:part for part in result}
    for part in incoming:
        if part['id'] not in indexed:
            saved=copy.deepcopy(part)
            if confirmed: saved['status']='CONFIRMED'
            result.append(saved);indexed[saved['id']]=saved
        else:
            saved=indexed[part['id']]
            if confirmed: saved['status']='CONFIRMED'
            for ev in part.get('evidences',[]):
                if ev not in saved.setdefault('evidences',[]): saved['evidences'].append(copy.deepcopy(ev))
    return result


def model_state(item):
    missing = not item.model or item.model.startswith(('未提取','图纸未注','待定','待核','未知'))
    if not missing:
        return {'code':'KNOWN','label':'已读到型号'}
    candidates = {norm(code) for e in item.evidences if e.role in ('model_text','model_candidate')
                  for code in extract_model_strings(e.raw_text)}
    candidates.update(norm(p.get('model')) for p in item.component_parts if p.get('model'))
    if candidates:
        return {'code':'ASSOCIATION_PENDING','label':'型号关联待核'}
    return {'code':'MISSING','label':'待选型'}


def components_pending(item):
    return any(p.get('kind') != 'main' and not p.get('materialized_item_id') and
               (p.get('status') != 'CONFIRMED' or p.get('relationship') != 'integral')
               for p in item.component_parts)
