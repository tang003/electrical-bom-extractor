"""Procurement decisions stay separate from observations and catalogue suggestions."""
from __future__ import annotations
import hashlib
import re
from .models import ReviewItem, SourceType
from .components import model_state, components_pending

UNKNOWN_MODEL = ("未提取", "图纸未注", "待定")
RANGE = re.compile(r"([A-Z]{1,8})(\d+)\s*([-–~～至到])\s*([A-Z]{1,8})?(\d+)", re.I)
TEMPLATE_NOTE = re.compile(r'本图适用|共用此图|相同回路|典型回路|适用范围|TYPICAL\s+(?:FOR|CIRCUIT)|APPLIES\s+TO', re.I)


def range_annotations(spans, page_title=''):
    """A printed range is page-wide only when the drawing identifies the repeated circuit."""
    found = []
    for sp in spans:
        label = sp['text'].strip()
        explicit = bool(TEMPLATE_NOTE.search(label))
        match = RANGE.search(label) if explicit else RANGE.fullmatch(label)
        if not match or (match[4] and match[4].upper() != match[1].upper()):
            continue
        count = int(match[5]) - int(match[2]) + 1
        if not 1 < count <= 1000:
            continue
        box = sp['norm_bbox']
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        nearby = [s['text'].strip() for s in spans if s is not sp
                  and abs((s['norm_bbox'][0]+s['norm_bbox'][2])/2-cx) < .045
                  and abs((s['norm_bbox'][1]+s['norm_bbox'][3])/2-cy) < .052]
        page_wide = explicit or any(TEMPLATE_NOTE.search(s) or (
            len(s) >= 2 and s.casefold() in (page_title or '').casefold() and re.search(r'[A-Za-z\u4e00-\u9fff]',s)
        ) for s in nearby)
        # A bare dash also separates a connector and pin, or sections of a wire number.
        # Retain unambiguous range notation; a single local group never multiplies the page.
        padded = len(match[2]) == len(match[5]) and match[2].startswith('0')
        if match[3] == '-' and not (page_wide or match[4] or padded):
            continue
        found.append({'text':label, 'count':count, 'norm_bbox':box,
                      'scope':'page' if page_wide else 'group'})
    return found


def pending_review(reviews, kind, item, text):
    rid='rev_proc_'+hashlib.sha256(f'{kind}:{item.id}'.encode()).hexdigest()[:20]
    if not any(r.id==rid for r in reviews):
        reviews.append(ReviewItem(rid,'WARNING',kind,item.tag,text,text,
            '定位证据确认采购范围、组件关系或实际数量，不按符号出现次数计数。',item.primary_page,item.primary_bbox))


def flag_template_quantities(items, reviews, pages):
    """A tag on a repeated template is not proof of a single installed unit."""
    def page_template(annotation):
        if 'scope' in annotation:
            return annotation['scope']=='page'
        # Older snapshots stored wire and connector identifiers as page ranges.
        m=RANGE.fullmatch(annotation.get('text',''))
        return bool(m and annotation.get('count',0)>1 and
                    (m[4] or (len(m[2])==len(m[5]) and m[2].startswith('0'))))
    by_page={p.page_num:[a for a in p.extraction_diagnostics.get('range_annotations',[])
                         if page_template(a)] for p in pages}
    accepted={tag.strip().upper() for r in reviews if r.status=='ACCEPTED' for tag in re.split(r'[,，、]',r.tag) if tag.strip()}
    for item in items:
        if item.source_type==SourceType.MANUAL_EDITED.value or item.quantity_status=='confirmed' or any(t.upper() in accepted for t in (item.tags or [item.tag])):
            continue
        annotations=[r for ev in item.evidences for r in by_page.get(ev.page,[])]
        if not annotations or item.entity_role in ('terminal','cable'):
            continue
        item.quantity_candidates=sorted({1,*(a['count'] for a in annotations)})
        item.quantity=None
        item.quantity_status='needs_confirmation'
        item.quantity_basis='repeated_template'
        pending_review(reviews,'TEMPLATE_QUANTITY',item,
            '所在页有范围标注 '+', '.join(dict.fromkeys(a['text'] for a in annotations))+'；待确认是否为多套共用模板，数量暂空。')


def item_readiness(item):
    issues=[]
    state=model_state(item)
    if state['code'] != 'KNOWN':issues.append(state['label'])
    if components_pending(item):issues.append('本体/附件待核')
    if item.quantity is None or item.quantity_status in ('unknown','needs_confirmation'):issues.append('待核数量/长度')
    if item.procurement_scope=='unknown':issues.append('待核采购范围')
    if item.procurement_scope=='excluded':issues.append('不在采购范围')
    if item.parent_tag or item.entity_role in ('assembly','contact'):issues.append('待核组件关系')
    if item.needs_review:issues.append('待审核证据')
    if any(s.get('status')=='PENDING' for s in item.selection_suggestions):issues.append('有选型建议未采纳')
    return {'status':'NEEDS_REVIEW' if issues else 'REVIEWED', 'issues':issues, 'model_state':state}


def procurement_summary(result):
    rows=[x for x in result.itemized_bom if not x.is_quantity_adjustment]
    decisions=[item_readiness(x) for x in rows]
    return {'total_items':len(rows),'reviewed_items':sum(x['status']=='REVIEWED' for x in decisions),
        'quantity_pending':sum('待核数量/长度' in x['issues'] for x in decisions),
        'scope_pending':sum('待核采购范围' in x['issues'] for x in decisions),
        'selection_pending':sum('待选型' in x['issues'] for x in decisions),
        'model_association_pending':sum('型号关联待核' in x['issues'] for x in decisions),
        'components_pending':sum('本体/附件待核' in x['issues'] for x in decisions),
        'definition':'按位号明细统计采购审核状态，不是识别准确率。'}


def suggest(item, model, reason):
    if model and model!=item.model and not any(s.get('model')==model for s in item.selection_suggestions):
        item.selection_suggestions.append({'model':model,'reason':reason,'source':'enterprise_rule','status':'PENDING'})
        item.needs_review=True
