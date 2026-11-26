"""Choose names from owned drawing labels, without treating a tag prefix as proof."""
from __future__ import annotations

import re

from .notation import device_code

DEVICE_CAPTION = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{0,18}(?:控制器|探测器|报警器|蜂鸣器|接触器|继电器|"
    r"断路器|保护器|变送器|传感器|互感器|变频器|驱动器|编码器|加热器|电源|电表|"
    r"空调|灯|按钮盒|按钮|开关|插座|模块|轴卡|电阻器?|电容器?|电抗器|变压器)"
)
FALLBACK_NAMES = {'电气元器件', '照明/加热元件', '电气元器件（类型待核）', '电阻（类型待核）'}

# Whole rating lines only: these values are never catalogue/order codes.
_NUMBER = r'\d+(?:[.,]\d+)?'
_RESISTANCE = rf'{_NUMBER}\s*(?:[kKmM]?\s*[ΩΩ]|(?:千|兆)?欧姆|(?:[kKmM]\s*)?ohms?)'
_POWER = rf'{_NUMBER}\s*(?:k?W|千?瓦)'
_SEPARATOR = r'[\s,，;；/、]*'
_RESISTOR_RATING = re.compile(rf'(?:{_RESISTANCE}(?:{_SEPARATOR}{_POWER})?|{_POWER}(?:{_SEPARATOR}{_RESISTANCE})?)', re.I)


def printed_device_caption(text):
    text = text.strip()
    if DEVICE_CAPTION.fullmatch(text):
        return text
    match = DEVICE_CAPTION.match(text)
    if match and re.search(r'电阻器?$', match[0]) and _RESISTOR_RATING.fullmatch(text[match.end():].strip()):
        return match[0]
    return ''


def resistor_parameters(text):
    text = text.strip()
    caption = printed_device_caption(text)
    if caption:
        text = text[len(caption):].strip()
    return text if _RESISTOR_RATING.fullmatch(text) else ''



def owned_caption(texts):
    """Only complete device captions count; purpose/rating/model lines stay notes."""
    captions = list(dict.fromkeys(caption for text in texts
                                 if (caption := printed_device_caption(text))))
    if not captions:
        return ''
    clean = lambda text: re.sub(r'([\u4e00-\u9fff])\1+', r'\1', text)
    longest = max(map(clean, captions), key=len)
    return longest if all(clean(text) in longest for text in captions) else ''


def owned_name(tag, texts):
    name = owned_caption(texts)
    if name:
        return name
    if device_code(tag) == 'R' and any(re.search(_RESISTANCE, resistor_parameters(text), re.I) for text in texts):
        return '电阻（类型待核）'
    if device_code(tag) == 'E' and any('反馈' in text for text in texts):
        return '反馈器件（类型待核）'
    return ''


def _protected(item, accepted):
    return (item.source_type == 'MANUAL_EDITED' or item.is_quantity_adjustment
            or item.quantity_status == 'confirmed' or set(item.tags or [item.tag]) & accepted
            or any(e.verified for e in item.evidences))


def _recover_labels(result, doc, eligible):
    """Rebind native drawing labels, using all printed tags as competing owners.

    The caller supplies the validated schematic-only document. No OCR/model
    requests, table answers, new devices or new quantities are introduced here.
    """
    from .classifier import extract_visual_spans
    from .models import EvidenceLocation
    from .notation import COMPONENT_TAG_RE, qualify_tag, with_component_anchors
    from .spatial_binding import device_captions, stacked_device_annotations, symbol_enclosures
    pages = {info.page_num: info for info in result.pages if info.page_type not in ('PART_LIST', 'COVER', 'CONTENTS')}
    targets = {}
    for item in eligible:
        for ev in item.evidences:
            if ev.role in ('main_circuit', 'coil', 'device', 'instrument_module', 'contact_or_symbol', 'main'):
                targets.setdefault(ev.page, {})[item.tag] = item
    for page_num, items in targets.items():
        if page_num not in pages or not 1 <= page_num <= len(doc):
            continue
        page, info = doc[page_num - 1], pages[page_num]
        spans = with_component_anchors(extract_visual_spans(page))
        spans = [s for s in spans if .02 * info.height < s['cy'] < .90 * info.height
                 and .02 * info.width < s['cx'] < .98 * info.width]
        anchors = [{**s, 'tag':qualify_tag(s['text'], info.location_code, info.function_code)}
                   for s in spans if COMPONENT_TAG_RE.fullmatch(s['text'].strip())]
        captions = device_captions(spans, anchors, symbol_enclosures(page))
        annotations = stacked_device_annotations(spans, anchors)
        for tag, item in items.items():
            seen = {(ev.page, ev.role, ev.raw_text, tuple(ev.bbox)) for ev in item.evidences}
            for role, labels in (('device_description', captions.get(tag, [])),
                                 ('device_annotation', annotations.get(tag, []))):
                for span in labels:
                    key = (page_num, role, span['text'], tuple(span['bbox']))
                    if key not in seen:
                        item.evidences.append(EvidenceLocation(page_num, span['bbox'], span['norm_bbox'],
                                                              raw_text=span['text'], role=role, device_tag=tag))
                        seen.add(key)


def repair_saved_component_names(result, doc=None):
    """Reapply common label rules to machine fallbacks, keeping human decisions.

    Pass a validated schematic document to recover missing literal name/parameter
    evidence. Existing groups, quantities, model codes and frozen facts stay intact.
    """
    if (result.summary.get('workflow_mode') == 'TABLE_EXTRACT'
            or result.summary.get('table_extraction', {}).get('policy') == 'parts-table-only-v1'
            or result.extraction_mode == 'PART_LIST_MODE'):
        return []
    accepted = {tag.strip() for review in result.review_items if review.status == 'ACCEPTED'
                for tag in re.split(r'[,，、]', review.tag) if tag.strip()}
    eligible = [item for item in result.itemized_bom
                if item.name in FALLBACK_NAMES and not _protected(item, accepted)]
    before = {item.id:(item.name, item.specification, len(item.evidences)) for item in eligible}
    if doc is not None:
        _recover_labels(result, doc, eligible)
    changed = []
    for item in eligible:
        labels = [e for e in item.evidences if e.role in ('device_description', 'device_annotation')
                  and e.device_tag == item.tag]
        texts = [e.raw_text for e in labels]
        replacement = owned_name(item.tag, texts)
        if not replacement and item.name == '照明/加热元件':
            replacement = '电气元器件（类型待核）'
        previous = item.name
        if replacement and replacement != previous:
            item.name = replacement
            if item.procurement_name == previous:
                item.procurement_name = replacement
            if replacement.endswith('（类型待核）'):
                item.needs_review = True
            item.observations.append({'source':'name_classification_repair', 'previous_name':previous,
                                      'name':replacement, 'evidence_texts':texts})
        if not item.specification and (device_code(item.tag) == 'R' or re.search(r'电阻器?$', item.name)):
            item.specification = '；'.join(dict.fromkeys(value for text in texts if (value := resistor_parameters(text))))
        if before[item.id] != (item.name, item.specification, len(item.evidences)):
            changed.append(item.id)
    details = {item.id:item for item in result.itemized_bom}
    for summary in result.aggregated_bom:
        members = [details[mid] for mid in summary.member_ids if mid in details]
        if (summary.name in FALLBACK_NAMES and not _protected(summary, accepted)
                and members and all(mid in details for mid in summary.member_ids)
                and all(item.id in changed for item in members)
                and len({item.name for item in members}) == 1):
            old = summary.name
            summary.name = members[0].name
            if summary.procurement_name == old:
                summary.procurement_name = summary.name
            if not summary.specification and len({item.specification for item in members}) == 1:
                summary.specification = members[0].specification
            summary.needs_review = summary.needs_review or any(item.needs_review for item in members)
    return changed
