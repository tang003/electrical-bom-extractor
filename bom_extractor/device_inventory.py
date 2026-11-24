"""Document-wide device index built from the isolated schematic view."""
from __future__ import annotations
import re

from .cad_import import cad_join_tag_fragments
from .classifier import extract_visual_spans
from .notation import COMPONENT_TAG_RE, device_code, is_connection_tag, qualify_tag, with_component_anchors
from .ocr_engine import DualChannelOCREngine


def build_device_inventory(doc, pages_info):
    """List each scoped designation once while retaining every printed location."""
    devices = {}
    for info in pages_info:
        page = doc[info.page_num - 1]
        native = extract_visual_spans(page)
        spans = DualChannelOCREngine.extract_dual_channel_spans(page, native, dpi=200)
        body = [span for span in spans
                if .02 * info.height < span['cy'] < .90 * info.height
                and .02 * info.width < span['cx'] < .98 * info.width]
        for span in with_component_anchors(cad_join_tag_fragments(body)):
            raw = span['text'].strip()
            if not COMPONENT_TAG_RE.fullmatch(raw) or is_connection_tag(raw):
                continue
            local = raw.upper() if raw.startswith(('-', '+', '=')) or raw.upper().startswith(('FR', 'HL', 'SQ')) else '-' + raw.upper()
            tag = qualify_tag(local, info.location_code, info.function_code)
            occurrence = {'page':info.page_num, 'bbox':span['norm_bbox'],
                          'text':span.get('source_text',raw)}
            found = devices.setdefault(tag, {'tag':tag, 'family':device_code(tag), 'occurrences':[]})
            if occurrence not in found['occurrences']:
                found['occurrences'].append(occurrence)
    def order(item):
        parts=re.split(r'(\d+)',item['tag'])
        return item['family'], tuple(int(part) if part.isdigit() else part for part in parts)

    return {'version':2, 'devices':sorted(devices.values(), key=order)}


def page_targets(inventory, page_number):
    """Give page recognition every known designation and its other locations."""
    return [{'tag':item['tag'], 'family':item['family'],
             'locations':item['occurrences']}
            for item in inventory['devices']
            if any(location['page'] == page_number for location in item['occurrences'])]


def inventory_review_issues(inventory, records, infos):
    """Audit recognition coverage by physical designation across all scanned pages."""
    from .ai_agent import scoped_tag

    scanned = {record['page'] for record in records}
    observed = {}
    for record in records:
        info = infos[record['page']]
        for item in record['data']['devices']:
            tag = scoped_tag(item.get('local_tag'), info)
            if tag:
                observed.setdefault(tag, set()).add(record['page'])

    issues = {}
    for item in inventory['devices']:
        expected = {location['page'] for location in item['occurrences']
                    if location['page'] in scanned
                    and infos[location['page']].page_type not in ('INSTALL_LAYOUT', 'CABINET_LAYOUT')}
        if not expected:
            continue
        missing = expected - observed.get(item['tag'], set())
        for page in missing:
            issues.setdefault(page, []).append(f"全图位号 {item['tag']} 在第 {page} 页尚未被 AI 确认，请按索引位置核对")
    return issues
