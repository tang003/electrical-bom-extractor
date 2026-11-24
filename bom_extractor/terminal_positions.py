"""Count individually drawn terminal positions across segments and sheets."""
from __future__ import annotations

import math
import re
import copy
from collections import defaultdict
from statistics import median

from .notation import qualify_tag
from .models import BOMItem, EvidenceLocation, ReviewItem, SourceType


TERMINAL_MODEL = re.compile(r'PT\s*\d+(?:[.,]\d+)?(?:\s*-\s*PE)?', re.I)
TERMINAL_TAG = re.compile(r'-(?:X|XT)\d+', re.I)
UNPREFIXED_TERMINAL_TAG = re.compile(r'(?:X|XT)\d+', re.I)
INLINE_TERMINAL = re.compile(r'^\s*(\d{1,3})\s+(PT\s*\d+(?:[.,]\d+)?(?:\s*-\s*PE)?)\s*$', re.I)


def _inline_annotation_parts(span):
    """Separate CAD text such as '1 PT 2,5' without losing its source box."""
    match = INLINE_TERMINAL.fullmatch(span['text'].strip())
    if not match:
        return []
    number, model = match.groups()
    x0, y0, x1, y1 = span['bbox']
    n0, n1, n2, n3 = span['norm_bbox']
    if x1 - x0 > y1 - y0:
        fraction = (len(number) + .5) / len(span['text'].strip())
        split_x = x0 + (x1 - x0) * fraction
        split_nx = n0 + (n2 - n0) * fraction
        boxes = [([x0, y0, split_x, y1], [n0, n1, split_nx, n3]),
                 ([split_x, y0, x1, y1], [split_nx, n1, n2, n3])]
    else:
        boxes = [(span['bbox'], span['norm_bbox'])] * 2
    result = []
    for text, (box, norm_box) in zip((number, model), boxes):
        result.append({**span, 'text': text, 'bbox': box, 'norm_bbox': norm_box,
                       'cx': (box[0] + box[2]) / 2, 'cy': (box[1] + box[3]) / 2})
    return result


def vector_terminal_symbols(page):
    """Locate small closed circular terminal marks in the PDF vector layer."""
    symbols = []
    seen = set()
    width, height = page.rect.width, page.rect.height
    for path in page.get_drawings():
        box = path['rect']
        if not (2 <= box.width <= 10 and 2 <= box.height <= 10 and
                .65 <= box.width / box.height <= 1.5 and
                len(path['items']) >= 4 and all(part[0] == 'c' for part in path['items'])):
            continue
        bbox = [box.x0, box.y0, box.x1, box.y1]
        key = tuple(round(value, 2) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        symbols.append({'cx': (box.x0 + box.x1) / 2, 'cy': (box.y0 + box.y1) / 2,
                        'bbox': bbox,
                        'norm_bbox': [box.x0 / width, box.y0 / height,
                                      box.x1 / width, box.y1 / height]})
    return symbols


def terminal_positions(pages_info, page_spans_map, page_symbols_map=None):
    """Return visible positions keyed by the complete strip designation.

    A position is a numbered symbol with its own nearby printed terminal type.
    Its page and coordinates define the occurrence. Printed numbers are kept for
    continuity checks but never used as the quantity key.
    """
    positions = {}
    excluded = {'COVER', 'CONTENTS', 'PART_LIST', 'INSTALL_LAYOUT', 'CABINET_LAYOUT'}
    explicit_series = set()
    for page_info in pages_info:
        for span in page_spans_map.get(page_info.page_num, []):
            if TERMINAL_TAG.fullmatch(span['text'].strip()):
                explicit_series.add(qualify_tag(span['text'].strip().upper(),
                                                page_info.location_code, page_info.function_code))
    for info in pages_info:
        if info.page_type in excluded:
            continue
        source_spans = page_spans_map.get(info.page_num, [])
        spans = [*source_spans, *(part for span in source_spans for part in _inline_annotation_parts(span))]
        models = [span for span in spans if TERMINAL_MODEL.fullmatch(span['text'].strip())]
        symbols = (page_symbols_map or {}).get(info.page_num, [])
        def nearest_symbol(number):
            candidates = [symbol for symbol in symbols if math.hypot(
                symbol['cx'] - number['cx'], symbol['cy'] - number['cy']) <= 12]
            if not candidates:
                return None
            adjacent_models = nearby_models(number)
            model = min(adjacent_models, key=lambda entry: math.hypot(
                entry['cx'] - number['cx'], entry['cy'] - number['cy']), default=None)
            # Two-digit labels reach toward the next terminal. The model text
            # shares the symbol's x coordinate, so use it to disambiguate.
            return min(candidates, key=lambda symbol: (
                abs(symbol['cx'] - model['cx']) if model else abs(symbol['cx'] - number['cx']),
                math.hypot(symbol['cx'] - number['cx'], symbol['cy'] - number['cy'])))

        def nearby_models(number):
            return [model for model in models if abs(model['cx'] - number['cx']) <= 14
                    and math.hypot(model['cx'] - number['cx'], model['cy'] - number['cy']) <= 26]

        numbers = [span for span in spans if span['text'].strip().isdigit()
                   and 1 <= int(span['text'].strip()) <= 1000
                   and (nearby_models(span) or nearest_symbol(span))]
        headers = [span for span in spans if TERMINAL_TAG.fullmatch(span['text'].strip())]
        explicit_tags = {header['text'].strip().upper() for header in headers}
        bare_groups = []
        for candidate in spans:
            raw_tag = candidate['text'].strip()
            if not UNPREFIXED_TERMINAL_TAG.fullmatch(raw_tag) or '-' + raw_tag.upper() in explicit_tags:
                continue
            # Bare X labels also occur on connectors and layout legends. Require a
            # numbered, typed terminal symbol immediately after the label.
            starts = [number for number in numbers
                      if ((0 <= number['cx']-candidate['cx'] <= 40 and abs(number['cy']-candidate['cy']) <= 12)
                          or (abs(number['cx']-candidate['cx']) <= 12 and 0 <= number['cy']-candidate['cy'] <= 40))
                      and nearby_models(number) and (page_symbols_map is None or nearest_symbol(number))]
            if not starts:
                continue
            first = min(starts, key=lambda number: math.hypot(
                number['cx']-candidate['cx'], number['cy']-candidate['cy']))
            horizontal = first['cx']-candidate['cx'] >= abs(first['cy']-candidate['cy'])
            same_run = [int(number['text'].strip()) for number in numbers
                        if (abs(number['cy']-first['cy']) <= 12 and number['cx'] >= first['cx'] if horizontal
                            else abs(number['cx']-first['cx']) <= 12 and number['cy'] >= first['cy'])
                        and nearby_models(number) and (page_symbols_map is None or nearest_symbol(number))]
            first_number = int(first['text'].strip())
            run_numbers = set(same_run)
            last_number = first_number
            while last_number + 1 in run_numbers:
                last_number += 1
            complete_tag = qualify_tag('-' + raw_tag.upper(), info.location_code, info.function_code)
            match = re.search(r'-(X|XT)(\d+)$', complete_tag)
            neighbors = []
            if match:
                prefix, family, index = complete_tag[:match.start()], match[1], int(match[2])
                for adjacent in (index - 1, index + 1):
                    neighbor = f'{prefix}-{family}{adjacent}'
                    if adjacent > 0 and neighbor in explicit_series:
                        neighbors.append(neighbor)
            bare_groups.append((complete_tag, first_number, last_number, {**candidate,
                'text':'-' + raw_tag.upper(), 'header_source':'unprefixed_contextual',
                'header_raw_text':raw_tag, 'sequence_neighbors':neighbors}))
        accepted_max = {tag:max(entry['number'] for entry in entries) for tag,entries in positions.items()}
        pending = list(bare_groups)
        while pending:
            accepted = []
            for group in pending:
                tag, first_number, last_number, header = group
                if first_number == 1 or first_number == accepted_max.get(tag, -1) + 1:
                    if not any(existing['bbox'] == header['bbox'] for existing in headers):
                        headers.append(header)
                    accepted_max[tag] = max(accepted_max.get(tag, 0), last_number)
                    accepted.append(group)
            if not accepted:
                break
            pending = [group for group in pending if group not in accepted]
        if not headers:
            continue
        axes = {}
        for header in headers:
            near = sorted(numbers, key=lambda entry: math.hypot(
                entry['cx'] - header['cx'], entry['cy'] - header['cy']))
            if not near:
                continue
            first = near[0]
            second = next((entry for entry in near[1:] if 6 <= math.hypot(
                entry['cx'] - first['cx'], entry['cy'] - first['cy']) <= 80), None)
            vx, vy = ((second['cx'] - first['cx'], second['cy'] - first['cy']) if second else
                      (first['cx'] - header['cx'], first['cy'] - header['cy']))
            length = math.hypot(vx, vy)
            if 3 <= length <= 80:
                axes[id(header)] = (vx / length, vy / length)
        for number in numbers:
            raw = number['text'].strip()
            nearby = nearby_models(number)
            ranked = []
            diagonal = []
            for header in headers:
                dx = number['cx'] - header['cx']
                dy = number['cy'] - header['cy']
                horizontal = 0 <= dx <= info.width * .75 and abs(dy) <= 12
                vertical = abs(dx) <= 12 and 0 <= dy <= info.height * .9
                if horizontal or vertical:
                    ranked.append((dx + abs(dy) * 2 if horizontal else abs(dx) + dy / 10, header))
                    continue
                axis = axes.get(id(header))
                if axis is not None and abs(axis[0]) >= .2 and abs(axis[1]) >= .2:
                    forward = dx * axis[0] + dy * axis[1]
                    lateral = abs(dx * axis[1] - dy * axis[0])
                    if -3 <= forward <= math.hypot(info.width, info.height) and lateral <= 12:
                        diagonal.append((lateral * 5 + max(0, forward) * .02, header))
            if not ranked:
                ranked = diagonal
            if not ranked:
                continue
            header = min(ranked, key=lambda entry: entry[0])[1]
            model = min(nearby, key=lambda entry: math.hypot(
                entry['cx'] - number['cx'], entry['cy'] - number['cy']), default=None)
            symbol = nearest_symbol(number)
            tag = qualify_tag(header['text'].strip().upper(), info.location_code, info.function_code)
            position = {'page':info.page_num, 'number':int(raw),
                        'model':re.sub(r'\s+', '', model['text'].strip().upper()).replace(',', '.') if model else '',
                        'model_source':'explicit' if model else 'unbound',
                        'header_bbox':header['bbox'],
                        'header_norm_bbox':header['norm_bbox'],
                        'header_source':header.get('header_source', 'explicit'),
                        'header_raw_text':header.get('header_raw_text', header['text'].strip()),
                        'sequence_neighbors':header.get('sequence_neighbors', []),
                        'bbox':number['bbox'], 'norm_bbox':number['norm_bbox'],
                        'model_bbox':model['bbox'] if model else None,
                        'model_norm_bbox':model['norm_bbox'] if model else None,
                        'model_page':info.page_num if model else None,
                        'symbol_bbox':symbol['bbox'] if symbol else None,
                        'symbol_norm_bbox':symbol['norm_bbox'] if symbol else None,
                        'symbol_checked':page_symbols_map is not None}
            entries = positions.setdefault(tag, [])
            if not any(entry['page'] == position['page'] and entry['bbox'] == position['bbox']
                       for entry in entries):
                entries.append(position)
    for entries in positions.values():
        explicit = {entry['model'] for entry in entries if entry['model_source'] == 'explicit'}
        numbers = {entry['number'] for entry in entries}
        consecutive = bool(numbers) and max(numbers) - min(numbers) + 1 == len(numbers)
        if len(explicit) != 1 or not consecutive:
            continue
        source = next(entry for entry in entries if entry['model_source'] == 'explicit')
        for entry in entries:
            if entry['model_source'] != 'unbound':
                continue
            entry['model'] = source['model']
            entry['model_source'] = 'same_strip_continuity'
            entry['model_bbox'] = source['model_bbox']
            entry['model_norm_bbox'] = source['model_norm_bbox']
            entry['model_page'] = source['model_page']
    return positions


def _number_label(tag, found, ambiguous):
    numbers = sorted({entry['number'] for entry in found})
    if ambiguous or len(numbers) != len(found):
        return tag
    ranges = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f'{start}~{previous}')
        start = previous = number
    ranges.append(str(start) if start == previous else f'{start}~{previous}')
    return tag + ':' + ','.join(ranges)


def _segment_breakdown(found):
    segments = defaultdict(list)
    for position in found:
        segments[(position['page'], tuple(position['header_bbox']))].append(position)
    pages = defaultdict(list)
    for (page, _), entries in segments.items():
        x_values = [(entry['bbox'][0] + entry['bbox'][2]) / 2 for entry in entries]
        y_values = [(entry['bbox'][1] + entry['bbox'][3]) / 2 for entry in entries]
        values = sorted(x_values if max(x_values)-min(x_values) >= max(y_values)-min(y_values) else y_values)
        gaps = [right-left for left, right in zip(values, values[1:])]
        threshold = max(40, median(gaps) * 3) if gaps else float('inf')
        count = 1
        for gap in gaps:
            if gap > threshold:
                pages[page].append(count)
                count = 1
            else:
                count += 1
        pages[page].append(count)
    return '、'.join(f'第{page}页 {"+".join(map(str, counts))}位'
                    for page, counts in sorted(pages.items()))


def apply_terminal_positions(items, reviews, positions):
    """Create one material row per terminal model, retaining position evidence."""
    existing_tags = {item.tag for item in items if item.entity_role == 'terminal'}
    for tag, found in positions.items():
        normalized = [entry for entry in found if entry.get('header_source') == 'unprefixed_contextual']
        if not normalized:
            continue
        source = normalized[0]
        reviews.append(ReviewItem(
            id=f'rev_terminal_tag_normalized_{re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")}',
            severity='WARNING', issue_type='TERMINAL_TAG_NORMALIZED', tag=tag,
            title=f'端子排位号缺少前缀：{source["header_raw_text"]}',
            description=(f'第{source["page"]}页原图写作 {source["header_raw_text"]}；'
                         f'其旁边有独立编号端子和型号，暂按 {tag} 归属。'
                         + (f'相邻已标位号：{"、".join(source["sequence_neighbors"])}。'
                            if source['sequence_neighbors'] else '')),
            suggestion='核对图纸位号；仅相邻序号或空间顺序不足以确认端子排归属。',
            page=source['page'], norm_bbox=source['header_norm_bbox']))
        if tag in existing_tags:
            continue
        page = source['page']
        items.append(BOMItem(
            id=f'term_normalized_p{page}_{re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")}',
            seq_no=max((item.seq_no for item in items), default=0) + 1,
            tag=tag, tags=[tag], local_tag=tag, name='端子（待核对位号）',
            entity_role='terminal', category='auxiliary', quantity=None,
            quantity_status='unknown', quantity_basis='terminal_positions_required',
            model_origin='terminal_tag_normalization', needs_review=True,
            primary_page=page, primary_bbox=source['header_norm_bbox'],
            evidences=[EvidenceLocation(page, source['header_bbox'], source['header_norm_bbox'],
                                        raw_text=source['header_raw_text'], role='main_circuit', device_tag=tag)]))
        existing_tags.add(tag)
    result = []
    for item in items:
        if item.entity_role != 'terminal' or item.tag not in positions:
            result.append(item)
            continue
        found = positions[item.tag]
        base_tag = item.tag
        page_numbers = defaultdict(set)
        for position in found:
            page_numbers[position['page']].add(position['number'])
        cross_page_overlap = any(left & right for page, left in page_numbers.items()
                                 for other_page, right in page_numbers.items() if page < other_page)
        repeated_numbers = len({(position['page'], position['number']) for position in found}) < len(found)
        inherited = [position for position in found if position['model_source'] == 'same_strip_continuity']
        unbound = [position for position in found if not position['model']]
        symbol_keys = [(entry['page'], tuple(entry['symbol_bbox'])) for entry in found
                       if entry['symbol_bbox'] is not None]
        missing_symbol = any(entry['symbol_checked'] and not entry['symbol_bbox'] for entry in found)
        reused_symbol = len(symbol_keys) != len(set(symbol_keys))
        uncertain = cross_page_overlap or missing_symbol or reused_symbol
        if cross_page_overlap:
            reviews.append(ReviewItem(
                id=f'rev_terminal_overlap_{base_tag}', severity='WARNING',
                issue_type='TERMINAL_POSITIONS_AMBIGUOUS', tag=base_tag,
                title=f'端子排跨页编号重叠：{base_tag}',
                description='跨页重复编号，需区分延续与重复画法。',
                suggestion='核对每一页的端子位置、型号和接线。',
                page=found[0]['page'], norm_bbox=found[0]['norm_bbox']))
        if missing_symbol or reused_symbol:
            reviews.append(ReviewItem(
                id=f'rev_terminal_symbol_{base_tag}', severity='WARNING',
                issue_type='TERMINAL_SYMBOL_UNVERIFIED', tag=base_tag,
                title=f'端子符号待核对：{base_tag}',
                description='部分编号未对应到独立的矢量端子符号，数量仅作为候选。',
                suggestion='核对缺失位置的端子符号，必要时用图像识别补充。',
                page=found[0]['page'], norm_bbox=found[0]['norm_bbox']))
        if repeated_numbers:
            reviews.append(ReviewItem(
                id=f'rev_terminal_reused_{base_tag}', severity='WARNING',
                issue_type='TERMINAL_NUMBER_REUSED', tag=base_tag,
                title=f'端子编号重复：{base_tag}',
                description=f'识别到 {len(found)} 个独立位置，部分编号在同一页重复。',
                suggestion='核对是否为实际端子片，确认设计中的重复编号。',
                page=found[0]['page'], norm_bbox=found[0]['norm_bbox']))
        if inherited:
            source = next(position for position in found if position['model_source'] == 'explicit')
            reviews.append(ReviewItem(
                id=f'rev_terminal_model_inferred_{base_tag}', severity='WARNING',
                issue_type='TERMINAL_MODEL_INFERRED', tag=base_tag,
                title=f'端子型号沿同排继承：{base_tag}',
                description=(f'{source["number"]} 号明文标注 {source["model"]}；'
                             f'{len(inherited)} 个未逐个标注的端子按同位号连续编号继承该型号。'),
                suggestion='核对未标注端子的实际规格，尤其是 PE 端子。',
                page=source['page'], norm_bbox=source['norm_bbox']))
        if unbound:
            reviews.append(ReviewItem(
                id=f'rev_terminal_model_unbound_{base_tag}', severity='WARNING',
                issue_type='TERMINAL_MODEL_UNBOUND', tag=base_tag,
                title=f'端子型号未绑定：{base_tag}',
                description=f'{len(unbound)} 个图示端子位置尚无可靠型号。',
                suggestion='核对各位置端子型号，避免将 PE 与普通端子混算。',
                page=unbound[0]['page'], norm_bbox=unbound[0]['norm_bbox']))
        if not unbound:
            reviews[:] = [review for review in reviews if not (
                review.tag == base_tag and review.issue_type in ('UNRESOLVED_MODEL', 'MODEL_ASSOCIATION'))]
        groups = defaultdict(list)
        for position in found:
            groups[position['model']].append(position)
        base_evidences = [evidence for evidence in item.evidences if evidence.role not in
                          ('model_text', 'model_candidate', 'terminal_position', 'terminal_symbol')]
        base_remark = item.remark
        base_model = item.model
        base_model_origin = item.model_origin
        for index, (model, subset) in enumerate(groups.items()):
            row = item if index == 0 else copy.deepcopy(item)
            row.id = item.id if index == 0 else f'{item.id}__terminal_{re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")}'
            row.tag = _number_label(base_tag, subset, uncertain or repeated_numbers)
            row.tags = [row.tag]
            row.terminal_strip_tag = base_tag
            listed_numbers = [entry['number'] for entry in subset]
            row.terminal_numbers = (sorted(listed_numbers) if len(set(listed_numbers)) == len(listed_numbers)
                                    else listed_numbers)
            row.model = model or base_model
            row.raw_model = model
            row.name = row.procurement_name = ('端子（型号待核）' if not model else
                                               '接地端子' if model.endswith('-PE') else '接线端子')
            row.unit = '个'
            row.category = 'auxiliary'
            inherited_subset = any(p['model_source'] == 'same_strip_continuity' for p in subset)
            row.source_type = (SourceType.UNRESOLVED.value if not model else
                               SourceType.TOPOLOGY_INFERRED.value if inherited_subset else SourceType.EXPLICIT.value)
            row.model_origin = 'terminal_strip_inference' if inherited_subset else base_model_origin
            row.selection_suggestions = []
            row.evidences = copy.deepcopy(base_evidences)
            for position in subset:
                row.evidences.append(EvidenceLocation(
                    position['page'], position['bbox'], position['norm_bbox'],
                    raw_text=str(position['number']), role='terminal_position', device_tag=base_tag))
                if position['model_source'] == 'explicit':
                    row.evidences.append(EvidenceLocation(
                        position['model_page'], position['model_bbox'], position['model_norm_bbox'],
                        raw_text=model, role='model_text', device_tag=base_tag))
                elif position['model_source'] == 'same_strip_continuity':
                    row.evidences.append(EvidenceLocation(
                        position['page'], position['bbox'], position['norm_bbox'],
                        raw_text=model, role='model_inference', device_tag=base_tag,
                        model_binding='same_strip_continuity'))
                if position['symbol_bbox']:
                    row.evidences.append(EvidenceLocation(
                        position['page'], position['symbol_bbox'], position['symbol_norm_bbox'],
                        raw_text=str(position['number']), role='terminal_symbol', device_tag=base_tag))
            row.primary_page = subset[0]['page']
            row.primary_bbox = subset[0]['symbol_norm_bbox'] or subset[0]['norm_bbox']
            row.pdf_page_label = ', '.join(f'第{page}页' for page in sorted({p['page'] for p in subset}))
            row.remark = '；'.join(filter(None, [base_remark,
                f'图示端子位置：{_segment_breakdown(subset)}，合计{len(subset)}位']))
            row.quantity = None if uncertain or not model else float(len(subset))
            row.quantity_status = 'needs_confirmation' if uncertain or not model else 'by_positions'
            row.quantity_basis = ('terminal_positions_ambiguous' if uncertain else
                                  'terminal_model_unbound' if not model else 'drawn_terminal_positions')
            row.quantity_candidates = [float(len(subset))] if uncertain or not model else []
            row.needs_review = uncertain or repeated_numbers or inherited_subset or not model or any(
                review.status == 'PENDING' and review.tag in (base_tag, row.tag) for review in reviews)
            row.review_note = ('跨页编号重叠，数量待核。' if cross_page_overlap else
                               '端子图形未逐个核实，数量待核。' if missing_symbol or reused_symbol else
                               '同排仅部分端子明文标型号，其余位置按连续编号继承；请核对规格。' if inherited_subset else
                               '端子型号尚未绑定，采购数量待核。' if not model else
                               '同页端子编号重复，数量按独立位置统计。' if repeated_numbers else '')
            result.append(row)
    items[:] = result
    for index, item in enumerate(items, 1):
        item.seq_no = index
