"""Document-wide sheet/column navigation, independent of purchasing items."""
from __future__ import annotations

import re

from .classifier import extract_visual_spans
from .ocr_engine import DualChannelOCREngine


INDEX_VERSION = 2
_NUMBER = re.compile(r'([1-9]\d*)[.．]([0-9])')
_FORWARD = re.compile(r'^/\s*([1-9]\d*)[.．]([0-9])(?:\s+.*)?$')
_REVERSE = re.compile(r'^([1-9]\d*)[.．]([0-9])\s*/(?:\s*.*)?$')


def _union(first, second):
    return [min(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), max(first[3], second[3])]


def _adjacent(left, right):
    gap = right['bbox'][0] - left['bbox'][2]
    height = max(left['bbox'][3] - left['bbox'][1], right['bbox'][3] - right['bbox'][1])
    return -.5 <= gap <= max(8, height * 1.5) and abs(left['cy'] - right['cy']) <= max(4, height * .5)


def _adjacent_vertical(top, bottom):
    gap = bottom['bbox'][1] - top['bbox'][3]
    height = max(top['bbox'][3] - top['bbox'][1], bottom['bbox'][3] - bottom['bbox'][1])
    overlap = min(top['bbox'][2], bottom['bbox'][2]) - max(top['bbox'][0], bottom['bbox'][0])
    width = min(top['bbox'][2] - top['bbox'][0], bottom['bbox'][2] - bottom['bbox'][0])
    return -.5 <= gap <= max(8, height * 1.5) and width > 0 and overlap / width >= .5


def _page_spans(page):
    native = extract_visual_spans(page)
    return native if sum(len(span['text']) for span in native) >= 80 else \
        DualChannelOCREngine.extract_dual_channel_spans(page, native, dpi=200)


def _grid_columns(page, info):
    columns = info.extraction_diagnostics.get('drawing_grid_columns', [])
    if len(columns) >= 8:
        return {int(entry['number']): float(entry['x']) for entry in columns}
    native = extract_visual_spans(page)
    labels = {int(span['text']): span['cx'] / page.rect.width for span in native
              if re.fullmatch(r'[0-9]', span['text'].strip()) and span['cy'] < page.rect.height * .04}
    return labels if len(labels) >= 8 else {}


def _markers(spans):
    slashes = [span for span in spans if span['text'].strip() == '/']
    for span in spans:
        raw = span['text'].strip().replace('／', '/')
        match = _FORWARD.fullmatch(raw)
        if match:
            yield int(match[1]), int(match[2]), span['norm_bbox'], raw
            continue
        match = _REVERSE.fullmatch(raw)
        if match:
            yield int(match[1]), int(match[2]), span['norm_bbox'], raw
            continue
        match = _NUMBER.fullmatch(raw)
        if not match:
            continue
        for slash in slashes:
            if _adjacent(slash, span):
                yield int(match[1]), int(match[2]), _union(slash['norm_bbox'], span['norm_bbox']), '/' + raw
                break
            if _adjacent(span, slash):
                yield int(match[1]), int(match[2]), _union(span['norm_bbox'], slash['norm_bbox']), raw + '/'
                break
            # Rotated CAD labels are stacked in visual coordinates. In the
            # bottom-to-top case the slash is above the printed sheet number.
            if _adjacent_vertical(slash, span):
                yield int(match[1]), int(match[2]), _union(slash['norm_bbox'], span['norm_bbox']), raw + '/'
                break
            if _adjacent_vertical(span, slash):
                yield int(match[1]), int(match[2]), _union(span['norm_bbox'], slash['norm_bbox']), '/' + raw
                break


def build_page_reference_index(doc, pages):
    """Resolve printed sheet numbers to PDF pages, retaining only valid destinations."""
    sheets = {}
    for info in pages:
        if str(info.sheet_no).isdigit():
            sheets.setdefault(int(info.sheet_no), []).append(info.page_num)
    by_page = {info.page_num: info for info in pages}
    target_columns = {info.page_num: _grid_columns(doc[info.page_num - 1], info)
                      for info in pages if 1 <= info.page_num <= len(doc)}
    links, seen = [], set()
    for source_page, info in sorted(by_page.items()):
        if not 1 <= source_page <= len(doc):
            continue
        for sheet, column, box, raw in _markers(_page_spans(doc[source_page - 1])):
            targets = sheets.get(sheet, [])
            if len(targets) != 1 or not 0 <= column <= 9:
                continue
            target_page = targets[0]
            grid = target_columns.get(target_page, {})
            if grid and column not in grid:
                continue
            if not .015 <= (box[1] + box[3]) / 2 <= .92:
                continue
            key = (source_page, sheet, column, *(round(value, 4) for value in box))
            if key in seen:
                continue
            seen.add(key)
            links.append({'key': ':'.join(map(str, key)), 'source_page': source_page,
                          'source_bbox': box, 'source_text': raw,
                          'sheet': sheet, 'column': column, 'target_page': target_page,
                          'grid_x': grid.get(column), 'column_verified': column in grid})
    return {'version': INDEX_VERSION, 'links': links}
