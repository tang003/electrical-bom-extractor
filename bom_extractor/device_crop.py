"""Choose a device image from drawing geometry and page text, not the UI highlight."""
from __future__ import annotations

import re

from .notation import is_connection_label


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _union(boxes):
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _pad(box, page, points):
    x, y = points / page.rect.width, points / page.rect.height
    return [max(0, box[0] - x), max(0, box[1] - y),
            min(1, box[2] + x), min(1, box[3] + y)]


def _matching_enclosure(page, anchor):
    """Match the four straight sides of a rounded CAD device outline."""
    w, h = page.rect.width, page.rect.height
    horizontal, vertical = [], []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] != 'l':
                continue
            a, b = item[1:3]
            if abs(a.y - b.y) < .5 and abs(a.x - b.x) > 35:
                horizontal.append((min(a.x, b.x), a.y, max(a.x, b.x)))
            elif abs(a.x - b.x) < .5 and abs(a.y - b.y) > 45:
                vertical.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    tops = [line for line in horizontal if abs(line[1] - anchor[1] * h) < 35
            and abs(line[0] - anchor[2] * w) < 30]
    found = []
    for top in tops:
        bottoms = [line for line in horizontal if 45 < line[1] - top[1] < .5 * h
                   and abs(line[0] - top[0]) < 2 and abs(line[2] - top[2]) < 2]
        for bottom in bottoms:
            lefts = [line for line in vertical if 0 < top[0] - line[0] < 15
                     and abs(line[1] - top[1]) < 15 and abs(line[2] - bottom[1]) < 15]
            rights = [line for line in vertical if 0 < line[0] - top[2] < 15
                      and abs(line[1] - top[1]) < 15 and abs(line[2] - bottom[1]) < 15]
            for left in lefts:
                for right in rights:
                    box = [left[0] / w, top[1] / h, right[0] / w, bottom[1] / h]
                    if box[0] < anchor[2] + .02 and box[2] > anchor[0] and box[1] < anchor[3]:
                        found.append(box)
    return min(found, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])) if found else None


def _grid_cell(spans, anchor, neighbors, page_num, target_id):
    cx = _center(anchor)[0]
    grid = sorted(_center(span['norm_bbox'])[0] for span in spans
                  if re.fullmatch(r'\d', span['text']) and span['norm_bbox'][1] < .025)
    if len(grid) >= 2:
        index = min(range(len(grid)), key=lambda i: abs(grid[i] - cx))
        if abs(grid[index] - cx) < .045:
            left = (grid[index - 1] + grid[index]) / 2 if index else grid[0] - (grid[1] - grid[0]) / 2
            right = (grid[index] + grid[index + 1]) / 2 if index + 1 < len(grid) else grid[-1] + (grid[-1] - grid[-2]) / 2
            return max(0, left), min(1, right)
    same_row = sorted(_center(ev.norm_bbox)[0] for other in neighbors if other.id != target_id
                      for ev in other.evidences if ev.page == page_num and
                      ev.role in {'main_circuit', 'coil', 'device'} and
                      abs(_center(ev.norm_bbox)[1] - _center(anchor)[1]) < .09)
    lefts = [x for x in same_row if x < cx]
    rights = [x for x in same_row if x > cx]
    left = (max(lefts) + cx) / 2 if lefts else cx - .09
    right = (min(rights) + cx) / 2 if rights else cx + .09
    return max(0, left), min(1, right)


def _page_footer_top(page, anchor_bottom):
    """Find a full-width title-block divider below the device, when present."""
    width, height = page.rect.width, page.rect.height
    edges = []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] != 'l':
                continue
            start, end = item[1:3]
            if (abs(start.y - end.y) < .5 and
                min(start.x, end.x) <= .03 * width and
                max(start.x, end.x) >= .97 * width and
                start.y / height > max(.82, anchor_bottom + .02)):
                edges.append(start.y / height)
    return min(edges) if edges else None


def _open_symbol_crop(page, anchor, page_num, target_id, neighbors, spans):
    left, right = _grid_cell(spans, anchor, neighbors, page_num, target_id)
    if not left < _center(anchor)[0] < right:
        return None
    device_roles = {'main_circuit', 'coil', 'device', 'contact_or_symbol'}
    next_rows = [ev.norm_bbox[1] for other in neighbors if other.id != target_id
                 for ev in other.evidences if ev.page == page_num and ev.role in device_roles
                 and left <= _center(ev.norm_bbox)[0] <= right
                 and ev.norm_bbox[1] > anchor[1] + .05]
    next_top = min(next_rows) if next_rows else None
    footer_top = _page_footer_top(page, anchor[3])
    if next_top is None and footer_top is None and anchor[3] > .75:
        # Near the page edge, a single crop cannot safely distinguish device notes
        # from an unruled title block. Let the two-image context path handle it.
        return None
    bottom_limit = (min(1, anchor[3] + .4)
                    if next_top is None and footer_top is None else 1)
    if next_top is not None:
        bottom_limit = min(bottom_limit, next_top - 8 / page.rect.height)
    if footer_top is not None:
        bottom_limit = min(bottom_limit, footer_top - 2 / page.rect.height)
    if bottom_limit <= anchor[3]:
        return None
    top_limit = max(0, anchor[1] - .02)
    zone = [span for span in spans if left <= _center(span['norm_bbox'])[0] <= right
            and top_limit <= span['norm_bbox'][1] and span['norm_bbox'][3] <= bottom_limit]
    catalogue = [span for span in zone if re.search(r'[A-Za-z]', span['text'])
                 and re.search(r'\d', span['text']) and re.search(r'[-/_.]', span['text'])
                 and not span['text'].startswith(('-', '=', '+'))
                 and not is_connection_label(span['text']) and 'mm²' not in span['text']]
    if not catalogue:
        return None
    catalogue.sort(key=lambda span: (span['norm_bbox'][1], abs(_center(span['norm_bbox'])[0] - _center(anchor)[0])))
    chosen = [catalogue[0]]
    for span in catalogue[1:]:
        if (span['norm_bbox'][1] - chosen[-1]['norm_bbox'][3]) * page.rect.height <= 18:
            chosen.append(span)
        else:
            break
    base_bottom = max(anchor[3], chosen[-1]['norm_bbox'][3])
    end = base_bottom + 8 / page.rect.height
    selected = [span for span in zone if _center(span['norm_bbox'])[1] <= end]
    # Labels below an open symbol are often its voltage, maker, or part number.
    # Follow nearby lines in the same device lane until a real row or footer
    # boundary, rather than stopping a fixed distance below the tag.
    lane_left = max(left, anchor[0] - 18 / page.rect.width)
    lane_right = min(right, anchor[2] + 18 / page.rect.width)
    trailing = sorted((span for span in zone if span['norm_bbox'][1] > base_bottom
                       and lane_left <= _center(span['norm_bbox'])[0] <= lane_right),
                      key=lambda span: span['norm_bbox'][1])
    last_bottom = base_bottom
    for span in trailing:
        if (span['norm_bbox'][1] - last_bottom) * page.rect.height > 22:
            break
        if (span['norm_bbox'][3] - base_bottom) * page.rect.height > 75:
            break
        selected.append(span)
        last_bottom = max(last_bottom, span['norm_bbox'][3])
    box = _pad(_union([anchor, *[span['norm_bbox'] for span in selected]]), page, 6)
    box[3] = min(box[3], bottom_limit)
    if any(box[0] <= _center(ev.norm_bbox)[0] <= box[2] and
           box[1] <= _center(ev.norm_bbox)[1] <= box[3]
           for other in neighbors if other.id != target_id for ev in other.evidences
           if ev.page == page_num and ev.role in device_roles):
        return None
    return box


def plan_device_crop(page, anchor, page_num, target_id, target_tag, neighbors, spans):
    """Return one complete image when geometry is reliable, otherwise None."""
    enclosure = _matching_enclosure(page, anchor)
    if enclosure and any(enclosure[0] <= _center(ev.norm_bbox)[0] <= enclosure[2] and
                         enclosure[1] <= _center(ev.norm_bbox)[1] <= enclosure[3]
                         for other in neighbors if other.id != target_id for ev in other.evidences
                         if ev.page == page_num and ev.role in {'main_circuit', 'coil', 'device'}):
        return None
    if enclosure:
        top_band, bottom_band = 24 / page.rect.height, 12 / page.rect.height
        selected = [span for span in spans if enclosure[0] <= _center(span['norm_bbox'])[0] <= enclosure[2]
                    and enclosure[1] - top_band <= _center(span['norm_bbox'])[1] <= enclosure[3] + bottom_band]
        return {'mode': 'enclosure', 'box': _pad(_union([anchor, enclosure,
                *[span['norm_bbox'] for span in selected]]), page, 5)}
    if ':' in target_tag or '~' in target_tag or re.match(r'-?X(?:T)?\d', target_tag, re.I):
        return None
    box = _open_symbol_crop(page, anchor, page_num, target_id, neighbors, spans)
    return {'mode': 'layout_cell', 'box': box} if box else None
