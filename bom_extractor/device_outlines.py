"""Find device bodies drawn as rectangular or rounded vector outlines."""
from __future__ import annotations

from collections import defaultdict

import pymupdf

from .spatial_binding import symbol_enclosures


def device_outline_boxes(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Return complete bodies; CAD exports often split rounded corners into paths."""
    width, height = page.rect.width, page.rect.height
    min_width, min_height = min(70.0, width * .075), min(35.0, height * .05)
    horizontal, vertical, curves = [], [], []
    matrix = page.rotation_matrix
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l':
                a, b = item[1] * matrix, item[2] * matrix
                if abs(a.y - b.y) < .8 and abs(a.x - b.x) >= min_width * .7:
                    horizontal.append((min(a.x, b.x), (a.y + b.y) / 2, max(a.x, b.x)))
                elif abs(a.x - b.x) < .8 and abs(a.y - b.y) >= min_height * .5:
                    vertical.append(((a.x + b.x) / 2, min(a.y, b.y), max(a.y, b.y)))
            elif item[0] == 'c':
                points = [point * matrix for point in item[1:5]]
                curves.append(pymupdf.Rect(min(p.x for p in points), min(p.y for p in points),
                                           max(p.x for p in points), max(p.y for p in points)))

    def has_curve(expected):
        return any(all(abs(a-b) < 2.0 for a, b in zip(curve, expected)) for curve in curves)

    by_edges = defaultdict(list)
    for left, y, right in horizontal:
        by_edges[(round(left / 2), round(right / 2))].append((left, y, right))
    rounded = []
    for group in by_edges.values():
        group.sort(key=lambda line: line[1])
        for index, (left, top, right) in enumerate(group):
            for _, bottom, _ in group[index + 1:]:
                if not min_height < bottom - top < height * .8:
                    continue
                left_sides = [(x, y0, y1) for x, y0, y1 in vertical
                              if 2 < left-x < 30 and abs(y0-top-(left-x)) < 2.0
                              and abs(bottom-y1-(left-x)) < 2.0]
                right_sides = [(x, y0, y1) for x, y0, y1 in vertical
                               if 2 < x-right < 30 and abs(y0-top-(x-right)) < 2.0
                               and abs(bottom-y1-(x-right)) < 2.0]
                for lx, ly0, ly1 in left_sides:
                    for rx, ry0, ry1 in right_sides:
                        body = pymupdf.Rect(lx, top, rx, bottom)
                        if not (min_width < body.width < width * .75 and body.y1 < height * .92):
                            continue
                        corners = (pymupdf.Rect(lx, top, left, ly0),
                                   pymupdf.Rect(right, top, rx, ry0),
                                   pymupdf.Rect(lx, ly1, left, bottom),
                                   pymupdf.Rect(right, ry1, rx, bottom))
                        if all(has_curve(corner) for corner in corners):
                            rounded.append(body)
    outlines = []
    for rect in [*symbol_enclosures(page), *rounded]:
        if not any(abs(rect.x0-old.x0) < 2 and abs(rect.y0-old.y0) < 2 and
                   abs(rect.x1-old.x1) < 2 and abs(rect.y1-old.y1) < 2 for old in outlines):
            outlines.append(rect)
    return outlines


def outline_near_tag(boxes: list[pymupdf.Rect], anchor: pymupdf.Rect) -> pymupdf.Rect | None:
    """Select only a body whose top edge is uniquely associated with a tag."""
    center = (anchor.tl + anchor.br) / 2
    matches = []
    for body in boxes:
        gap_x = max(body.x0-center.x, center.x-body.x1, 0)
        gap_y = max(body.y0-center.y, center.y-body.y1, 0)
        if gap_x > 22 or gap_y > 18 or center.y > body.y0 + min(30, body.height * .35):
            continue
        score = gap_x + gap_y + abs(center.y-body.y0) * .2 + body.get_area() * .0001
        matches.append((score, body))
    matches.sort(key=lambda match: match[0])
    if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < 5):
        return None
    return matches[0][1]
