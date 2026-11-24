import pymupdf

from bom_extractor.classifier import extract_visual_spans
from bom_extractor.device_outlines import device_outline_boxes, outline_near_tag
from bom_extractor.topology_graph import VectorTopologyBuilder


def _rounded_body(page, left, top, right, bottom, radius=8):
    page.draw_line((left + radius, top), (right - radius, top))
    page.draw_line((left + radius, bottom), (right - radius, bottom))
    page.draw_line((left, top + radius), (left, bottom - radius))
    page.draw_line((right, top + radius), (right, bottom - radius))
    k = radius * .55
    for points in (
        ((left + radius, top), (left + radius - k, top), (left, top + radius - k), (left, top + radius)),
        ((right - radius, top), (right - radius + k, top), (right, top + radius - k), (right, top + radius)),
        ((left, bottom - radius), (left, bottom - radius + k), (left + radius - k, bottom), (left + radius, bottom)),
        ((right, bottom - radius), (right, bottom - radius + k), (right - radius + k, bottom), (right - radius, bottom)),
    ):
        page.draw_bezier(*points)


def test_rounded_device_body_outlines_are_distinct_from_tag_and_nearby_text():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    _rounded_body(page, 120, 100, 240, 220)
    _rounded_body(page, 340, 100, 460, 220)
    page.insert_text((105, 110), '-U1')
    page.insert_text((325, 110), '-U2')
    page.insert_text((125, 155), 'FILTER')
    page.insert_text((125, 300), '-KM1')

    outlines = device_outline_boxes(page)
    assert len(outlines) == 2
    first = outline_near_tag(outlines, pymupdf.Rect(105, 101, 120, 111))
    assert first is not None
    assert tuple(round(value) for value in first) == (120, 100, 240, 220)
    assert outline_near_tag(outlines, pymupdf.Rect(125, 285, 155, 300)) is None

    topology = VectorTopologyBuilder.extract_wires_and_symbols(page, extract_visual_spans(page))
    u1 = next(candidate for candidate in topology['candidates'] if candidate['tag'] == '-U1')
    u2 = next(candidate for candidate in topology['candidates'] if candidate['tag'] == '-U2')
    assert u1['bbox_source'] == u2['bbox_source'] == 'vector_device_outline'
    assert u1['bbox'][0] < 120 and u1['bbox'][2] > 240
    assert u1['bbox'][2] < u2['bbox'][0]
    doc.close()
