"""Sheet/column links belong to drawing navigation, not BOM counting."""

import pymupdf

from bom_extractor.models import PageInfo
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.page_references import build_page_reference_index


def _pages():
    grid = [{'number': number, 'x': .05 + number * .1} for number in range(10)]
    return [PageInfo(1, 'CONTROL_CIRCUIT', 'Control', sheet_no='1', width=800, height=500),
            PageInfo(2, 'CONTROL_CIRCUIT', 'Control', sheet_no='2', width=800, height=500,
                     extraction_diagnostics={'drawing_grid_columns': grid})]


def test_forward_and_reverse_sheet_column_notation(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans',
                        lambda page, native, **kwargs: native)
    with pymupdf.open() as doc:
        doc.new_page(width=800, height=500)
        doc.new_page(width=800, height=500)
        source = doc[0]
        source.insert_text((100, 100), '/2.1', fontsize=10)
        source.insert_text((200, 160), '2.2', fontsize=10)
        source.insert_text((217, 160), '/', fontsize=10)
        source.insert_text((300, 220), '2.3', fontsize=10)
        source.insert_text((400, 280), '/9.1', fontsize=10)
        links = build_page_reference_index(doc, _pages())['links']

    assert {(link['source_text'], link['target_page'], link['column']) for link in links} == {
        ('/2.1', 2, 1), ('2.2/', 2, 2)}
    assert all(link['column_verified'] for link in links)


def test_ambiguous_sheet_number_does_not_make_a_false_link(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans',
                        lambda page, native, **kwargs: native)
    with pymupdf.open() as doc:
        doc.new_page(width=800, height=500)
        doc.new_page(width=800, height=500)
        doc.new_page(width=800, height=500)
        source = doc[0]
        source.insert_text((100, 100), '/2.1', fontsize=10)
        pages = _pages() + [PageInfo(3, 'CONTROL_CIRCUIT', 'Control', sheet_no='2')]
        assert build_page_reference_index(doc, pages)['links'] == []


def test_rotated_sheet_column_marker_uses_vertical_adjacency(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans',
                        lambda page, native, **kwargs: native)
    with pymupdf.open() as doc:
        doc.new_page(width=800, height=500)
        doc.new_page(width=800, height=500)
        source = doc[0]
        source.insert_text((100, 300), '2.4', fontsize=10, rotate=90)
        source.insert_text((100, 270), '/', fontsize=10, rotate=90)
        source.insert_text((200, 300), '2.5', fontsize=10, rotate=90)
        links = build_page_reference_index(doc, _pages())['links']

    assert [(link['source_text'], link['target_page'], link['column']) for link in links] == [
        ('2.4/', 2, 4)]
