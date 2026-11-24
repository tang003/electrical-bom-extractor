"""Check model ownership using synthetic drawings of unrelated products."""

import pymupdf
import pytest

from bom_extractor.classifier import extract_visual_spans
from bom_extractor.models import PageInfo
from bom_extractor.notation import extract_model_strings, model_compatible
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.rules import load_rules_config
from bom_extractor.schematic_extractor import extract_from_schematic_pdf, _preferred_device_caption
from bom_extractor.vision import VisionRecognizer


def test_vector_evidence_keeps_original_page_after_table_exclusion(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, "extract_dual_channel_spans", lambda page, spans, **kwargs: spans)
    finding = {"type": "VECTOR_TEXT_MODEL", "page": 2, "tag": "-KT8", "name": "Timer",
               "model": "TR-28", "norm_bbox": [.1, .2, .3, .4]}
    monkeypatch.setattr(VisionRecognizer, "analyze_vector_regions_on_page", lambda *args: [finding])
    with pymupdf.open() as doc:
        doc.new_page(width=100, height=100)
        doc.new_page(width=800, height=500)
        retained = PageInfo(2, "CONTROL_CIRCUIT", "Control", width=800, height=500)
        devices, _ = extract_from_schematic_pdf(doc, [retained], load_rules_config())
    timer = next(item for item in devices if item.tag == "-KT8")
    evidence, = timer.evidences
    assert evidence.page == 2
    assert evidence.bbox == [80, 100, 240, 200]


@pytest.mark.parametrize("load_cabinet, expected_reference", [("+A", True), ("+B", False)])
def test_spatial_model_matching_a_field_load_on_another_page(monkeypatch, load_cabinet, expected_reference):
    monkeypatch.setattr(DualChannelOCREngine, "extract_dual_channel_spans", lambda page, spans, **kwargs: spans)
    monkeypatch.setattr(VisionRecognizer, "analyze_vector_regions_on_page", lambda *args: [])
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=400)
        page.insert_text((100, 100), "-KA22")
        page.insert_text((100, 114), "YV73")
        doc.new_page(width=600, height=400).insert_text((100, 100), "-YV73")
        infos = [PageInfo(i, "CONTROL_CIRCUIT", "Control", width=600, height=400, location_code=cabinet)
                 for i, cabinet in enumerate(("+A", load_cabinet), start=1)]
        devices, _ = extract_from_schematic_pdf(doc, infos, load_rules_config())
    relay = next(item for item in devices if item.tag == "+A-KA22")
    references = [e for e in relay.evidences if e.role == "field_load_reference"]
    assert (relay.model != "YV73") == expected_reference
    assert [e.device_tag for e in references] == (["+A-YV73"] if expected_reference else [])


@pytest.fixture
def read_native_devices(monkeypatch):
    # These PDFs contain native text and vectors. Keep the test independent of OCR
    # availability and of the workstation's model service.
    monkeypatch.setattr(DualChannelOCREngine, "extract_dual_channel_spans", lambda page, spans, **kwargs: spans)
    monkeypatch.setattr(VisionRecognizer, "analyze_vector_regions_on_page", lambda *args: [])

    def read(doc):
        info = PageInfo(1, "INSTRUMENT_WIRING", "Synthetic control panel", width=600, height=400)
        devices, reviews = extract_from_schematic_pdf(doc, [info], load_rules_config())
        return {device.tag: device for device in devices}, reviews

    return read


def test_thermal_relay_spelling_merges_evidence_only_within_scope(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, "extract_dual_channel_spans", lambda page, spans, **kwargs: spans)
    monkeypatch.setattr(VisionRecognizer, "analyze_vector_regions_on_page", lambda *args: [])
    with pymupdf.open() as doc:
        infos = []
        for number, (label, cabinet, model) in enumerate([
            ("FR8", "", ""), ("-FR8", "", "LRD365"), ("-FR8", "+C", "LRD335")
        ], start=1):
            page = doc.new_page(width=600, height=400)
            page.insert_text((100, 90), label)
            if model:
                page.insert_text((100, 105), model)
            infos.append(PageInfo(number, "CONTROL_CIRCUIT", "Control", width=600, height=400,
                                  location_code=cabinet))
        devices, _ = extract_from_schematic_pdf(doc, infos, load_rules_config())
    relays = {item.tag: item for item in devices if "FR8" in item.tag}
    assert {tag: item.model for tag, item in relays.items()} == {"FR8": "LRD365", "+C-FR8": "LRD335"}
    assert {e.page for e in relays["FR8"].evidences} == {1, 2}


@pytest.mark.parametrize(("tag", "model", "caption_above"), [
    ("-PI22", "FDM-703-Z7", False),
    ("-LT4", "CWT-850-D2", True),
])
def test_caption_across_device_body_belongs_to_device_not_closer_coil(
    read_native_devices, tag, model, caption_above
):
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=400)
        body = pymupdf.Rect(180, 124, 278, 280)
        if caption_above:
            # Some CAD plotters encode the same rectangular outline as a quad.
            page.draw_quad(pymupdf.Quad(body.tr, body.br, body.tl, body.bl))
            tag_y, model_y, coil_x, coil_y = 296, 112, 140, 145
        else:
            page.draw_rect(body)
            tag_y, model_y, coil_x, coil_y = 114, 297, 301, 251
        page.insert_text((219, tag_y), tag, fontsize=8)
        page.insert_text((205, model_y), model, fontsize=8)
        page.draw_rect(pymupdf.Rect(coil_x, coil_y+4, coil_x+13, coil_y+14))
        page.insert_text((coil_x, coil_y), "-K41", fontsize=8)
        # A different catalogue number is genuinely attached to the coil.
        coil_model_x = coil_x-42 if caption_above else coil_x+19
        page.insert_text((coil_model_x, coil_y + 17), "RY-118R", fontsize=8)
        items, _ = read_native_devices(doc)

    assert items[tag].model == model
    assert items["-K41"].model == "RY-118R"
    evidence = [ev for ev in items[tag].evidences if ev.role == "model_text" and ev.raw_text == model]
    assert evidence and all(ev.device_tag == tag for ev in evidence)


def test_model_between_equally_close_unboxed_tags_stays_unassigned(read_native_devices):
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=400)
        # The identical x coordinates and symmetric y offsets support both tags equally.
        page.insert_text((220, 160), "-K11", fontsize=8)
        page.insert_text((220, 200), "-K12", fontsize=8)
        page.insert_text((220, 180), "SX-991", fontsize=8)
        assert any(span["text"] == "SX-991" for span in extract_visual_spans(page))
        items, reviews = read_native_devices(doc)

    assert all(items[tag].model != "SX-991" for tag in ("-K11", "-K12"))
    assert any(review.issue_type == "UNBOUND_MODEL" and "SX-991" in review.title for review in reviews)


def test_wire_labels_cannot_become_part_numbers():
    for label in ("KM2L1", "KM2L2", "KM2L3", "FU6L2", "IN01_1", "V2-10"):
        assert extract_model_strings(label) == []
    assert extract_model_strings("LC1D12BDC") == ["LC1D12BDC"]


def test_axis_card_code_and_caption_are_owned_by_their_printed_tag(monkeypatch):
    def with_captions(page, spans, **kwargs):
        captions = []
        for x, label in ((100, '4轴数字量轴卡'), (320, '4轴模拟量轴卡')):
            for text, y in ((label, 104), ('轴卡', 111)):
                box = [x, y, x + len(text) * 6, y + 6]
                captions.append({'text':text, 'bbox':box,
                                 'norm_bbox':[box[0]/600,box[1]/300,box[2]/600,box[3]/300],
                                 'cx':(box[0]+box[2])/2, 'cy':(box[1]+box[3])/2,
                                 'size':6})
        return [*spans, *captions]

    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans', with_captions)
    monkeypatch.setattr(VisionRecognizer, 'analyze_vector_regions_on_page', lambda *args: [])
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=300)
        for x, tag, code in ((100, '-AX1515N', 'CK3W_AX1515N'),
                             (320, '-AX2323N', 'CK3W_AX2323N')):
            page.insert_text((x, 100), tag, fontsize=8)
            page.insert_text((x, 130), code, fontsize=8)
            page.insert_text((x+20, 170), 'V_FLAG0', fontsize=8)
            page.insert_text((x+40, 185), 'CH0', fontsize=8)
        info = PageInfo(1, 'MAIN_CIRCUIT', 'Main', width=600, height=300)
        devices, _ = extract_from_schematic_pdf(doc, [info], load_rules_config())
    by_tag = {device.tag:device for device in devices}
    assert [(by_tag[tag].name, by_tag[tag].model, by_tag[tag].needs_review)
            for tag in ('-AX1515N','-AX2323N')] == [
                ('4轴数字量轴卡','CK3W_AX1515N',False),
                ('4轴模拟量轴卡','CK3W_AX2323N',False)]
    assert all(any(e.role == 'device_description' for e in by_tag[tag].evidences)
               for tag in ('-AX1515N','-AX2323N'))


def test_compound_module_code_is_one_model_and_io_labels_are_not_models():
    assert extract_model_strings('CK3W_AX2323N') == ['CK3W_AX2323N']
    assert model_compatible('-AX2323N', 'CK3W_AX2323N')
    assert not model_compatible('-AX1515N', 'CK3W_AX2323N')
    for signal in ('V_FLAG0','CH0','IN14','OUT00','FAULT0','RJ45'):
        assert extract_model_strings(signal) == []
    assert _preferred_device_caption(['轴卡','4轴模拟量轴卡','4轴模拟量量轴卡']) == '4轴模拟量轴卡'
    assert _preferred_device_caption(['控制器','变频器']) == ''


def test_pdf_header_grid_columns_are_saved_for_drawing_navigation(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans', lambda page, spans, **kwargs: spans)
    monkeypatch.setattr(VisionRecognizer, 'analyze_vector_regions_on_page', lambda *args: [])
    with pymupdf.open() as doc:
        page = doc.new_page(width=842, height=595)
        for number in range(10):
            page.insert_text((55 + number * 80, 10), str(number), fontsize=8)
        page.insert_text((620, 200), '-KM2', fontsize=8)
        info = PageInfo(1, 'MAIN_CIRCUIT', 'Main', width=842, height=595)
        extract_from_schematic_pdf(doc, [info], load_rules_config())
    columns = info.extraction_diagnostics['drawing_grid_columns']
    assert [entry['number'] for entry in columns] == list(range(10))
    assert columns[7]['x'] == pytest.approx((615 + 2.2) / 842, abs=.005)


def test_cross_page_contact_and_coil_share_one_device(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, "extract_dual_channel_spans", lambda page, spans, **kwargs: spans)
    monkeypatch.setattr(VisionRecognizer, "analyze_vector_regions_on_page", lambda *args: [])
    with pymupdf.open() as doc:
        for _ in range(16):
            doc.new_page(width=800, height=500)
        power = doc[1]
        power.insert_text((250, 200), "-KM2", fontsize=8)
        power.insert_text((250, 210), "/16.7", fontsize=8)
        power.insert_text((300, 260), "KM2L1", fontsize=8)
        control = doc[15]
        control.insert_text((600, 200), "-KM2", fontsize=8)
        control.insert_text((600, 215), "LC1D12BDC", fontsize=8)
        infos = [PageInfo(page_num=n, page_type=kind, page_type_label=kind,
                          sheet_no=str(n), width=800, height=500)
                 for n, kind in ((2, "MAIN_CIRCUIT"), (16, "CONTROL_CIRCUIT"))]
        devices, reviews = extract_from_schematic_pdf(doc, infos, load_rules_config())

    contactor, = [item for item in devices if item.tag == "-KM2"]
    assert contactor.quantity == 1
    assert contactor.model == "LC1D12BDC"
    references = [ev for ev in contactor.evidences if ev.role == "cross_reference"]
    assert len(references) == 1
    assert references[0].cad_refs[0]["resolved"] is True
    assert not any(review.issue_type == "CROSS_REFERENCE_MISMATCH" for review in reviews)
