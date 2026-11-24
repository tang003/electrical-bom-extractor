import pymupdf

from bom_extractor.device_inventory import build_device_inventory, page_targets, inventory_review_issues
from bom_extractor.classifier import extract_visual_spans
from bom_extractor.models import BOMItem, ExtractionResult, PageInfo
from bom_extractor.notation import is_connection_tag
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.rules import load_rules_config
from bom_extractor.schematic_extractor import extract_from_schematic_pdf
from bom_extractor.vision import VisionRecognizer
from bom_extractor.topology_graph import VectorTopologyBuilder


def test_off_page_voltage_labels_are_not_materials(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans',
                        lambda page, spans, **kwargs: spans)
    monkeypatch.setattr(VisionRecognizer, 'analyze_vector_regions_on_page', lambda *args: [])
    monkeypatch.setattr(VectorTopologyBuilder, '_run_yolo_if_available', lambda *args: [])
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=400)
        page.insert_text((120, 160), '-V2-07')
        page.insert_text((120, 175), '5.7')
        page.insert_text((320, 160), '-V2-08')
        page.insert_text((320, 175), '5.7')
        page.insert_text((220, 250), '-KM1')
        info = PageInfo(1, 'CONTROL_CIRCUIT', 'Control', width=600, height=400)
        inventory = build_device_inventory(doc, [info])
        devices, _ = extract_from_schematic_pdf(doc, [info], load_rules_config(), inventory=inventory)
        topology = VectorTopologyBuilder.extract_wires_and_symbols(page, extract_visual_spans(page))
    assert is_connection_tag('-V2-07') and is_connection_tag('-V2+01')
    assert not is_connection_tag('-V2')
    assert {item['tag'] for item in inventory['devices']} == {'-KM1'}
    assert {item.tag for item in devices} == {'-KM1'}
    assert {'-V2-07', '-V2-08'} <= {candidate['tag'] for candidate in topology['excluded_candidates']}


def test_inventory_groups_repeated_tags_before_device_recognition(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine, 'extract_dual_channel_spans',
                        lambda page, spans, **kwargs: spans)
    with pymupdf.open() as doc:
        first = doc.new_page(width=600, height=400)
        first.insert_text((100, 100), '-F1')
        first.insert_text((200, 100), '-F2')
        second = doc.new_page(width=600, height=400)
        second.insert_text((100, 100), '-F1')
        second.insert_text((200, 100), '-FU1')
        infos = [PageInfo(n, 'MAIN_CIRCUIT', '动力', width=600, height=400)
                 for n in (1, 2)]
        inventory = build_device_inventory(doc, infos)

    devices = {item['tag']:item for item in inventory['devices']}
    assert set(devices) == {'-F1', '-F2', '-FU1'}
    assert [location['page'] for location in devices['-F1']['occurrences']] == [1, 2]
    assert devices['-F1']['family'] == 'F'
    assert devices['-FU1']['family'] == 'FU'
    assert {item['tag'] for item in page_targets(inventory, 2)} == {'-F1', '-FU1'}

    records = [
        {'page':1, 'data':{'devices':[{'local_tag':'-F1'}, {'local_tag':'-F2'}]}},
        {'page':2, 'data':{'devices':[{'local_tag':'-FU1'}]}},
    ]
    issues = inventory_review_issues(inventory, records, {info.page_num:info for info in infos})
    assert set(issues) == {2}
    assert '-F1' in issues[2][0]


def test_inventory_endpoint_filters_device_family(tmp_path, monkeypatch):
    import backend.app as backend

    pdf=tmp_path/'drawing.pdf'
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        page.insert_text((100,100),'-F1')
        page.insert_text((200,100),'-FU1')
        doc.save(pdf)
    project='inventory_test'
    info=PageInfo(1,'MAIN_CIRCUIT','动力',width=600,height=400)
    result=ExtractionResult(project,pdf.name,'测试','','SCHEMATIC_MODE',1,
                            pages=[info],itemized_bom=[BOMItem('f1',1,'-F1',model='IC65N',quantity=1)])
    monkeypatch.setattr(backend,'OUTPUT_DIR',tmp_path)
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',
                        lambda page,spans,**kwargs:spans)
    backend.store.results[project]=result
    backend.store.pdf_paths[project]=pdf
    try:
        response=backend.get_device_inventory(project,'F')
    finally:
        backend.store.results.pop(project,None)
        backend.store.pdf_paths.pop(project,None)

    assert response['total_devices']==2
    assert [(item['tag'],item['model'],item['quantity']) for item in response['devices']]==[('-F1','IC65N',1)]


def test_inventory_candidate_without_device_becomes_review(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',
                        lambda page,spans,**kwargs:spans)
    monkeypatch.setattr(VisionRecognizer,'analyze_vector_regions_on_page',lambda *args:[])
    inventory={'devices':[{'tag':'-F1','occurrences':[{'page':1,'bbox':[.1,.2,.2,.3]}]}]}
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        info=PageInfo(1,'MAIN_CIRCUIT','动力',width=600,height=400)
        items,reviews=extract_from_schematic_pdf(doc,[info],load_rules_config(),inventory=inventory)

    assert not any(item.tag=='-F1' for item in items)
    assert any(review.issue_type=='INVENTORY_UNBOUND' and review.tag=='-F1' for review in reviews)
