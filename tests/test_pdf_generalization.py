"""Independent drawing and geometry regressions; no filename-specific extraction rules."""
from pathlib import Path
import pymupdf
import pytest

from bom_extractor.classifier import extract_visual_spans
from bom_extractor.models import BOMItem, PageInfo, SourceType
from bom_extractor.notation import COMPONENT_TAG_RE, extract_model_strings, qualify_tag
from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.rules import build_aggregated_from_itemized, load_rules_config
from bom_extractor.schematic_extractor import extract_from_schematic_pdf
from bom_extractor.spatial_binding import bind_models
from bom_extractor.vision import VisionRecognizer
from bom_extractor.ocr_engine import DualChannelOCREngine


def test_bare_pdf_tags_follow_devices_across_contact_and_power_pages(tmp_path,monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',lambda page,spans,**kw:spans)
    monkeypatch.setattr(VisionRecognizer,'analyze_vector_regions_on_page',lambda *args:[])
    path=tmp_path/'motor.pdf'
    with pymupdf.open() as doc:
        page=doc.new_page(width=620,height=410)
        for i,line in enumerate(['MC874','TH874','DQ3','P48V30002','N48V30002']):
            page.insert_text((75+i*90,80),line)
        page=doc.new_page(width=620,height=410)
        for x,tag,model in [(70,'MC874','LC1D50A'),(275,'TH874','LRN22N')]:
            page.insert_text((x,100),tag)
            page.insert_text((x,115),model)
        page.insert_text((280,123),'U874')
        page.insert_text((280,137),'V874')
        doc.save(path)
    r=run_extraction_pipeline(path,tmp_path/'run')
    items={i.tag:i for i in r.itemized_bom}
    assert (items['-MC874'].model,items['-MC874'].quantity)==('LC1D50A',1)
    assert (items['-TH874'].model,items['-TH874'].quantity)==('LRN22N',1)
    assert len({e.page for e in items['-TH874'].evidences})==2
    assert not any(r.issue_type=='PARAM_MISMATCH' for r in r.review_items)
    assert not {'-DQ3','-U874','-P48V30002'} & items.keys()


@pytest.mark.parametrize("tag",["-10QF1","-21A1","-K0.0","-K0.4","-KPH1","-19A2","-18BP2","-17FD1","-11S1","=SYS+MCC1-21A1","+BOX1-H1","-KM1-2"])
def test_iec_numbering_does_not_require_a_customer_prefix(tag):
    assert COMPONENT_TAG_RE.fullmatch(tag)


@pytest.mark.parametrize("text",["24V1.7","E0.0","11F0-L1","13.5:D","AC220V","DC24V 4P","C16A/2P","-X24V1","-XM0V1","-K0.0","RS485通讯"])
def test_signals_and_terminal_names_are_not_models(text):
    assert not extract_model_strings(text)


@pytest.mark.parametrize(("text","expected"),[
    ("CPU ST30 DC/DC/DC (6ES7 288-1ST30-0AA1)","6ES7288-1ST30-0AA1"),
    ("MOXA EDS-205A","EDS-205A"),("FBOX-4G-Lite(FLEXEM)","FBOX-4G-Lite"),
    ("ACS510-01-157A-4","ACS510-01-157A-4"),("UnknownMaker ZETA-9320","ZETA-9320"),
    ("A9D65216","A9D65216"),("AMC96L-E4/KC","AMC96L-E4/KC"),
    ("6AV2 124-0MC01-0AX0","6AV2124-0MC01-0AX0"),
])
def test_catalogues_can_be_embedded_in_descriptions(text,expected):
    assert expected in extract_model_strings(text)


def test_model_in_an_unlabelled_box_does_not_attach_to_external_switch():
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        page.draw_rect(pymupdf.Rect(300,100,550,300))
        page.insert_text((315,140),"ZETA-9320")
        page.insert_text((100,150),"-Q1")
        spans=extract_visual_spans(page)
        anchors=[{**sp,"tag":sp["text"]} for sp in spans if COMPONENT_TAG_RE.fullmatch(sp["text"])]
        bound,_,unbound,_=bind_models(page,spans,anchors)
        assert not bound
        assert any(model["model_text"]=="ZETA-9320" for model in unbound)


def test_display_order_code_is_kept_whole_and_distant_network_unit_is_excluded(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,"extract_dual_channel_spans",lambda page,spans,**kw:spans)
    monkeypatch.setattr(VisionRecognizer,"analyze_vector_regions_on_page",lambda *args:[])
    with pymupdf.open() as doc:
        page=doc.new_page(width=1100,height=700)
        page.insert_text((200,150),"-HMI9")
        page.insert_text((200,165),"6AV2 124-0MC01-0AX0")
        page.insert_text((430,150),"NET-4290")
        items,review=extract_from_schematic_pdf(doc,[PageInfo(1,"CONTROL_CIRCUIT","control",width=1100,height=700)],load_rules_config())
    assert len(items)==1 and items[0].name=="触摸屏" and items[0].model=="6AV2124-0MC01-0AX0"
    assert any('NET-4290' in r.title and r.issue_type=='UNBOUND_MODEL' for r in review)


@pytest.mark.parametrize('tag,expected',[('-ES742-8','label_stack'),('-ES743-8',''),('-ES2','')])
def test_three_line_caption_requires_matching_indexed_series_and_power_current(monkeypatch,tag,expected):
    monkeypatch.setattr(DualChannelOCREngine,"extract_dual_channel_spans",lambda page,spans,**kw:spans)
    monkeypatch.setattr(VisionRecognizer,"analyze_vector_regions_on_page",lambda *args:[])
    with pymupdf.open() as doc:
        page=doc.new_page(width=800,height=600)
        for line,text in enumerate([tag,'30 kW, 61 A','ES742-4X650-Q3']):
            page.insert_text((200,150+line*15),text,fontsize=10)
        items,_=extract_from_schematic_pdf(doc,[PageInfo(1,"MAIN_CIRCUIT","power",width=800,height=600)],load_rules_config())
    item=next(i for i in items if i.tag==tag)
    stack=[e for e in item.evidences if e.model_binding=='label_stack']
    assert bool(stack)==bool(expected)
    if expected:
        assert item.name=='功率设备（型式待核）' and item.specification=='30 kW, 61 A'


def test_same_local_tag_in_two_cabinets_stays_two_unknown_devices(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,"extract_dual_channel_spans",lambda page,spans,**kw:spans)
    monkeypatch.setattr(VisionRecognizer,"analyze_vector_regions_on_page",lambda *args:[])
    with pymupdf.open() as doc:
        for _ in range(2):
            doc.new_page(width=600,height=400).insert_text((100,100),"-H1")
        pages=[PageInfo(1,"CONTROL_CIRCUIT","control",width=600,height=400,location_code="+BOX1"),
               PageInfo(2,"CONTROL_CIRCUIT","control",width=600,height=400,location_code="+BOX2")]
        items,_=extract_from_schematic_pdf(doc,pages,load_rules_config())
    assert {it.tag for it in items}=={"+BOX1-H1","+BOX2-H1"}
    assert len(build_aggregated_from_itemized(items))==2
    assert all(it.source_type==SourceType.UNRESOLVED.value for it in items)


def test_unknown_quantity_correction_remains_with_its_device():
    import backend.app as backend
    from tests.test_regressions import result, register
    item=BOMItem("unknown",1,"-K1",tags=["-K1"],name="Relay",model="未提取到型号(待核实)",source_type="UNRESOLVED")
    res=result([item])
    pid=register(res)
    row=res.aggregated_bom[0]
    data=backend.update_bom_item(pid,backend.BOMItemUpdatePayload(id=row.id,tag=row.tag,name=row.name,model=row.model,quantity=5))
    assert len(data["aggregated_bom"])==1
    assert data["aggregated_bom"][0]["quantity"]==5


def test_real_eplan_drawing_recovers_independently_checked_models(tmp_path):
    pdf=Path(__file__).resolve().parents[1]/"参考文件"/"2-重庆长城诺博75KW.pdf"
    if not pdf.is_file():
        pytest.skip("本地参考图纸未提供")
    result=run_extraction_pipeline(pdf,output_dir=tmp_path/"new_drawing")
    by_tag={it.tag:it for it in result.itemized_bom}
    # Checked against rendered pages 3/4/5/6/8/12/14/15, not the extractor's old output.
    expected={
        "+MCC1-10P1":"AMC96L-E4/KC", "+MCC1-11F1":"A9D65216",
        "+MCC1-11F0":"IC65N 3P C6", "+MCC1-11F2":"IC65N 2P C6",
        "+MCC1-11F3":"IC65N 2P C10", "+MCC1-KPH1":"RM22TA33",
        "+MCC1-13G1":"EDR-150-24", "+MCC1-15Q1":"EZD160M3160 MAN",
        "+MCC1-15KM1":"LC1D09M7C", "+MCC1-15TB1":"ACS510-01-157A-4",
        "+MCC1-19A1":"EDS-205A", "+MCC1-19A2":"FBOX-4G-Lite",
        "+MCC1-21A1":"6ES7288-1ST30-0AA1", "+MCC1-22A1":"6ES7288-3AM03-0AA0",
    }
    for tag,model in expected.items():
        assert by_tag[tag].model==model, (tag,by_tag[tag].model)
        assert any(ev.role=="model_text" for ev in by_tag[tag].evidences)
    assert by_tag["+MCC1-21A1"].quantity==1
    assert {ev.page for ev in by_tag["+MCC1-21A1"].evidences}=={14,18}
    assert by_tag["+MCC1-15KM1"].quantity==1  # main contacts and coil on page 8
    assert by_tag["+BOX1-Q1"].source_type=="UNRESOLVED"
    assert any(r.issue_type=="UNBOUND_MODEL" and "MV01-07" in r.title for r in result.review_items)
    assert not any(it.model in ("X24V1","X24V2","XM0V1","RXM2LB2BD+RXZE1M2C") for it in result.itemized_bom)
    assert result.project_name=="干式除尘系统"
    assert result.pages[0].page_type=="COVER" and result.pages[1].page_type=="COVER"
    assert result.pages[13].page_type=="PLC_WIRING"
    assert len(result.itemized_bom)>40
    assert result.summary["extraction_quality"]["readable_text_pages"]==20
    assert not any(it.source_type=="ESTIMATED_AUX" for it in result.itemized_bom)
