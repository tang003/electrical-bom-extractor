from types import SimpleNamespace

import pymupdf

from bom_extractor.cad_import import _label_cabinet_sheet_sets, cad_join_tag_fragments, cad_stacked_model_owner
from bom_extractor.cad_rack import read_rack
from bom_extractor.models import BOMItem
from bom_extractor.reconciliation import _keys, tag_members


def sheet(number, bottom, left):
    return {"title": f"电装图({number})", "layout": "Model", "cad_bbox": [left, bottom, left+100, bottom+70]}


def heading(text, y):
    return {"text": text, "cad_bbox": [40, y, 50, y+4]}


def test_separate_cabinets_require_numbered_sets_and_distinct_aligned_headings():
    pages = [sheet(1, 310, 0), sheet(2, 310, 110), sheet(1, 0, 0), sheet(2, 0, 110)]
    _label_cabinet_sheet_sets(pages, [heading("水泵柜 × 3", 412), heading("风机柜", 112)])
    assert [page["cabinet_heading"]["name"] for page in pages] == ["水泵柜"]*2 + ["风机柜"]*2
    row = BOMItem("r", 1, "QF1,2", location_code="+风机柜")
    assert _keys(row) == ["+风机柜-QF1", "+风机柜-QF2"]
    assert row.tag == "QF1,2"


def test_revisions_and_unplaced_captions_do_not_invent_cabinet_scope():
    for captions in ([], [heading("控制柜", 412), heading("控制柜", 112)],
                     [heading("水泵柜", 412), heading("其他柜", 418), heading("风机柜", 112)],
                     [heading("水泵柜", 650), heading("风机柜", 112)]):
        pages = [sheet(1, 310, 0), sheet(2, 310, 110), sheet(1, 0, 0), sheet(2, 0, 110)]
        _label_cabinet_sheet_sets(pages, captions)
        assert all("cabinet_heading" not in p for p in pages)


def test_cad_shorthand_keeps_prefix_on_suffixed_tags_and_subgroups():
    assert tag_members("ALT7A/7B/8A/8B") == ["-ALT7A","-ALT7B","-ALT8A","-ALT8B"]
    assert tag_members("WD2、7-2~4") == ["-WD2","-WD7-2","-WD7-3","-WD7-4"]
    assert tag_members("-KM1-2") == ["-KM1-2"]


def _native(text, x0, y0, x1, y1):
    return dict(text=text, source="cad_native", entity_type="TEXT", layer="labels", block_path=[],
                handle=f"{x0}:{y0}",bbox=[x0,y0,x1,y1],norm_bbox=[x0/300,y0/175,x1/300,y1/175],
                cx=(x0+x1)/2,cy=(y0+y1)/2)


def test_native_split_identifier_and_owned_catalogue_label_require_unique_geometry():
    parts=[_native("K",22,40,28,48),_native("M9",26,40,38,48)]
    joined=cad_join_tag_fragments(parts)
    assert joined[-1]["text"]=="KM9"
    tag={**joined[-1],"tag":"+A-KM9"}
    code={**_native("AX-6502 24V",19,48,41,55),"model_text":"AX-6502"}
    assert cad_stacked_model_owner(code,[tag])=="+A-KM9"
    assert cad_stacked_model_owner(code,[tag,{**tag,"tag":"+A-KM1"}]) is None
    assert cad_stacked_model_owner({**code,"layer":"unrelated"},[tag]) is None
    assert len(cad_join_tag_fragments([parts[0],{**parts[1],"layer":"other"}]))==2
    assert len(cad_join_tag_fragments([parts[0],_native("M9",45,40,57,48)]))==2


def test_numbered_and_bounded_rack_slots_preserve_codes_and_ignore_blank_slot_and_io_addresses():
    with pymupdf.open() as doc:
        page=doc.new_page(width=300,height=175)
        spans=[_native("CPU",22,73,40,79)]
        for slot in range(4):
            x=64+slot*30
            page.draw_rect(pymupdf.Rect(x,66,x+30,115))
            spans.append(_native(str(slot),x+12,58,x+18,64))
        spans.extend([_native("MX650",68,74,90,80),_native("ADX",101,74,120,80),
                      _native("MU41",130,74,149,80),_native("B72",130,81,145,87),
                      _native("X040",72,94,90,100)])
        info=SimpleNamespace(page_num=1,function_code="",location_code="+A")
        rack,used=read_rack(page,spans,info)
        assert [(r.model,r.quantity) for r in rack]==[("MX650",1),("MU41B72",1)]
        assert len(used)==3 and all(r.tag=="" for r in rack)
        assert read_rack(page,spans[1:],info)==([],set())
