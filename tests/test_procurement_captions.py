import pymupdf

from bom_extractor.notation import extract_model_strings, qualify_tag
from bom_extractor.schematic_extractor import _infer_component_name_and_brand, _normalize_tag
from bom_extractor.spatial_binding import bind_models, device_captions


def label(text, x, y):
    return dict(text=text,bbox=[x,y,x+24,y+8],cx=x+12,cy=y+4,norm_bbox=[0,0,.1,.1])


def test_part_numbers_keep_decimal_size_and_printed_capacity():
    assert extract_model_strings('(HX-V210-2.25Z)') == ['HX-V210-2.25Z']
    assert extract_model_strings('ACME PZ-095(1.5kW)') == ['PZ-095(1.5kW)']
    assert extract_model_strings('RO2NC') == []
    assert extract_model_strings('RO12C') == []


def test_drawn_tag_can_have_its_own_location_with_an_inherited_function():
    assert qualify_tag(_normalize_tag('+Aux-V9'), '+Main', '=07') == '=07+AUX-V9'
    assert qualify_tag('=08+Aux-V9', '+Main', '=07') == '=08+AUX-V9'


def test_a_repeated_bare_tag_is_not_a_neighbours_model():
    with pymupdf.open() as doc:
        page=doc.new_page(width=300,height=300)
        one=dict(label('-ZZ8',10,30),tag='=07+BOX-ZZ8')
        two=dict(label('-A9',90,30),tag='=07+BOX-A9')
        bare=label('ZZ8',90,39)
        assert extract_model_strings(bare['text']) == ['ZZ8']
        assert not bind_models(page,[one,two,bare],[one,two])[0]


def test_side_labels_are_assigned_to_the_nearest_symbol_before_its_caption():
    left=pymupdf.Rect(60,40,170,110)
    right=pymupdf.Rect(213,40,323,110)
    tags=[dict(label('-A9',34,40),tag='-A9'),dict(label('-A4',185,40),tag='-A4')]
    captions=[label('主机风机',80,110),label('辅助电源',245,110),label('开启信号',260,118)]
    bound=device_captions(captions,tags,[left,right])
    assert [s['text'] for s in bound['-A4']]==['辅助电源']
    assert '-A9' not in bound
    assert _infer_component_name_and_brand('-QF12','XY-570',[])[0]=='开关器件（型式待核）'
