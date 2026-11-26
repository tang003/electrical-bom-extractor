"""General label rules: unrelated tags, pages, scales and rating spellings."""
import copy

import pymupdf
import pytest

from bom_extractor.classifier import extract_visual_spans
from bom_extractor.component_names import owned_name, printed_device_caption, resistor_parameters, repair_saved_component_names
from bom_extractor.models import BOMItem, EvidenceLocation, ExtractionResult, PageInfo
from bom_extractor.notation import extract_model_strings, qualify_tag
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.rules import load_rules_config
from bom_extractor.schematic_extractor import extract_from_schematic_pdf
from bom_extractor.spatial_binding import stacked_device_annotations
from bom_extractor.vision import VisionRecognizer


def draw_resistor(page, tag, caption, rating, scale=1, combined=False):
    def text(x, y, value, chinese=False):
        page.insert_text((x*scale,y*scale),value,fontsize=8*scale,
                         fontname='china-s' if chinese else 'helv')
    text(100,100,tag)
    if combined:
        text(100,114,caption+' '+rating,True)
    else:
        text(100,114,rating,True)
        text(100,126,caption,True)
        # This pin falls between the parameter and name in top-to-bottom order.
        tag_span = next(s for s in extract_visual_spans(page) if s['text'] == tag)
        page.insert_text((tag_span['bbox'][2]+2*scale,115*scale),'2',fontsize=8*scale)


@pytest.mark.parametrize('text', ['20欧姆 800W','47Ω 1kW','0.5kΩ / 200W','800W / 20 OHM','2,2千欧姆','120Ω'])
def test_resistance_ratings_are_parameters_not_order_codes(text):
    assert resistor_parameters(text) == text
    assert extract_model_strings(text) == []
    assert owned_name('-R84',[text]) == '电阻（类型待核）'


@pytest.mark.parametrize('text', ['RXLG-800W-20R','X轴反馈','20Ω 信号说明','20Ω 800W 采购3个'])
def test_partial_values_and_order_codes_are_not_used_as_complete_parameters(text):
    assert resistor_parameters(text) == ''


@pytest.mark.parametrize(('tag','caption','rating','scale','number','combined'), [
    ('-R83','主轴制动电阻','20欧姆 800W',1,1,False),
    ('-12R7','泄放电阻','47Ω 1kW',1.5,3,False),
    ('-R406','限流电阻器','50 OHM 200W',.8,2,True),
])
def test_new_drawings_extract_name_and_rating_across_tags_pages_and_scales(
        monkeypatch,tag,caption,rating,scale,number,combined):
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',lambda page,spans,**kwargs:spans)
    monkeypatch.setattr(VisionRecognizer,'analyze_vector_regions_on_page',lambda *args:[])
    with pymupdf.open() as doc:
        for _ in range(number): page=doc.new_page(width=600*scale,height=400*scale)
        draw_resistor(page,tag,caption,rating,scale,combined)
        info=PageInfo(number,'MAIN_CIRCUIT','Synthetic drawing',width=600*scale,height=400*scale,
                      location_code='+CAB',function_code='=DRIVE')
        items,_=extract_from_schematic_pdf(doc,[info],load_rules_config())
    item=next(i for i in items if i.tag==qualify_tag(tag,'+CAB','=DRIVE'))
    assert item.name==caption
    assert rating in item.specification
    assert item.model.startswith('未提取到型号')
    assert item.raw_model==''
    assert any(e.page==number and caption in e.raw_text and e.device_tag==item.tag
               and e.role in ('device_description','device_annotation') for e in item.evidences)
    assert any(e.role=='specification' and e.raw_text==rating for e in item.evidences)


def test_side_pin_does_not_cut_off_the_caption_but_a_new_tag_still_stops_the_stack():
    def span(text,x,y,width=24):
        return dict(text=text,bbox=[x,y,x+width,y+6],cx=x+width/2,cy=y+3,
                    norm_bbox=[x/600,y/400,(x+width)/600,(y+6)/400])
    anchor=dict(span('-R88',100,80),tag='-R88')
    next_anchor=dict(span('-R89',100,130),tag='-R89')
    spans=[span('20Ω 800W',100,89,60),span('2',126,91,4),span('制动电阻',100,98,40),
           next_anchor,span('另一路电阻',100,138,50)]
    notes=stacked_device_annotations(spans,[anchor,next_anchor])
    assert [s['text'] for s in notes['-R88']]==['20Ω 800W','制动电阻']
    assert '另一路电阻' not in [s['text'] for s in notes['-R88']]


def test_inline_caption_is_separated_from_its_rating():
    assert printed_device_caption('泄放电阻 50Ω 200W')=='泄放电阻'
    assert resistor_parameters('泄放电阻 50Ω 200W')=='50Ω 200W'
    assert printed_device_caption('电阻 采购3个')==''
    assert owned_name('-A7',['控制变压器'])=='控制变压器'
    assert owned_name('-C42',['滤波电容器'])=='滤波电容器'


@pytest.mark.parametrize('manual',[False,True])
def test_reapply_to_saved_drawings_recovers_literal_evidence_without_reextracting(manual):
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        draw_resistor(page,'-R88','泄放电阻','47欧姆 500W')
        anchor=next(s for s in extract_visual_spans(page) if s['text']=='-R88')
        item=BOMItem('old',1,'-R88',name='电气元器件',model='未提取到型号(待核实)',raw_model='',quantity=None,
                     source_type='MANUAL_EDITED' if manual else 'UNRESOLVED',
                     evidences=[EvidenceLocation(1,anchor['bbox'],anchor['norm_bbox'],raw_text='-R88',
                                                 role='main_circuit',device_tag='-R88')])
        aggregate=copy.deepcopy(item);aggregate.id='summary';aggregate.member_ids=['old']
        result=ExtractionResult('generic','generic.pdf','','','SCHEMATIC_MODE',1,
            pages=[PageInfo(1,'MAIN_CIRCUIT','Synthetic',width=600,height=400)],
            itemized_bom=[item],aggregated_bom=[aggregate],independent_bom=[copy.deepcopy(item)])
        frozen=copy.deepcopy(result.independent_bom[0].to_dict())
        changed=repair_saved_component_names(result,doc)
        if manual:
            assert changed==[]
            assert item.name=='电气元器件' and len(item.evidences)==1
        else:
            assert changed==['old']
            assert item.name==aggregate.name=='泄放电阻'
            assert item.specification==aggregate.specification=='47欧姆 500W'
            assert item.quantity is None and item.raw_model=='' and item.model=='未提取到型号(待核实)'
            assert repair_saved_component_names(result,doc)==[]
        assert result.independent_bom[0].to_dict()==frozen


def test_project_recovery_reapplies_rules_across_saved_projects(tmp_path):
    from backend.app import ProjectStore
    store=ProjectStore()
    for index,(tag,caption) in enumerate([('-R18','限流电阻'),('-R207','制动电阻')]):
        pid=f'project_{index}';folder=tmp_path/pid;folder.mkdir()
        with pymupdf.open() as doc:
            page=doc.new_page(width=600,height=400)
            draw_resistor(page,tag,caption,'33欧姆 200W')
            anchor=next(s for s in extract_visual_spans(page) if s['text']==tag)
            doc.save(folder/'source.pdf')
        item=BOMItem('saved',1,tag,name='电气元器件',quantity=1,
            evidences=[EvidenceLocation(1,anchor['bbox'],anchor['norm_bbox'],raw_text=tag,
                                        role='main_circuit',device_tag=tag)])
        result=ExtractionResult(pid,'source.pdf','','','SCHEMATIC_MODE',1,
            pages=[PageInfo(1,'MAIN_CIRCUIT','Synthetic',width=600,height=400)],itemized_bom=[item])
        store.results[pid]=result;store.pdf_paths[pid]=folder/'source.pdf'
        store.refresh_drawing_label_rules(pid)
        assert item.name==caption and item.specification=='33欧姆 200W'


def test_saved_label_recovery_requires_the_validated_isolated_view(tmp_path):
    from backend.app import ProjectStore
    store=ProjectStore()
    source=tmp_path/'source.pdf'
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        draw_resistor(page,'-R18','制动电阻','33欧姆 200W')
        doc.save(source)
    item=BOMItem('saved',1,'-R18',name='电气元器件')
    result=ExtractionResult('missing_view','source.pdf','','','SCHEMATIC_MODE',1,itemized_bom=[item],
        summary={'independent_analysis':{'policy':'independent-schematic-v1','view_sha256':'missing'}})
    store.results[result.project_id]=result;store.pdf_paths[result.project_id]=source
    store.refresh_drawing_label_rules(result.project_id)
    assert item.name=='电气元器件' and item.evidences==[]
