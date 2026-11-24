"""Candidate exclusions remain inspectable and never hide normal device anchors."""
import pymupdf
import pytest

from bom_extractor.classifier import extract_visual_spans
from bom_extractor.topology_graph import VectorTopologyBuilder
from bom_extractor.notation import with_component_anchors, COMPONENT_TAG_RE, extract_model_strings


def drawing(mechanical=True, model=False, font_size=2):
    doc=pymupdf.open()
    page=doc.new_page(width=600,height=400)
    page.insert_text((30,30),'CABINET CUTOUT' if mechanical else 'CONTROL CIRCUIT',fontsize=8)
    page.draw_rect(pymupdf.Rect(200,75,500,84),color=None,fill=(0,0,0))
    page.draw_rect(pymupdf.Rect(200,75,210,280),color=None,fill=(0,0,0))
    page.insert_text((195,80),'-M4',fontsize=font_size)
    for x in [260,330,400]:page.insert_text((x,170),'M16',fontsize=8)
    page.insert_text((350,220),'-QF2',fontsize=8)
    if model:page.insert_text((175,65),'ZETA-9320',fontsize=6)
    return doc,page


def topology(page,monkeypatch):
    monkeypatch.setattr(VectorTopologyBuilder,'_run_yolo_if_available',lambda *args:[])
    return VectorTopologyBuilder.extract_wires_and_symbols(page,extract_visual_spans(page))


def test_tiny_mechanical_frame_label_is_excluded_but_remains_inspectable(monkeypatch):
    doc,page=drawing()
    try:
        result=topology(page,monkeypatch)
        assert [c['tag'] for c in result['candidates']]==['-QF2']
        excluded=result['excluded_candidates']
        assert [c['tag'] for c in excluded]==['-M4']
        assert excluded[0]['exclusion_reason'] and excluded[0]['anchor_norm_bbox']
        assert '-M4' not in str(result['branches'])
        assert len({c['som_id'] for c in result['candidates']+excluded})==2
    finally:doc.close()


@pytest.mark.parametrize('options',[{'mechanical':False},{'font_size':6},{'model':True}])
def test_mechanical_context_alone_does_not_exclude_supported_or_readable_tags(monkeypatch,options):
    doc,page=drawing(**options)
    try:
        result=topology(page,monkeypatch)
        assert '-M4' in {c['tag'] for c in result['candidates']}
        assert not result['excluded_candidates']
    finally:doc.close()


def test_native_cad_combined_tag_text_is_kept_and_gets_a_tight_candidate(monkeypatch):
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        for text,x,y in [('QF0 400A 3P',100,90),('<EZD400E3400K>',100,70),('QF1',330,160),('FU1',190,220),('M16',450,220),('AC380V',50,40)]:
            page.insert_text((x,y),text,fontsize=10)
        spans=[{**s,'source':'cad_native'} for s in extract_visual_spans(page)]
        original=next(s for s in spans if s['text']=='QF0 400A 3P')
        augmented=with_component_anchors(spans)
        anchors=[s for s in augmented if COMPONENT_TAG_RE.fullmatch(s['text'])]
        assert {s['text'] for s in anchors}=={'-QF0','-QF1','-FU1'}
        qf0=next(s for s in anchors if s['text']=='-QF0')
        assert qf0['bbox'][2]-qf0['bbox'][0]<original['bbox'][2]-original['bbox'][0]
        assert qf0['source_text']=='QF0 400A 3P' and original in augmented
        monkeypatch.setattr(VectorTopologyBuilder,'_run_yolo_if_available',lambda *args:[])
        result=VectorTopologyBuilder.extract_wires_and_symbols(page,spans)
        assert {c['tag'] for c in result['candidates']}=={'QF0','QF1','FU1'}
        assert not extract_model_strings('QF0 400A 3P')
        assert 'EZD400E3400K' in extract_model_strings('<EZD400E3400K>')


def test_grouped_table_tags_are_not_expanded_into_devices_by_anchor_parser():
    spans=[{'text':tag,'bbox':[100,100,160,110],'norm_bbox':[.1,.1,.16,.11],'cx':130,'cy':105,'size':10,'source':'cad_native'} for tag in ['QF1,2','-FU1~2','M16','AC380V']]
    assert not any(COMPONENT_TAG_RE.fullmatch(s['text']) for s in with_component_anchors(spans))


def test_project_warning_has_no_fabricated_component_coordinates():
    from bom_extractor.models import ReviewItem
    review=ReviewItem('cad','WARNING','CAD_IMPORT_REVIEW','','CAD导入提醒','字体替换','核对导入说明',1,scope='project')
    assert review.norm_bbox is None and review.scope=='project'
    legacy=ReviewItem('old','WARNING','CAD_IMPORT_REVIEW','','提醒','兼容性提醒','核对报告',1,norm_bbox=[.1,.1,.3,.3],status='ACCEPTED')
    assert legacy.norm_bbox is None and legacy.scope=='project' and legacy.status=='ACCEPTED'


def test_unnumbered_native_lamp_owns_its_model_instead_of_nearby_fuse():
    from bom_extractor.spatial_binding import bind_models
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        for text,x,y in [('FU1',100,160),('WL',110,230),('<XA2EVQ1LC>',140,230)]:
            page.insert_text((x,y),text,fontsize=10)
        spans=with_component_anchors([{**s,'source':'cad_native'} for s in extract_visual_spans(page)])
        anchors=[{**s,'tag':s['text']} for s in spans if COMPONENT_TAG_RE.fullmatch(s['text'])]
        bound,_,_,_=bind_models(page,spans,anchors)
        assert bound['-WL'][0]['model_text']=='XA2EVQ1LC'
        assert '-FU1' not in bound
        assert next(s for s in anchors if s['tag']=='-WL')['unindexed_tag']
