import pymupdf

from bom_extractor.classifier import classify_pdf_pages
from bom_extractor.models import BOMItem, EvidenceLocation, PageInfo
from bom_extractor.procurement import flag_template_quantities, range_annotations
from bom_extractor.schematic_extractor import extract_from_schematic_pdf


def label(text, x=.3, y=.4):
    return {'text':text, 'norm_bbox':[x,y,x+.025,y+.012]}


def test_only_a_range_for_the_repeated_circuit_can_make_page_quantities_uncertain():
    unrelated = [label(t) for t in ('L3-11','CN2-25','R5-5','K11~13')]
    annotations = range_annotations(unrelated, '电源配线')
    assert [(a['text'],a['scope']) for a in annotations] == [('K11~13','group')]
    # Identifiers and artwork differ from the reference projects.
    template = [label('DV02-06'), label('排水泵',y=.38)]
    assert range_annotations(template, '排水泵就地控制')[0]['scope'] == 'page'
    page = PageInfo(1,'CONTROL_CIRCUIT','回路图',extraction_diagnostics={'range_annotations':annotations})
    unit = BOMItem('u',1,'-Q8',evidences=[EvidenceLocation(1,[0,0,10,10],[0,0,.1,.1])])
    flag_template_quantities([unit],[],[page])
    assert unit.quantity == 1
    page.extraction_diagnostics['range_annotations'] = range_annotations(template, '排水泵就地控制')
    flag_template_quantities([unit],[],[page])
    assert unit.quantity is None and unit.quantity_candidates == [1,5]


def test_competing_codes_for_same_device_do_not_become_an_order_by_reading_order(monkeypatch):
    from bom_extractor.ocr_engine import DualChannelOCREngine
    from bom_extractor.vision import VisionRecognizer
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',lambda page,spans,**kw:spans)
    monkeypatch.setattr(VisionRecognizer,'analyze_vector_regions_on_page',lambda *args:[])
    with pymupdf.open() as doc:
        for code in ('AX-287','BX-491'):
            page=doc.new_page(width=500,height=350)
            page.insert_text((80,85),'-U47')
            page.insert_text((80,101),code)
        pages=classify_pdf_pages(doc)[-1]
        for p in pages:
            p.page_type='CONTROL_CIRCUIT'
        items,reviews=extract_from_schematic_pdf(doc,pages,{})
    device=next(i for i in items if i.tag=='-U47')
    assert device.source_type=='UNRESOLVED' and device.raw_model==''
    assert any(r.issue_type=='PARAM_MISMATCH' and 'AX-287' in r.description and 'BX-491' in r.description for r in reviews)
