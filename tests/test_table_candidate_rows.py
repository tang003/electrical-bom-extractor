"""Grouped references keep source quantities and table markers stay within their own row."""
import copy

import pymupdf
import pytest
from fastapi.testclient import TestClient

import backend.app as backend
from bom_extractor.models import BOMItem,EvidenceLocation,ExtractionResult,PageInfo
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.reconciliation import tag_members
from bom_extractor.som_renderer import SoMVisualPromptBuilder


@pytest.mark.parametrize(('raw','members'),[
    ('QF3~4',['-QF3','-QF4']),('qf1,2',['-QF1','-QF2']),
    ('QF1，2、4～5',['-QF1','-QF2','-QF4','-QF5']),
    ('QF1;2/4至5',['-QF1','-QF2','-QF4','-QF5']),
    ('FU01~03',['-FU01','-FU02','-FU03']),
    ('=SYS+BOX-QF3~QF4',['=SYS+BOX-QF3','=SYS+BOX-QF4']),
    ('-KA1A~1C',['-KA1A','-KA1B','-KA1C']),('-KM1-2',['-KM1-2']),
    ('QF4~3',['-QF4~3']),('QF1~9999',['-QF1~9999']),
    ('=A-QF1~=B-QF2',['=A-QF1~=B-QF2']),
])
def test_reference_notation_formats_without_guessing(raw,members):
    assert tag_members(raw)==members


def drawing():
    doc=pymupdf.open();page=doc.new_page(width=400,height=300)
    page.insert_text((20,35),'-K1',fontsize=10)
    rows=[]
    for index,(tag,quantity) in enumerate([('QF3~4',99),('QF5',None),('WL1',0)]):
        top=80+index*30;bottom=top+30
        page.insert_text((60,bottom-8),tag,fontsize=10)
        page.insert_text((160,bottom-8),'MODEL-123',fontsize=10)
        page.draw_rect(pymupdf.Rect(50,top,380,bottom))
        ev=EvidenceLocation(1,[50,top,380,bottom],[50/400,top/300,380/400,bottom/300],raw_text=tag,role='table_row')
        rows.append(BOMItem(f'row{index}',index+1,tag,tags=[tag],name='Device',model='MODEL-123',quantity=quantity,evidences=[ev]))
    return doc,page,rows


@pytest.mark.parametrize('whole',[True,False])
def test_table_rows_never_merge_with_neighbors_or_become_branch_nodes(tmp_path,monkeypatch,whole):
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',lambda page,spans,**kw:spans)
    doc,page,rows=drawing()
    try:
        before=[copy.deepcopy(r.to_dict()) for r in rows]
        topo=SoMVisualPromptBuilder.render_som_annotated_page(page,tmp_path/'som.png',dpi=72,reference_rows=rows,whole_table=whole)
        table=[c for c in topo['candidates'] if c['detector_source']=='PARTS_TABLE_ROW']
        assert [c['tag'] for c in table]==['QF3~4','QF5','WL1']
        assert table[0]['table_members']==['-QF3','-QF4']
        assert [c['reference_quantity'] for c in table]==[99,None,0]
        for c in table:
            box,region=c['norm_bbox'],c['reference_row_box']
            assert region[0]<=box[0]<box[2]<=region[2] and region[1]<=box[1]<box[3]<=region[3]
        assert all(a['norm_bbox'][3]<b['norm_bbox'][1] for a,b in zip(table,table[1:]))
        reference_ids={c['som_label'] for c in table}
        assert not any(reference_ids.intersection(b['node_labels']) for b in topo['branches'])
        assert before==[r.to_dict() for r in rows]
        assert len(topo['candidates'])==(3 if whole else 4)
    finally:doc.close()


def test_membership_metadata_round_trips_without_splitting_rows_or_quantities():
    doc,_,rows=drawing()
    try:
        result=ExtractionResult('members','test.pdf','Test','','PART_LIST_MODE',1,itemized_bom=rows,declared_parts=copy.deepcopy(rows))
        data=result.to_dict()
        assert len(data['itemized_bom'])==3 and data['itemized_bom'][0]['quantity']==99
        assert data['itemized_bom'][0]['tag']=='QF3~4' and data['itemized_bom'][0]['tag_members']==['-QF3','-QF4']
        assert ExtractionResult.from_dict(data).to_dict()==data
        result.itemized_bom[0].tag='QF7,8';result.itemized_bom[0].tags=['QF7,8']
        changed=result.to_dict()
        assert changed['itemized_bom'][0]['tag_members']==['-QF7','-QF8']
        assert changed['declared_parts'][0]['tag_members']==['-QF3','-QF4']
    finally:doc.close()


def test_topology_endpoint_uses_literal_table_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',lambda page,spans,**kw:spans)
    doc,_,rows=drawing();source=tmp_path/'source.pdf';doc.save(source);doc.close()
    result=ExtractionResult('tablemarkers','source.pdf','Test','','PART_LIST_MODE',1,
        pages=[PageInfo(1,'PART_LIST','Parts table',width=400,height=300)],itemized_bom=copy.deepcopy(rows),declared_parts=rows,
        summary={'table_extraction':{'policy':'parts-table-only-v1'}})
    before=copy.deepcopy(result.to_dict());original=source.read_bytes()
    backend.store.results[result.project_id]=result;backend.store.pdf_paths[result.project_id]=source
    with TestClient(backend.app) as client:
        response=client.get('/api/projects/tablemarkers/pages/1/topology?view=original')
        assert response.status_code==200
        assert [c['tag'] for c in response.json()['candidates']]==['QF3~4','QF5','WL1']
        assert not response.json()['branches']
    assert result.to_dict()==before and source.read_bytes()==original
