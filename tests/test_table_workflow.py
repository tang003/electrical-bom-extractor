from __future__ import annotations

import asyncio
import copy
import io
from pathlib import Path

import ezdxf
import openpyxl
import pymupdf
import pytest
from fastapi import UploadFile,HTTPException
from fastapi.testclient import TestClient

import backend.app as backend
from bom_extractor.cad_import import prepare_cad,native_document
from bom_extractor.independent import analysis_source,legacy_table_project
from bom_extractor.models import SourceType
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.table_workflow import TABLE_MODE,CHECK_MODE,table_policy


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,'get_ocr',lambda:None)


def table_pdf(path,with_table=True,repeat=1,rotation=0):
    doc=pymupdf.open()
    for _ in range(repeat):
        page=doc.new_page(width=900,height=600)
        def text(x,y,t):page.insert_text((x,y),t,fontname='china-s',fontsize=10)
        if with_table:
            # Body text is left aligned while headers are centered: real table geometry matters.
            text(450,70,'部品表')
            edges=[180,370,500,720,790,860]
            for left,right,header in zip(edges,edges[1:],['记号','品名','型式仕样','数量','品牌']):
                page.draw_rect(pymupdf.Rect(left,85,right,110))
                text((left+right)/2-10,102,header)
            for y,tag,name,model,quantity in [(135,'QF1,2','断路器','IC65N Vigi+ 2P C63A','2'),
                                              (165,'FU1~2','熔断器','2A',''),(195,'','插座','A9A06110','0')]:
                for x,t in [(190,tag),(380,name),(510,model),(735,quantity)]:text(x,y,t)
        else:
            text(70,150,'-QF1');text(70,170,'EZD100E')
        page.set_rotation(rotation)
    doc.save(path);doc.close();return path


@pytest.mark.parametrize('rotation',[0,90,180,270])
def test_table_only_keeps_literal_cells_blanks_zero_and_no_schematic_analysis(tmp_path,monkeypatch,rotation):
    import bom_extractor.pipeline as pipeline
    def forbidden(*a,**k):raise AssertionError('Table-only mode must not analyze schematics or apply BOM rules')
    monkeypatch.setattr(pipeline,'extract_from_schematic_pdf',forbidden)
    monkeypatch.setattr(pipeline,'apply_engineering_rules',forbidden)
    result=run_extraction_pipeline(table_pdf(tmp_path/'source.pdf',rotation=rotation),output_dir=tmp_path/'out',workflow_mode=TABLE_MODE)
    assert table_policy(result) and not legacy_table_project(result)
    assert result.summary['workflow_mode']==TABLE_MODE
    assert len(result.itemized_bom)==3 and not result.aggregated_bom
    assert not result.independent_bom and not result.reconciliation
    assert result.itemized_bom[0].tag=='QF1,2'
    assert result.itemized_bom[0].model=='IC65N Vigi+ 2P C63A'
    assert result.itemized_bom[0].quantity==2, 'Grouped tags do not multiply table quantity'
    assert result.itemized_bom[1].quantity is None
    assert result.itemized_bom[1].name=='熔断器' and result.itemized_bom[1].raw_model=='2A'
    assert result.itemized_bom[2].quantity==0, 'Zero rows remain in literal extraction'
    assert result.declared_parts[2].observations[0]['cells']['brand']==''
    assert all(e.role=='table_row' for i in result.itemized_bom for e in i.evidences)
    assert not (tmp_path/'out/schematic_only.pdf').exists()
    with pytest.raises(ValueError,match='仅提取部件表'):analysis_source(result,tmp_path/'source.pdf',tmp_path/'out')
    wb=openpyxl.load_workbook(result.summary['excel_path'])
    assert wb.sheetnames==['物料清单']
    assert wb['物料清单']['E8'].value is None
    assert wb['物料清单']['E9'].value==0
    assert '未经原理图核对' in wb['物料清单']['A4'].value


def test_missing_table_does_not_fallback_and_pdf_duplicates_are_kept(tmp_path):
    empty=run_extraction_pipeline(table_pdf(tmp_path/'drawing.pdf',with_table=False),output_dir=tmp_path/'empty',workflow_mode=TABLE_MODE)
    assert not empty.itemized_bom and empty.summary['table_extraction']['status']=='absent'
    assert empty.review_items and not empty.independent_bom
    multiple=run_extraction_pipeline(table_pdf(tmp_path/'repeat.pdf',repeat=2),output_dir=tmp_path/'multi',workflow_mode=TABLE_MODE)
    assert len(multiple.itemized_bom)==6, 'Identical PDF pages may represent separate assemblies or revisions'
    assert not multiple.aggregated_bom
    chosen=run_extraction_pipeline(tmp_path/'repeat.pdf',output_dir=tmp_path/'chosen',workflow_mode=TABLE_MODE,table_pages=[2])
    assert len(chosen.itemized_bom)==3 and {i.primary_page for i in chosen.itemized_bom}=={2}
    for pages in [[],[0],[3],[True]]:
        with pytest.raises(ValueError):run_extraction_pipeline(tmp_path/'repeat.pdf',output_dir=tmp_path/'bad',workflow_mode=TABLE_MODE,table_pages=pages)


def test_nested_cad_frames_are_collapsed_but_separate_locations_stay_separate(tmp_path):
    doc=ezdxf.new('R2018');space=doc.modelspace()
    for offset in [0,470]:
        for a,b,c,d in [(0,0,420,297),(0,0,420,297),(5,5,415,292)]:
            space.add_lwpolyline([(offset+a,b),(offset+c,b),(offset+c,d),(offset+a,d)],close=True)
        for value,x,y in [('部品表',200,280),('记号',50,260),('品名',140,260),('型式仕样',250,260),('数量',350,260),
                          ('QF1',50,240),('断路器',140,240),('ABC-123',250,240),('2',350,240)]:
            space.add_text(value,dxfattribs={'insert':(offset+x,y),'height':4})
    doc.saveas(tmp_path/'source.dxf')
    pdf=prepare_cad(tmp_path/'source.dxf',tmp_path)
    with pymupdf.open(pdf) as rendered:
        assert len(rendered)==2
        assert all(len(p['overlapping_frame_handles'])==2 for p in native_document(rendered)['pages'])
    result=run_extraction_pipeline(pdf,output_dir=tmp_path/'out',workflow_mode=TABLE_MODE)
    assert len(result.itemized_bom)==2
    assert [i.quantity for i in result.itemized_bom]==[2,2]
    assert not result.aggregated_bom


def test_upload_mode_clone_reextract_and_restart_preserve_originals(tmp_path,monkeypatch):
    source=table_pdf(tmp_path/'input.pdf')
    monkeypatch.setattr(backend.VisionRecognizer,'load_saved_config',lambda:{'auto_run_on_upload':True})
    monkeypatch.setattr(backend.ai_jobs,'start',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No paid AI in table mode')))
    out=asyncio.run(backend.upload_pdf(UploadFile(io.BytesIO(source.read_bytes()),filename='部品表.pdf'),TABLE_MODE))
    pid=out['project_id'];folder=backend.OUTPUT_DIR/pid
    assert out['summary']['workflow_mode']==TABLE_MODE and 'ai_job_id' not in out['summary']
    before=(folder/'project_state.json').read_bytes()
    clone=asyncio.run(backend.create_table_project(pid,backend.TableExtractionRequest(pages=[1])))
    assert clone['project_id']!=pid and clone['summary']['table_extraction']['requested_pages']==[1]
    assert (folder/'project_state.json').read_bytes()==before
    saved=backend.store.results[pid]
    saved.itemized_bom[0].model='MANUAL-999';saved.itemized_bom[0].source_type=SourceType.MANUAL_EDITED.value
    backend.store.sync_and_persist(pid)
    backend.reextract_project(pid)
    saved=backend.store.results[pid]
    assert table_policy(saved) and any(i.model=='MANUAL-999' for i in saved.itemized_bom)
    assert saved.declared_parts[0].raw_model=='IC65N Vigi+ 2P C63A'
    before=(folder/'project_state.json').read_bytes()
    restored=backend.ProjectStore();restored.initialize_defaults()
    assert not restored.load_errors and table_policy(restored.results[pid])
    assert (folder/'project_state.json').read_bytes()==before


def test_reextract_preserves_edit_and_deletion_by_table_page_and_row(tmp_path):
    pdf = table_pdf(tmp_path/'two.pdf', repeat=2)
    response = asyncio.run(backend.upload_pdf(UploadFile(io.BytesIO(pdf.read_bytes()), filename='two.pdf'), TABLE_MODE))
    pid = response['project_id']
    first, _, blank, second, _, other_blank = response['itemized_bom']
    backend.update_bom_item(pid, backend.BOMItemUpdatePayload(id=first['id'], model='P1-EDIT'))
    backend.update_bom_item(pid, backend.BOMItemUpdatePayload(id=blank['id'], quantity=7))
    backend.delete_bom_item(pid, other_blank['id'])
    for _ in range(2):
        fresh = backend.reextract_project(pid)
        rows = fresh['itemized_bom']
        assert [(r['primary_page'], r['tag']) for r in rows] == [
            (1, 'QF1,2'), (1, 'FU1~2'), (1, ''), (2, 'QF1,2'), (2, 'FU1~2')]
        assert rows[0]['model'] == 'P1-EDIT' and rows[2]['quantity'] == 7
        assert rows[3]['model'] == second['model']
        assert len(fresh['declared_parts']) == 6


def test_http_mode_contract_is_validated(tmp_path):
    source=table_pdf(tmp_path/'input.pdf')
    with TestClient(backend.app) as client:
        bad=client.post('/api/projects/upload',files={'file':('test.pdf',source.read_bytes(),'application/pdf')},data={'workflow_mode':'UNKNOWN'})
        assert bad.status_code==422
        good=client.post('/api/projects/upload',files={'file':('test.pdf',source.read_bytes(),'application/pdf')},data={'workflow_mode':TABLE_MODE})
        assert good.status_code==200 and good.json()['summary']['workflow_mode']==TABLE_MODE
        pid=good.json()['project_id']
        assert client.post(f'/api/projects/{pid}/table-extraction',json={'pages':[True]}).status_code==422
