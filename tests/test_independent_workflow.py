from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi import HTTPException
import openpyxl

import backend.app as backend
from bom_extractor.ai_agent import ElectricalDrawingAgent, merge_independent_observations
from bom_extractor.classifier import classify_pdf_pages, extract_visual_spans
from bom_extractor.independent import POLICY, VIEW_NAME, analysis_source, prepare_independent_view
from bom_extractor.models import BOMItem, EvidenceLocation, ExtractionResult, PageInfo, SourceType
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.reconciliation import reconcile, tag_members
from bom_extractor.vision import VisionRecognizer


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,'get_ocr',lambda:None)


def drawing(path, model='SECRET-999', mixed=True, rotation=0):
    doc=pymupdf.open();p=doc.new_page(width=900,height=600)
    def put(x,y,text):p.insert_text((x,y),text,fontname='china-s',fontsize=10)
    if mixed:
        put(70,180,'-QF1');put(70,200,'EZD100E');put(70,220,'3P 63A')
    put(465,80,'元件清单')
    for x,text in [(340,'序号'),(410,'代号'),(480,'名称'),(610,'型号'),(740,'数量'),(800,'单位')]:put(x,105,text)
    for y,seq,tag in [(140,'1','-QF1'),(180,'2','-K2')]:
        for x,text in [(340,seq),(410,tag),(480,'断路器'),(610,model),(740,'8'),(800,'个')]:put(x,y,text)
    p.set_rotation(rotation);doc.save(path);doc.close()
    return path


def item(tag='-K1',model='OLD-100',qty=1,identifier='s1'):
    return BOMItem(identifier,1,tag,tags=[tag],name='器件',model=model,raw_model=model,quantity=qty,
        evidences=[EvidenceLocation(1,[50,50,100,100],[.1,.1,.2,.2],raw_text=tag+' '+model,role='device')],model_origin='schematic')


def project(schematic=None,declared=None):
    frozen=schematic if schematic is not None else [item()]
    result=ExtractionResult('proj_independent','test.pdf','测试','','SCHEMATIC_MODE',1,
        pages=[PageInfo(1,'CONTROL_CIRCUIT','控制回路',width=500,height=500)],
        itemized_bom=copy.deepcopy(frozen),independent_bom=copy.deepcopy(frozen),declared_parts=declared or [],
        summary={'independent_analysis':{'policy':POLICY,'schematic_pages':[1],'excluded_table_pages':[],'mixed_table_pages':[]},'declared_reference_status':'loaded'})
    reconcile(result);return result


@pytest.mark.parametrize('rotation',[0,90,180,270])
def test_mixed_page_masks_pixels_native_text_and_ai_prompt(tmp_path,monkeypatch,rotation):
    source=drawing(tmp_path/'source.pdf',rotation=rotation)
    with pymupdf.open(source) as doc:infos=classify_pdf_pages(doc)[3]
    view=tmp_path/VIEW_NAME
    plan=prepare_independent_view(source,view,infos)
    assert plan['mixed_table_pages']==[1] and plan['schematic_pages']==[1]
    calls=[]
    vision=VisionRecognizer(api_key='test-key')
    def capture(content,**kwargs):
        calls.append(content)
        return {'data':{'page':1,'sheet_no':'','devices':[],'review_reasons':[]},'usage':{}}
    monkeypatch.setattr(vision,'request_json',capture)
    with pymupdf.open(view) as doc:
        p=doc[0];text=' '.join(s['text'] for s in extract_visual_spans(p))
        assert 'EZD100E' in text and 'SECRET' not in text
        assert 'SECRET' not in p.get_text()
        box=plan['pages'][0]['regions'][0];w,h=p.rect.width,p.rect.height
        pix=p.get_pixmap(alpha=False)
        import numpy as np
        pixels=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
        crop=pixels[int(box[1]*h)+2:int(box[3]*h)-2,int(box[0]*w)+2:int(box[2]*w)-2]
        assert crop.size and crop.min()==255,'AI rendered pixels must not contain table glyphs'
        ElectricalDrawingAgent(vision,tmp_path/'cache').extract_page(p,infos[0],plan['view_sha256'],effort='medium')
    prompt=' '.join(c.get('text','') for c in calls[0])
    assert 'SECRET' not in prompt and 'EZD100E' in prompt
    assert len([c for c in calls[0] if c['type']=='input_image'])==5


def test_table_changes_cannot_change_schematic_facts_and_snapshot_is_saved_first(tmp_path,monkeypatch):
    from bom_extractor import cad_part_list
    original=cad_part_list.extract_cad_part_lists
    order=[]
    def reference(doc,infos,**kwargs):
        folder=Path(doc.name).parent/'out'
        assert (folder/'independent_snapshot.json').is_file()
        order.append(json.loads((folder/'independent_snapshot.json').read_text(encoding='utf-8')))
        return original(doc,infos,**kwargs)
    monkeypatch.setattr(cad_part_list,'extract_cad_part_lists',reference)
    results=[]
    for n,model in enumerate(['SECRET-999','DIFFERENT-777']):
        folder=tmp_path/str(n);folder.mkdir()
        source=drawing(folder/'source.pdf',model=model)
        results.append(run_extraction_pipeline(source,output_dir=folder/'out',project_id='same'))
    assert len(order)==2
    assert results[0].independent_bom and results[0].declared_parts
    assert [i.to_dict() for i in results[0].independent_bom]==[i.to_dict() for i in results[1].independent_bom]
    assert results[0].declared_parts[0].model!=results[1].declared_parts[0].model
    assert all(e.role!='table_row' for i in results[0].independent_bom for e in i.evidences)
    workbook=openpyxl.load_workbook(results[0].summary['excel_path'])
    assert set(workbook.sheetnames)=={'物料清单','待确认事项'}
    from bom_extractor.exporter import export_to_excel
    path=tmp_path/'diagnostics.xlsx';export_to_excel(results[0],path,diagnostic=True)
    diagnostics=openpyxl.load_workbook(path)
    assert {'原理图独立快照','图内清单原文','原理图与清单核对'}<=set(diagnostics.sheetnames)


def test_table_only_has_no_independent_bom(tmp_path):
    result=run_extraction_pipeline(drawing(tmp_path/'source.pdf',mixed=False),output_dir=tmp_path/'out')
    assert not result.itemized_bom and not result.independent_bom
    assert len(result.declared_parts)==2
    assert result.summary['independent_analysis']['excluded_table_pages']==[1]
    assert {r['status'] for r in result.reconciliation}=={'TABLE_ONLY'}


def test_reconciliation_handles_scopes_suffixes_ranges_and_unknown_quantities():
    assert tag_members('NFB2/3')==['-NFB2','-NFB3']
    assert tag_members('NFB4~7')==['-NFB4','-NFB5','-NFB6','-NFB7']
    assert tag_members('CR0-CR7')==['-CR'+str(n) for n in range(8)]
    assert tag_members('MC9/11-16/21-23')==['-MC'+str(n) for n in [9,11,12,13,14,15,16,21,22,23]]
    assert tag_members('OCR4A~4F')==['-OCR4'+letter for letter in 'ABCDEF']
    assert tag_members('-KM1-1')==['-KM1-1']
    r=project([item('+A-KM1-1',identifier='a'),item('+B-KM1-1',identifier='b'),item('+A-KM1',identifier='c')],[item('-KM1-1',identifier='d')])
    assert r.reconciliation[0]['status']=='AMBIGUOUS'
    assert r.reconciliation[0]['schematic_ids']==['a','b'] or set(r.reconciliation[0]['schematic_ids'])=={'a','b'}
    r=project([item('-KA'),item('-KA0',identifier='zero')],[item('-KA',identifier='d')])
    assert r.reconciliation[0]['schematic_ids']==['s1']
    r=project([item('-K1',qty=None)],[item('-K1',qty=20,identifier='d')])
    assert 'QUANTITY_INCOMPLETE' in r.reconciliation[0]['issues']
    assert r.independent_bom[0].quantity is None
    r=project([item('-NFB2'),item('-NFB3',identifier='s2')],[item('NFB2/3',qty=9,identifier='d')])
    assert 'QUANTITY_MISMATCH' in r.reconciliation[0]['issues']


def register(result,tmp_path):
    folder=backend.OUTPUT_DIR/result.project_id;folder.mkdir(parents=True,exist_ok=True)
    source=drawing(folder/'source.pdf');backend.store.pdf_paths[result.project_id]=source
    backend.store.results[result.project_id]=result
    return folder


def test_manual_adoption_does_not_mutate_snapshot_or_future_ai_input(tmp_path):
    result=project([item(model='未提取到型号(待核实)')],declared=[item(model='TABLE-200',identifier='d')]);register(result,tmp_path)
    before=copy.deepcopy(result.independent_bom[0].to_dict());row=result.reconciliation[0]
    backend.reconciliation_action(result.project_id,row['id'],backend.ReconciliationAction(signature=row['signature'],action='ADOPT_MODEL',expected_working_model=result.itemized_bom[0].model))
    assert result.itemized_bom[0].model=='TABLE-200'
    assert result.itemized_bom[0].raw_model==before['raw_model']
    assert result.itemized_bom[0].model_origin=='drawing_parts_list_confirmed'
    assert result.independent_bom[0].to_dict()==before
    with pytest.raises(HTTPException) as exc:
        backend.reconciliation_action(result.project_id,row['id'],backend.ReconciliationAction(signature=row['signature'],action='KEEP_SCHEMATIC'))
    assert exc.value.status_code==409
    from tests.test_ai_agent import record,observation
    merge_independent_observations(result,[record(observations=[observation(tag='-K1',model='NEW-123')])])
    assert result.independent_bom[0].model=='NEW-123'
    assert result.itemized_bom[0].model=='TABLE-200'
    reconcile(result)
    assert result.reconciliation[0]['decision']=='PENDING', 'Changed evidence resets the old decision'


def test_stale_actions_manual_reference_add_and_restart_keep_snapshots(tmp_path):
    result=project([],declared=[item(model='TABLE-200',qty=30,identifier='d')]);folder=register(result,tmp_path)
    row=result.reconciliation[0]
    with pytest.raises(HTTPException) as exc:
        backend.reconciliation_action(result.project_id,row['id'],backend.ReconciliationAction(signature='stale',action='CONFIRM'))
    assert exc.value.status_code==409
    backend.reconciliation_action(result.project_id,row['id'],backend.ReconciliationAction(signature=row['signature'],action='ADD_REFERENCE'))
    assert result.itemized_bom[0].quantity is None and result.itemized_bom[0].quantity_candidates==[30]
    assert not result.independent_bom
    state=(folder/'project_state.json').read_bytes()
    restored=backend.ProjectStore();restored.initialize_defaults()
    assert not restored.load_errors
    assert (folder/'project_state.json').read_bytes()==state
    assert restored.results[result.project_id].reconciliation[0]['decision']=='ADD_REFERENCE'
    assert restored.pdf_paths[result.project_id].name=='source.pdf'


def test_missing_or_modified_view_fails_closed_and_legacy_clone_preserves_state(tmp_path):
    result=project();folder=register(result,tmp_path)
    result.summary['output_dir']=str(folder)
    with pytest.raises(ValueError,match='缺失'):analysis_source(result,folder/'source.pdf',folder)
    (folder/VIEW_NAME).write_bytes(b'bad')
    result.summary['independent_analysis']['view_sha256']='wrong'
    with pytest.raises(ValueError,match='变化'):analysis_source(result,folder/'source.pdf',folder)
    result.extraction_mode='PART_LIST_MODE';result.summary.pop('independent_analysis');result.independent_bom=[]
    backend.store.sync_and_persist(result.project_id)
    state=(folder/'project_state.json').read_bytes()
    with pytest.raises(ValueError,match='历史'):analysis_source(result,folder/'source.pdf',folder)
    out=backend._build_independent_project(result.project_id)
    assert out['project_id']!=result.project_id
    assert out['summary']['independent_analysis']['policy']==POLICY
    assert (folder/'project_state.json').read_bytes()==state
    assert 'ai_job_id' not in out['summary']


def test_job_queue_uses_masked_source_excludes_tables_and_counts_eligible_coverage(tmp_path,monkeypatch):
    import backend.ai_jobs as jobs
    from tests.test_ai_agent import record,observation
    folder=backend.OUTPUT_DIR/'proj_queue';folder.mkdir(parents=True)
    a=drawing(tmp_path/'mixed.pdf');b=drawing(tmp_path/'table.pdf',mixed=False)
    with pymupdf.open(a) as doc,pymupdf.open(b) as table:
        doc.insert_pdf(table);doc.save(folder/'source.pdf')
    result=run_extraction_pipeline(folder/'source.pdf',output_dir=folder,project_id='proj_queue')
    backend.store.results[result.project_id]=result;backend.store.pdf_paths[result.project_id]=folder/'source.pdf'
    vision=VisionRecognizer(api_key='test-key')
    monkeypatch.setattr(jobs,'VisionRecognizer',lambda:vision)
    manager=jobs.AIJobManager(lambda:backend.store,lambda:backend.OUTPUT_DIR)
    monkeypatch.setattr(manager.executor,'submit',lambda *a:None)
    with pytest.raises(ValueError,match='整页物料表'):manager.start(result.project_id,[2])
    seen=[]
    def extract(self,page,info,pdf_hash,**kwargs):
        assert Path(page.parent.name).name==VIEW_NAME
        assert 'SECRET' not in ' '.join(s['text'] for s in extract_visual_spans(page))
        seen.append((info.page_num,pdf_hash))
        return record(observations=[observation(tag='-QF1',model='EZD100E')])
    monkeypatch.setattr(ElectricalDrawingAgent,'extract_page',extract)
    job=manager.start(result.project_id)
    assert job['pages']==[1]
    live=manager.jobs[(result.project_id,job['id'])]
    manager.run(live,vision)
    assert live['status']=='completed'
    assert seen and all(h==result.summary['independent_analysis']['view_sha256'] for _,h in seen)
    saved=backend.store.results[result.project_id]
    assert saved.summary['ai_last_run']['coverage']=='FULL_DOCUMENT'
    assert saved.declared_parts and saved.independent_bom
    assert all(i.model!='SECRET-999' for i in saved.independent_bom)
