from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from bom_extractor.ai_agent import ElectricalDrawingAgent, Observation, PageObservations, merge_observations, preserve_human_state, observation_schema, annotation_contains_model
from bom_extractor.models import BOMItem, ExtractionResult, PageInfo, SourceType, ReviewItem, EvidenceLocation
from bom_extractor.vision import VisionRecognizer


def info(n=1, location="+PANEL"):
    return PageInfo(n,"CONTROL_CIRCUIT","控制回路",sheet_no=str(n),width=600,height=800,location_code=location)


def observation(tag="-K1", model="NEW-123", role="device"):
    return Observation(local_tag=tag,name="测试元件",model_raw=model,brand_explicit=None,specification="",
        role=role,scope="unknown",parent_tag=None,model_span_ids=[0] if model else [],
        bbox_normalized=[.2,.2,.4,.4],evidence_text=f"{tag} {model or ''}").model_dump()


def record(n=1, observations=None):
    return {"page":n,"data":{"page":n,"sheet_no":str(n),"devices":observations or [observation()],"review_reasons":[]},
        "spans":[{"id":0,"text":"NEW-123","norm_bbox":[.2,.2,.4,.4]}],"checks":[],"usage":{"input_tokens":10,"output_tokens":5}}


def project(items=None):
    return ExtractionResult("proj_ai","test.pdf","测试","","SCHEMATIC_MODE",2,pages=[info(),info(2)],itemized_bom=items or [])


def test_responses_sends_explicit_effort_and_image_without_temperature(monkeypatch):
    sent=[]
    def post(session,url,**kwargs):
        sent.append((url,kwargs))
        response={"status":"completed","model":"gpt-6.1-sol","reasoning":{"effort":"high"},"usage":{"input_tokens":10},
            "output":[{"type":"message","content":[{"type":"output_text","text":json.dumps(record()['data'])}]}]}
        return SimpleNamespace(ok=True,json=lambda:response)
    import requests
    monkeypatch.setattr(requests.Session,"post",post)
    vision=VisionRecognizer(api_key="test-key",base_url="http://localhost:60563/v1",model="gpt-6.1-sol")
    out=vision.request_json([{"type":"input_image","image_url":"data:image/png;base64,AAAA","detail":"high"}],effort="high",schema=observation_schema())
    payload=sent[0][1]['json']
    assert sent[0][0].endswith('/responses')
    assert payload['reasoning']=={'effort':'high'}
    assert payload['store'] is False and 'temperature' not in payload
    assert payload['input'][0]['content'][0]['type']=='input_image'
    assert payload['text']['format']['strict'] is True
    assert out['reported_effort']=='high'


def test_truncated_response_never_becomes_a_success(monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session,"post",lambda *a,**k:SimpleNamespace(ok=True,json=lambda:{'status':'incomplete','output':[]}))
    with pytest.raises(RuntimeError,match='截断'):
        VisionRecognizer(api_key='test-key').request_json([],effort='medium',schema=observation_schema())


def test_proxy_cannot_silently_override_requested_effort(monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session,'post',lambda *a,**k:SimpleNamespace(ok=True,json=lambda:{'status':'completed','reasoning':{'effort':'low'},'output':[]}))
    with pytest.raises(RuntimeError,match='思考程度'):
        VisionRecognizer(api_key='test-key').request_json([],effort='high',schema=observation_schema())


@pytest.mark.parametrize('box',[[.4,.1,.2,.5],[0,0,float('nan'),1],[-.1,0,.2,.4]])
def test_bad_model_coordinates_are_rejected(box):
    raw=observation()
    raw['bbox_normalized']=box
    with pytest.raises(ValueError):
        Observation.model_validate(raw)


def test_cache_separates_efforts_and_reuses_success(tmp_path,monkeypatch):
    from bom_extractor import ai_agent
    calls=[]
    vision=VisionRecognizer(api_key='test-key',model='gpt-6.1-sol')
    monkeypatch.setattr(vision,'request_json',lambda content,**kw:(calls.append(kw['effort']) or {'data':record()['data'],'usage':{}}))
    monkeypatch.setattr(ai_agent,'page_evidence',lambda page:record()['spans'])
    monkeypatch.setattr(ai_agent,'image_content',lambda page:[])
    agent=ElectricalDrawingAgent(vision,tmp_path)
    assert not agent.extract_page(None,info(),'source-hash',effort='medium')['cache_hit']
    assert agent.extract_page(None,info(),'source-hash',effort='medium')['cache_hit']
    agent.extract_page(None,info(),'source-hash',effort='high')
    assert calls==['medium','high']
    assert 'test-key' not in ''.join(p.read_text(encoding='utf-8') for p in tmp_path.iterdir())


def test_page_ai_receives_document_device_inventory(tmp_path, monkeypatch):
    from bom_extractor import ai_agent
    sent=[]
    vision=VisionRecognizer(api_key='test-key',model='gpt-6.1-sol')
    monkeypatch.setattr(vision,'request_json',lambda content,**kwargs:
                        (sent.append(content[0]['text']) or {'data':record()['data'],'usage':{}}))
    monkeypatch.setattr(ai_agent,'page_evidence',lambda page:record()['spans'])
    monkeypatch.setattr(ai_agent,'image_content',lambda page:[])
    agent=ElectricalDrawingAgent(vision,tmp_path)
    agent.inventory={'version':1,'devices':[{'tag':'+PANEL-F1','family':'F',
        'occurrences':[{'page':1,'bbox':[.2,.2,.3,.3],'text':'-F1'},
                       {'page':2,'bbox':[.4,.4,.5,.5],'text':'-F1'}]}]}

    result=agent.extract_page(None,info(),'source-hash',effort='medium')
    assert result['inventory_targets'][0]['tag']=='+PANEL-F1'
    assert '"page": 2' in sent[0]
    assert any('+PANEL-F1' in issue for issue in result['checks'])


def test_coils_contacts_and_pages_do_not_multiply_purchase_count():
    result=project()
    records=[record(1,[observation(role='coil'),observation(role='contact')]),record(2)]
    counts=merge_observations(result,records)
    assert counts['added']==1
    assert len(result.itemized_bom)==1
    assert result.itemized_bom[0].quantity==1
    assert {e.page for e in result.itemized_bom[0].evidences}=={1,2}


def test_same_local_tag_in_different_cabinets_is_not_merged():
    result=project()
    result.pages[1].location_code='+OTHER'
    merge_observations(result,[record(),record(2)])
    assert {it.tag for it in result.itemized_bom}=={'+PANEL-K1','+OTHER-K1'}


def test_unknown_model_is_not_resolved_from_unsupported_visual_text():
    result=project()
    candidate=observation(model='INVENTED-999')
    merge_observations(result,[record(observations=[candidate])])
    assert result.itemized_bom[0].source_type==SourceType.UNRESOLVED.value
    assert result.itemized_bom[0].model!='INVENTED-999'


def test_model_text_of_a_distant_device_is_not_applied():
    result=project()
    data=record()
    data['spans'][0]['norm_bbox']=[.8,.8,.9,.9]
    merge_observations(result,[data])
    assert result.itemized_bom[0].source_type==SourceType.UNRESOLVED.value


def test_ai_preserves_human_model_quantity_and_deletion():
    item=BOMItem('human',1,'+PANEL-K1',tags=['+PANEL-K1'],model='CONFIRMED-77',quantity=4,source_type=SourceType.MANUAL_EDITED.value)
    result=project([item])
    merge_observations(result,[record()])
    assert (item.model,item.quantity)==('CONFIRMED-77',4)
    assert len(result.itemized_bom)==1
    assert any(r.issue_type=='AI_HUMAN_CONFLICT' for r in result.review_items)
    empty=project()
    empty.summary['manually_deleted_tags']=['+PANEL-K1']
    merge_observations(empty,[record()])
    assert empty.itemized_bom==[]


def test_native_placeholder_can_be_filled_without_changing_identity():
    item=BOMItem('native',1,'+PANEL-K1',tags=['+PANEL-K1'],model='未提取到型号(待核实)',source_type=SourceType.UNRESOLVED.value)
    result=project([item])
    merge_observations(result,[record()])
    assert len(result.itemized_bom)==1 and item.id=='native'
    assert item.model=='NEW-123' and item.needs_review


def test_contact_without_model_does_not_conflict_with_known_device():
    item=BOMItem('native',1,'+PANEL-K1',tags=['+PANEL-K1'],model='NEW-123')
    result=project([item])
    merge_observations(result,[record(observations=[observation(model=None,role='contact')])])
    assert item.model=='NEW-123' and item.quantity==1
    assert not any(r.severity=='CONFLICT' for r in result.review_items)


@pytest.mark.parametrize('saved,raw',[
    ('A9D65216','iC65N Vigi+RCBO A9D65216'),
    ('6ES7288-1ST30-0AA1','CPU ST30 DC/DC/DC (6ES7 288-1ST30-0AA1)'),
])
def test_existing_complete_model_matches_a_longer_label(saved,raw):
    assert annotation_contains_model(saved,raw)


@pytest.mark.parametrize('raw',['ABC-1234','ABC-123+BASE-456','ABC-123-OTHER'])
def test_model_suffixes_are_not_lost(raw):
    assert not annotation_contains_model('ABC-123',raw)


def test_terminals_are_retained_with_unknown_count_and_untagged_assemblies_are_reviewed():
    result=project()
    merge_observations(result,[record(observations=[observation(role='terminal'),observation(tag=None,role='assembly')])])
    assert len(result.itemized_bom)==1
    assert result.itemized_bom[0].entity_role=='terminal'
    assert result.itemized_bom[0].quantity is None
    assert any(r.issue_type=='AI_REGION_REVIEW' for r in result.review_items)


def test_ai_page_observation_preserves_grounded_terminal_position_count():
    item=BOMItem('terminal',1,'+PANEL-X9',tags=['+PANEL-X9'],model='PT2.5-PE',
                 raw_model='PT2.5-PE',entity_role='terminal',quantity=13,
                 quantity_status='by_positions',quantity_basis='drawn_terminal_positions',
                 evidences=[EvidenceLocation(1,[100,100,108,106],[.16,.12,.18,.14],
                                             raw_text='1',role='terminal_position')])
    result=project([item])
    ai_record=record(observations=[observation(tag='-X9',model='PT2.5-PE',role='device')])
    ai_record['spans'][0]['text']='PT2.5-PE'
    merge_observations(result,[ai_record])
    assert (item.quantity,item.quantity_status,item.quantity_basis,item.entity_role)==(
        13,'by_positions','drawn_terminal_positions','terminal')


def test_ai_strip_observation_does_not_merge_distinct_terminal_material_rows():
    ordinary=BOMItem('x2_pt6',1,'+PANEL-X2:1~3',tags=['+PANEL-X2:1~3'],model='PT6',
                     raw_model='PT6',name='接线端子',entity_role='terminal',quantity=3,
                     quantity_status='by_positions',quantity_basis='drawn_terminal_positions',
                     terminal_strip_tag='+PANEL-X2',terminal_numbers=[1,2,3],
                     source_type=SourceType.EXPLICIT.value)
    earth=BOMItem('x2_pe',2,'+PANEL-X2:4',tags=['+PANEL-X2:4'],model='PT6-PE',
                 raw_model='PT6-PE',name='接地端子',entity_role='terminal',quantity=1,
                 quantity_status='by_positions',quantity_basis='drawn_terminal_positions',
                 terminal_strip_tag='+PANEL-X2',terminal_numbers=[4],
                 source_type=SourceType.EXPLICIT.value)
    result=project([ordinary,earth])
    ai_record=record(observations=[observation(tag='-X2',model='PT6',role='terminal')])
    ai_record['spans'][0]['text']='PT6'
    merge_observations(result,[ai_record])
    assert [(it.tag,it.model,it.quantity) for it in result.itemized_bom]==[
        ('+PANEL-X2:1~3','PT6',3),('+PANEL-X2:4','PT6-PE',1)]


def test_conflicting_models_are_not_silently_applied():
    result=project()
    other=record(2,[observation(model='OTHER-123')])
    other['spans'][0]['text']='OTHER-123'
    merge_observations(result,[record(),other])
    assert result.itemized_bom==[]
    assert any(r.issue_type=='AI_MODEL_CONFLICT' for r in result.review_items)


def test_part_list_remains_authoritative():
    result=project()
    result.extraction_mode='PART_LIST_MODE'
    merge_observations(result,[record()])
    assert result.itemized_bom==[]
    assert any(r.issue_type=='AI_PART_LIST_CROSSCHECK' for r in result.review_items)


def test_reextract_preserves_edits_and_deleted_accessory_without_deleting_main():
    human=BOMItem('human',3,'+PANEL-K2',tags=['+PANEL-K2'],model='CONFIRMED',quantity=3,source_type=SourceType.MANUAL_EDITED.value)
    previous=project([human])
    previous.summary['manually_deleted_items']=[{'tags':['+PANEL-K1'],'model':'BASE-123'}]
    main=BOMItem('main',1,'+PANEL-K1',tags=['+PANEL-K1'],model='NEW-123')
    accessory=BOMItem('base',2,'+PANEL-K1',tags=['+PANEL-K1'],model='BASE-123')
    fresh=project([main,accessory,BOMItem('auto',3,'+PANEL-K2',tags=['+PANEL-K2'],model='AUTO')])
    preserve_human_state(previous,fresh)
    assert {(x.id,x.model,x.quantity) for x in fresh.itemized_bom}=={('main','NEW-123',1),('human','CONFIRMED',3)}


def test_accepted_multi_tag_review_protects_each_device():
    item=BOMItem('native',1,'+PANEL-K1',tags=['+PANEL-K1'],model='CONFIRMED')
    result=project([item])
    result.review_items.append(ReviewItem('accepted','INFO','CONFIRM','+PANEL-K1, +PANEL-K2','','','',1,status='ACCEPTED'))
    merge_observations(result,[record()])
    assert item.model=='CONFIRMED'


def test_reextract_avoids_id_collisions_and_keeps_review_target():
    item=BOMItem('sch_1',1,'+PANEL-K1',tags=['+PANEL-K1'],model='HUMAN',source_type=SourceType.MANUAL_EDITED.value)
    previous=project([item])
    previous.review_items.append(ReviewItem('accepted','INFO','UPDATE_ITEM','+PANEL-K1','','','',1,status='ACCEPTED',action_type='UPDATE_ITEM',proposed_data={'target_seq_no':1,'model':'HUMAN'}))
    other=BOMItem('sch_1',1,'+PANEL-K2',tags=['+PANEL-K2'],model='AUTO')
    fresh=project([other])
    preserve_human_state(previous,fresh)
    assert len({x.id for x in fresh.itemized_bom})==2
    assert next(x for x in fresh.itemized_bom if x.id=='sch_1').model=='HUMAN'
    data=next(x for x in fresh.review_items if x.id=='accepted').proposed_data
    assert data['target_id']=='sch_1'
    assert next(x for x in fresh.itemized_bom if x.seq_no==data['target_seq_no']).id=='sch_1'


def setup_job(tmp_path,monkeypatch):
    import backend.app as backend
    import backend.ai_jobs as jobs
    pdf=tmp_path/'source.pdf'
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=800)
        doc.new_page(width=600,height=800)
        doc.save(pdf)
    result=project()
    backend.store.results[result.project_id]=result
    backend.store.pdf_paths[result.project_id]=pdf
    vision=VisionRecognizer(api_key='test-key',model='gpt-6.1-sol')
    monkeypatch.setattr(jobs,'VisionRecognizer',lambda:vision)
    manager=jobs.AIJobManager(lambda:backend.store,lambda:backend.OUTPUT_DIR)
    monkeypatch.setattr(manager.executor,'submit',lambda *a,**k:None)
    return backend,manager,vision


def test_job_routes_ambiguity_to_high_and_keeps_valid_page_if_another_fails(tmp_path,monkeypatch):
    backend,manager,vision=setup_job(tmp_path,monkeypatch)
    efforts=[]
    def extract(agent,page,pi,pdf_hash,**kw):
        if pi.page_num==2:
            raise ValueError('invalid page response')
        agent.before_call()
        efforts.append(kw['effort'])
        out=record()
        out['checks']=['原文冲突'] if kw['effort']=='medium' else []
        return out
    monkeypatch.setattr(ElectricalDrawingAgent,'extract_page',extract)
    job=manager.start('proj_ai')
    manager.run(manager.jobs[('proj_ai',job['id'])],vision)
    final=manager.get('proj_ai',job['id'])
    assert efforts==['medium','high']
    assert final['status']=='partial' and final['reviewed_pages']==1
    assert final['api_calls']==2 and final['input_tokens']==20
    assert backend.store.results['proj_ai'].itemized_bom[0].quantity==1


def test_transient_page_retried_after_others_with_no_double_count(tmp_path,monkeypatch):
    from bom_extractor.vision import TransientVisionError
    _,manager,vision=setup_job(tmp_path,monkeypatch)
    visited=[]
    def extract(agent,page,pi,pdf_hash,**kw):
        agent.before_call()
        visited.append(pi.page_num)
        if visited==[1]:
            raise TransientVisionError('timeout')
        return record(n=pi.page_num)
    monkeypatch.setattr(ElectricalDrawingAgent,'extract_page',extract)
    job=manager.start('proj_ai')
    manager.run(manager.jobs[('proj_ai',job['id'])],vision)
    final=manager.get('proj_ai',job['id'])
    assert visited==[1,2,1] and final['completed_pages']==2
    assert final['status']=='completed' and not final['errors'] and final['api_calls']==3


def test_cancel_and_restart_do_not_trigger_unrequested_inference(tmp_path,monkeypatch):
    backend,manager,vision=setup_job(tmp_path,monkeypatch)
    job=manager.start('proj_ai',[1])
    manager.cancel('proj_ai',job['id'])
    manager.run(manager.jobs[('proj_ai',job['id'])],vision)
    assert manager.get('proj_ai',job['id'])['status']=='cancelled'
    assert not backend.store.results['proj_ai'].itemized_bom
    other=manager.start('proj_ai',[1])
    manager.jobs.clear()
    assert manager.get('proj_ai',other['id'])['status']=='interrupted'


@pytest.mark.parametrize('stage', ['save_project', 'submit'])
def test_failed_job_start_can_be_retried(tmp_path, monkeypatch, stage):
    backend, manager, _ = setup_job(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError('simulated startup failure')
    with monkeypatch.context() as m:
        if stage == 'save_project':
            m.setattr(backend.store, 'sync_and_persist', fail)
        else:
            m.setattr(manager.executor, 'submit', fail)
        with pytest.raises(RuntimeError, match='simulated startup failure'):
            manager.start('proj_ai', [1])
    failed, = manager.jobs.values()
    assert failed['status'] == 'failed'
    manager.jobs.clear()
    assert manager.get('proj_ai', failed['id'])['status'] == 'failed'
    retried = manager.start('proj_ai', [1])
    assert retried['id'] != failed['id'] and retried['status'] == 'queued'


def test_review_budget_prioritizes_later_template_quantity_over_early_questions(tmp_path,monkeypatch):
    backend,manager,vision=setup_job(tmp_path,monkeypatch)
    vision.config['max_high_review_pages']=1
    backend.store.results['proj_ai'].pages[1].extraction_diagnostics={'range_annotations':[{'text':'DV02-10','count':9}]}
    calls=[]
    def extract(agent,page,pi,pdf_hash,**kw):
        agent.before_call();calls.append((pi.page_num,kw['effort']))
        out=record(pi.page_num);out['checks']=['需要核查']
        return out
    monkeypatch.setattr(ElectricalDrawingAgent,'extract_page',extract)
    job=manager.start('proj_ai');manager.run(manager.jobs[('proj_ai',job['id'])],vision)
    final=manager.get('proj_ai',job['id'])
    assert calls==[(1,'medium'),(2,'medium'),(2,'high')]
    assert final['deferred_review_pages']==[1]


def test_invalid_config_does_not_replace_private_key():
    VisionRecognizer.save_config({'api_key':'original'})
    with pytest.raises(ValueError):
        VisionRecognizer.save_config({'api_key':'replacement','reasoning_effort':'unlimited'})
    assert VisionRecognizer.load_saved_config()['api_key']=='original'


def test_upload_auto_queues_ai_after_native_result_is_available(tmp_path,monkeypatch):
    import backend.app as backend
    import asyncio, io
    from fastapi import UploadFile
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=800)
        data=doc.tobytes()
    queued=[]
    VisionRecognizer.save_config({'api_key':'test-key','auto_run_on_upload':True})
    monkeypatch.setattr(backend,'run_extraction_pipeline',lambda *a,**kw:ExtractionResult(kw['project_id'],'source.pdf','','','SCHEMATIC_MODE',1,pages=[info()]))
    def queue(pid):
        assert pid in backend.store.results
        assert backend.store.pdf_paths[pid].exists()
        queued.append(pid)
    monkeypatch.setattr(backend.ai_jobs,'start',queue)
    out=asyncio.run(backend.upload_pdf(UploadFile(io.BytesIO(data),filename='new.pdf')))
    assert queued==[out['project_id']]
