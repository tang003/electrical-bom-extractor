"""Body and accessory output stays grounded, separate and reviewable."""
import copy
import importlib

import pymupdf
import pytest
from pydantic import ValidationError
from openpyxl import load_workbook

from bom_extractor.ai_agent import (Observation, observation_schema, merge_observations,
                                   conflict_pages, check_observations, PLACEHOLDER)
from bom_extractor.components import DeviceComponent, validate_components, model_state
from bom_extractor.field_classifier import DeviceFields, validate_fields, decision_summary
from bom_extractor.models import BOMItem, EvidenceLocation, ExtractionResult, ReviewItem
from bom_extractor.procurement import item_readiness
from bom_extractor.rules import build_aggregated_from_itemized
from bom_extractor.exporter import export_to_excel
from test_ai_agent import observation, record, project


def component(kind,model,ids,box,owner='-K1'):
    return DeviceComponent(kind=kind,name={'main':'本体','socket':'底座','contact_block':'触点模块','accessory':'附件'}.get(kind,'部件'),
        owner_tag=owner,model_raw=model,model_span_ids=ids,bbox_normalized=box,
        relationship='integral' if kind=='main' else 'separate_part',relationship_basis='inferred',
        reason='型号为图纸原文，采购关系需核实').model_dump()


def page_with_parts(main='RXG12BD',accessory='RGZE1S35M',kind='socket',number=1):
    obs=observation(model=None)
    obs['components']=[component('main',main,[0],[.205,.225,.39,.275]),
                       component(kind,accessory,[1],[.205,.29,.39,.34])]
    page=record(number,[obs])
    page['spans']=[dict(id=0,text=main,norm_bbox=[.21,.23,.32,.25]),
                   dict(id=1,text=accessory,norm_bbox=[.21,.30,.36,.32]),
                   dict(id=2,text='-K1',norm_bbox=[.21,.205,.28,.22])]
    return page


def native_item(main='RXG12BD',base='RGZE1S35M',extra=None):
    item=BOMItem('native',1,'+PANEL-K1',tags=['+PANEL-K1'],name='器件',model=PLACEHOLDER,
                 raw_model='',source_type='UNRESOLVED',quantity=1)
    for text,box in [(main,[.21,.23,.32,.25]),(base,[.21,.30,.36,.32])]:
        item.evidences.append(EvidenceLocation(1,[],box,raw_text=text,role='model_text',device_tag=item.tag))
    if extra:item.evidences.append(EvidenceLocation(2,[],[.4,.4,.5,.5],raw_text=extra,role='model_text',device_tag=item.tag))
    return item


def test_both_ai_paths_have_structured_components_and_forbid_invented_quantity():
    assert 'components' in observation_schema()['$defs']['Observation']['properties']
    assert 'components' in DeviceFields.model_json_schema()['properties']
    raw=page_with_parts()['data']['devices'][0]['components'][0]
    with pytest.raises(ValidationError): DeviceComponent.model_validate({**raw,'quantity':3})
    assert Observation.model_validate(observation()).components==[]


@pytest.mark.parametrize(('main','accessory','kind'),[
    ('RXG12BD','RGZE1S35M','socket'),('XB2BA31','ZBE101','contact_block'),('DRV-600-A','FLT-12-C','accessory')])
def test_whole_page_keeps_parts_separate_without_accessory_quantity(main,accessory,kind):
    result=project();page=page_with_parts(main,accessory,kind)
    merge_observations(result,[page,page_with_parts(main,accessory,kind,2)])
    item,=result.itemized_bom
    assert item.model==main
    assert len(item.component_parts)==2
    assert {p['model'] for p in item.component_parts}=={main,accessory}
    assert all(p['quantity'] is None for p in item.component_parts)
    assert item.quantity==1
    assert len(item.component_parts[0]['evidences'])==2  # Two pages, one part candidate.
    assert '本体/附件待核' in item_readiness(item)['issues']
    assert not any(r.issue_type=='AI_MODEL_CONFLICT' for r in result.review_items)


def test_native_body_and_socket_candidates_become_a_component_review():
    item=native_item();result=project([item])
    conflict=ReviewItem('rev_conflict_+PANEL-K1','CONFLICT','PARAM_MISMATCH',item.tag,'型号冲突','','',1)
    result.review_items=[conflict]
    merge_observations(result,[page_with_parts()])
    assert item.model=='RXG12BD'
    assert conflict.issue_type=='COMPONENT_RELATION' and conflict.status=='PENDING'
    assert len(result.itemized_bom)==1 and item.quantity==1


def test_unexplained_cross_page_model_is_not_silently_removed():
    item=native_item(extra='CONN-9-K');result=project([item])
    result.review_items=[ReviewItem('rev_conflict_+PANEL-K1','CONFLICT','PARAM_MISMATCH',item.tag,'冲突','','',1)]
    merge_observations(result,[page_with_parts()])
    assert item.model==PLACEHOLDER
    assert len(item.component_parts)==2
    assert any(e.raw_text=='CONN-9-K' for e in item.evidences)
    assert model_state(item)['code']=='ASSOCIATION_PENDING'


def test_different_body_models_still_produce_a_real_conflict():
    item=native_item();result=project([item])
    records=[page_with_parts(),page_with_parts(main='RXG22BD',number=2)]
    assert conflict_pages(records,{p.page_num:p for p in result.pages})
    merge_observations(result,records)
    assert item.model==PLACEHOLDER
    assert any(r.issue_type=='AI_MODEL_CONFLICT' for r in result.review_items)


def test_human_edits_and_existing_components_are_protected():
    item=native_item();item.source_type='MANUAL_EDITED';item.model='USER-9'
    result=project([item]);original=copy.deepcopy(item.to_dict())
    merge_observations(result,[page_with_parts()])
    assert item.to_dict()==original


@pytest.mark.parametrize('bad',['invented','foreign','missing','off_page','off_box','neighbor','same_span','printed_relation','inferred_kind_refs','rating'])
def test_bad_part_evidence_is_rejected(bad):
    page=page_with_parts();parts=page['data']['devices'][0]['components'];neighbors=[]
    if bad=='invented':parts[1]['model_raw']='INVENTED-9'
    if bad=='foreign':parts[1]['owner_tag']='-K2'
    if bad=='missing':parts[1]['model_span_ids']=[99]
    if bad=='off_page':page['spans'][1]['page']=2
    if bad=='off_box':parts[1]['bbox_normalized']=[.6,.6,.8,.8]
    if bad=='neighbor':neighbors=[{'tag':'-K2','box':[.2,.28,.39,.34]}]
    if bad=='same_span':parts[1].update(model_span_ids=[0],model_raw='RXG12BD',bbox_normalized=[.205,.225,.39,.275])
    if bad=='printed_relation':parts[1]['relationship_basis']='printed'
    if bad=='printed_kind':parts[1].update(kind_basis='printed',kind_span_ids=[1])
    if bad=='inferred_kind_refs':parts[1]['kind_span_ids']=[1]
    if bad=='rating':parts[1]['model_raw']='24VDC';page['spans'][1]['text']='24VDC'
    with pytest.raises(ValueError):
        validate_components(parts,'-K1',page['spans'],1,[.2,.2,.4,.4],neighbors)


def field_context():
    page=page_with_parts()
    return {'tag':'-K1','page':1,'anchor_box':[.21,.205,.28,.22],'focus_box':[.2,.2,.4,.4],
            'context_box':[.1,.1,.5,.5],'text_spans':page['spans'], 'neighbor_tags':[],
            'existing_evidence':[], 'selection_status':'resolved'}


def field_decision():
    return DeviceFields(tag='-K1',assignments=[{'span_id':i,'role':'model','owner_tag':'-K1'} for i in [0,1]],
        model_raw=None,model_span_ids=[],material_name='中间继电器',name_source='inferred',material_name_span_ids=[],
        specification=None,specification_span_ids=[],brand_explicit=None,brand_span_ids=[],visual_brand_candidate=None,
        uncertain=False,reason='分别识别本体和底座',components=page_with_parts()['data']['devices'][0]['components'])


def test_single_device_uses_main_model_and_keeps_socket_citations_separate():
    context=field_context();decision=validate_fields(field_decision().model_dump(),context)
    summary=decision_summary(decision,context)
    assert summary['model']=='RXG12BD' and summary['can_propose_update']
    assert summary['can_propose_components'] and len(summary['components'])==2
    assert summary['components'][1]['evidences'][0]['raw_text']=='RGZE1S35M'
    broken=field_decision().model_dump();broken['assignments'][1]['owner_tag']='-K2'
    with pytest.raises(ValueError,match='归属'): validate_fields(broken,context)


def test_component_references_are_checked_before_whole_page_merge():
    page=page_with_parts();page['data']['devices'][0]['components'][1]['model_span_ids']=[99]
    assert any('部件证据' in s for s in check_observations(page,project().pages[0]))


def test_component_data_survives_restore_aggregation_and_excel(tmp_path):
    result=project();merge_observations(result,[page_with_parts()])
    restored=ExtractionResult.from_dict(result.to_dict())
    assert restored.itemized_bom[0].component_parts==result.itemized_bom[0].component_parts
    second=copy.deepcopy(restored.itemized_bom[0]);second.id='another';second.tag='+PANEL-K8';second.tags=[second.tag]
    for part in second.component_parts:
        part['id']+='other';part['owner_tag']=second.tag
    restored.itemized_bom.append(second)
    restored.aggregated_bom=build_aggregated_from_itemized(restored.itemized_bom)
    assert sum(len(i.component_parts) for i in restored.aggregated_bom)==4
    path=tmp_path/'parts.xlsx';export_to_excel(restored,path)
    workbook=load_workbook(path,read_only=True)
    text='\n'.join(str(v) for sheet in workbook for row in sheet.iter_rows(values_only=True) for v in row if v is not None)
    assert '底座：RGZE1S35M' in text
    workbook.close()


def test_single_device_acceptance_saves_components_without_adding_purchase_quantity(tmp_path,monkeypatch):
    server=importlib.import_module('backend.app');pdf=tmp_path/'source.pdf'
    with pymupdf.open() as doc:doc.new_page(width=600,height=800);doc.save(pdf)
    item=native_item();item.tag='-K1';item.tags=['-K1']
    result=project([item]);result.project_id='parts-test'
    monkeypatch.setattr(server.store,'results',{result.project_id:result})
    monkeypatch.setattr(server.store,'pdf_paths',{result.project_id:pdf})
    monkeypatch.setattr(server.store,'sync_and_persist',lambda *args,**kwargs:result)
    monkeypatch.setattr(server.ai_jobs,'jobs',{})
    monkeypatch.setattr(server,'analysis_source',lambda *args:pdf)
    monkeypatch.setattr(server,'build_context',lambda *args:field_context())
    monkeypatch.setattr(server,'OUTPUT_DIR',tmp_path)
    monkeypatch.setattr(server,'request_qwen',lambda *args:(field_decision(),{}))
    response=server.classify_item_text(result.project_id,item.id)
    assert item.model==PLACEHOLDER and item.component_parts==[]
    server.handle_review_action(result.project_id,response['review_id'],server.ReviewActionPayload(action='ACCEPT'))
    assert item.model=='RXG12BD' and len(item.component_parts)==2
    assert all(p['status']=='CONFIRMED' and p['quantity'] is None for p in item.component_parts)
    assert item.quantity==1 and len(result.itemized_bom)==1
    assert all(e.role=='component_model' for e in item.evidences if e.raw_text=='RGZE1S35M')


def test_materializing_a_confirmed_part_is_idempotent_and_leaves_quantity_unknown(monkeypatch):
    server=importlib.import_module('backend.app')
    result=project();result.project_id='parts';merge_observations(result,[page_with_parts()])
    parent=result.itemized_bom[0];part=next(p for p in parent.component_parts if p['kind']=='socket')
    monkeypatch.setattr(server.store,'results',{'parts':result})
    monkeypatch.setattr(server.store,'sync_and_persist',lambda *args,**kwargs:result)
    monkeypatch.setattr(server.ai_jobs,'jobs',{})
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error: server.materialize_component('parts',parent.id,part['id'])
    assert error.value.status_code==409
    part['status']='CONFIRMED'
    server.materialize_component('parts',parent.id,part['id'])
    child=next(i for i in result.itemized_bom if i.id==part['materialized_item_id'])
    assert child.model=='RGZE1S35M' and child.quantity is None
    assert child.parent_tag==parent.tag and child.tag==''
    assert parent.quantity==1
    server.materialize_component('parts',parent.id,part['id'])
    assert len(result.itemized_bom)==2
    result.itemized_bom.remove(child)
    with pytest.raises(HTTPException): server.materialize_component('parts',parent.id,part['id'])


def test_split_order_code_is_allowed_but_two_order_codes_cannot_become_one_kit():
    page=page_with_parts();parts=page['data']['devices'][0]['components'][:1]
    parts[0].update(model_raw='ABC-123-X',model_span_ids=[0,3])
    page['spans'][0].update(text='ABC-123-',norm_bbox=[.21,.23,.29,.25])
    page['spans'].append(dict(id=3,text='X',norm_bbox=[.29,.23,.31,.25]))
    assert validate_components(parts,'-K1',page['spans'],1,[.2,.2,.4,.4])[0]['model']=='ABC-123-X'
    page['spans'][0]['text']='ABC-123';page['spans'][-1]['text']='XYZ-456'
    parts[0]['model_raw']='ABC-123XYZ-456'
    with pytest.raises(ValueError):validate_components(parts,'-K1',page['spans'],1,[.2,.2,.4,.4])


def test_type_without_a_literal_name_is_downgraded_without_losing_grounded_model():
    page=page_with_parts();parts=page['data']['devices'][0]['components']
    parts[1].update(kind_basis='printed',kind_span_ids=[1])
    saved=validate_components(parts,'-K1',page['spans'],1,[.2,.2,.4,.4])
    assert saved[1]['model']=='RGZE1S35M' and saved[1]['kind_basis']=='inferred'
    assert saved[1]['status']=='PENDING'
    assert all(ev['role']!='kind' for ev in saved[1]['evidences'])


def test_symbol_only_part_box_is_extended_to_verified_labels_below_it():
    context=field_context();raw=field_decision().model_dump()
    for part in raw['components']:
        part['bbox_normalized']=[.205,.205,.39,.225]
    # The names and order codes lie outside the AI's symbol-only box, but all
    # cited text is from the same device, page and verified owner region.
    checked=validate_fields(raw,context)
    summary=decision_summary(checked,context)
    main,socket=summary['components']
    assert summary['model']=='RXG12BD' and summary['can_propose_components']
    assert main['bbox_normalized']==[.205,.205,.39,.25]
    assert socket['bbox_normalized']==[.205,.205,.39,.32]
    assert all(p['quantity'] is None and p['status']=='PENDING' for p in summary['components'])
    assert checked.components[1].bbox_normalized==socket['bbox_normalized']
    # Validation/summary may run twice; repairing the box must be idempotent.
    assert decision_summary(validate_fields(checked.model_dump(),context),context)==summary


def test_whole_page_merge_also_includes_part_labels_below_symbol():
    result=project();page=page_with_parts()
    for part in page['data']['devices'][0]['components']:
        part['bbox_normalized']=[.205,.205,.39,.225]
    merge_observations(result,[page])
    item,=result.itemized_bom
    assert item.model=='RXG12BD' and len(item.component_parts)==2
    assert item.component_parts[1]['bbox_normalized'][3]==.32
    assert item.quantity==1


@pytest.mark.parametrize('bad',['outside_owner','other_page','neighbor'])
def test_box_repair_never_admits_text_from_outside_the_owner(bad):
    context=field_context();parts=field_decision().model_dump()['components']
    for part in parts:part['bbox_normalized']=[.205,.205,.39,.225]
    span=context['text_spans'][1]
    if bad=='outside_owner':span['norm_bbox']=[.41,.30,.49,.32]
    if bad=='other_page':span['page']=2
    if bad=='neighbor':context['neighbor_tags']=[{'tag':'-K2','box':[.20,.29,.39,.34],'page':1}]
    with pytest.raises(ValueError):
        validate_components(parts,context['tag'],context['text_spans'],1,
                            context['focus_box'],context['neighbor_tags'])


def test_order_codes_do_not_count_as_printed_proof_of_component_relationship():
    context=field_context();raw=field_decision().model_dump()
    raw['components'][1].update(relationship_basis='printed',relationship_span_ids=[0])
    checked=validate_fields(raw,context)
    summary=decision_summary(checked,context)
    socket=summary['components'][1]
    assert socket['model']=='RGZE1S35M' and socket['relationship_basis']=='inferred'
    assert checked.components[1].relationship_basis=='inferred'
    assert checked.components[1].relationship_span_ids==[]
    assert all(ev['role']!='relationship' for ev in socket['evidences'])
    assert '关系按推断保留待核' in socket['reason']


@pytest.mark.parametrize('assignment',['wrong_role','missing'])
def test_verified_component_fields_reconcile_redundant_line_roles(assignment):
    context=field_context();raw=field_decision().model_dump()
    if assignment=='wrong_role':raw['assignments'][1]['role']='rating'
    else:raw['assignments'].pop()
    checked=validate_fields(raw,context)
    socket_assignment=next(a for a in checked.assignments if a.span_id==1)
    assert socket_assignment.role=='model' and socket_assignment.owner_tag=='-K1'
    assert decision_summary(checked,context)['components'][1]['model']=='RGZE1S35M'
    assert validate_fields(checked.model_dump(),context).model_dump()==checked.model_dump()


def test_reconciling_line_roles_cannot_validate_an_invented_model():
    context=field_context();raw=field_decision().model_dump()
    raw['assignments'][1]['role']='rating'
    raw['components'][1]['model_raw']='INVENTED-9'
    with pytest.raises(ValueError,match='逐字原文证据'):
        validate_fields(raw,context)
