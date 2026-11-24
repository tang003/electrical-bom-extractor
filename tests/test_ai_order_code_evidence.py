from copy import deepcopy

import pytest

from bom_extractor.ai_agent import PLACEHOLDER, grounded_model, merge_observations
from bom_extractor.models import BOMItem, EvidenceLocation, ReviewItem
from test_ai_agent import observation, project, record


def span(index, text, box):
    return dict(id=index, text=text, norm_bbox=box)


def test_one_annotation_keeps_the_complete_order_code_and_short_literal():
    cpu = observation('-CPU9', '6ES7 288-1ST30')
    page = record(observations=[cpu])
    page['spans'][0]['text']='CPU ST30 DC/DC/DC (6ES7 288-1ST30-0AA1)'
    assert not grounded_model(cpu,page)
    cpu['model_raw']='6ES7 288-1ST30-0AA1'
    assert grounded_model(cpu,page)
    cpu['model_raw']='6ES7'
    assert not grounded_model(cpu,page)
    page['spans'][0]['text']='VX-A880-2050(2.25kW)'
    cpu['model_raw']='VX-A880-2050'
    assert not grounded_model(cpu,page)
    page['spans'][0]['text']='短路保护 ZEB8 100A'
    cpu['model_raw']='ZEB8'
    assert grounded_model(cpu,page)


@pytest.mark.parametrize('text',['RO4A','DC24V','C16A/2P'])
def test_printed_pins_and_ratings_cannot_become_an_order_code(text):
    obs=observation(model=text)
    page=record(observations=[obs]); page['spans'][0]['text']=text
    result=project(); merge_observations(result,[page])
    assert result.itemized_bom[0].model == PLACEHOLDER


def test_split_part_number_needs_adjacent_text_and_no_foreign_owner():
    a=[.21,.30,.25,.31]; b=[.255,.30,.30,.31]
    obs=observation('-K9','MM-7050-R8'); obs['model_span_ids']=[0,1]
    page=record(observations=[obs]); page['spans']=[span(0,'MM-7050',a),span(1,'-R8',b)]
    assert grounded_model(obs,page)
    distant=deepcopy(page);distant['spans'][1]['norm_bbox']=[.37,.37,.39,.38]
    assert not grounded_model(obs,distant)
    original=BOMItem('psu',1,'+PANEL-G8',model='MM-7050',evidences=[
        EvidenceLocation(1,a,a,raw_text='MM-7050',role='model_text',model_binding='enclosure')])
    result=project([original]); merge_observations(result,[page])
    assert next(i for i in result.itemized_bom if i.tag=='+PANEL-K9').model == PLACEHOLDER
    assert original.model=='MM-7050'


def test_two_devices_cannot_claim_one_unowned_model_occurrence():
    first=observation('-K4','XC-472');second=observation('-G5','XC-472')
    page=record(observations=[first,second]); page['spans'][0]['text']='XC-472'
    result=project();merge_observations(result,[page])
    assert len(result.itemized_bom)==2 and all(i.model==PLACEHOLDER for i in result.itemized_bom)
    assert any('多个设备同时引用' in r.description for r in result.review_items)


def test_two_adjacent_complete_codes_do_not_form_a_new_order_code():
    obs=observation('-K9','RL-701PS-840'); obs['model_span_ids']=[0,1]
    page=record(observations=[obs]); page['spans']=[span(0,'RL-701',[.20,.30,.25,.31]),span(1,'PS-840',[.26,.30,.30,.31])]
    assert not grounded_model(obs,page)


def test_localized_ai_model_is_a_proposal_when_native_codes_disagree():
    native=BOMItem('relay',1,'+PANEL-K1',model=PLACEHOLDER,source_type='UNRESOLVED')
    result=project([native])
    result.review_items=[ReviewItem('rev_conflict_+PANEL-K1','CONFLICT','PARAM_MISMATCH','+PANEL-K1','两处标注','NEW-123 / NEW-125','核对原图',1)]
    merge_observations(result,[record()])
    assert native.model == PLACEHOLDER
    assert any(r.issue_type=='AI_MODEL_CONFLICT' and r.proposed_data['model']=='NEW-123' for r in result.review_items)
