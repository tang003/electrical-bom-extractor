import copy
import hashlib
import json

import pytest
from fastapi import HTTPException

import backend.app as backend
from bom_extractor.ai_agent import preserve_human_state
from bom_extractor.reconciliation import reconcile, supply_note
from tests.test_independent_workflow import item, project, register


def accessory_project(native_model='COIL-175'):
    native = item('+A-K12', native_model)
    native.name = '中间继电器'
    main = item('+A-K12', 'COIL-175', identifier='listed-main')
    main.name, main.placement = '继电器', '=A/6.1:D'
    base = item('+A-K12', 'BASE-822', identifier='listed-base')
    base.name, base.placement = '继电器底座', main.placement
    return project([native], [main, base])


def test_same_device_accessory_has_own_purchase_row_and_survives_reanalysis(tmp_path):
    result = accessory_project()
    main, accessory = result.reconciliation
    assert main['status']=='MATCH' and accessory['status']=='TABLE_ACCESSORY'
    register(result,tmp_path)
    def act(action):
        return backend.reconciliation_action(result.project_id,accessory['id'],
            backend.ReconciliationAction(action=action,signature=accessory['signature']))
    with pytest.raises(HTTPException) as err:
        act('ADOPT_MODEL')
    assert err.value.status_code == 409
    act('ADD_REFERENCE')
    with pytest.raises(HTTPException) as err:
        act('ADD_REFERENCE')
    assert err.value.status_code == 409
    fresh=accessory_project()
    preserve_human_state(result,fresh)
    assert [(i.model,i.quantity) for i in fresh.itemized_bom]==[('COIL-175',1),('BASE-822',None)]
    assert fresh.itemized_bom[1].parent_tag=='+A-K12'
    assert [i.model for i in fresh.independent_bom]==['COIL-175']


def test_unknown_main_can_be_reviewed_without_merging_with_accessory():
    result=accessory_project('未提取到型号')
    assert 'MODEL_INCOMPLETE' in result.reconciliation[0]['issues']
    assert result.reconciliation[1]['status']=='TABLE_ACCESSORY'
    result.declared_parts[0].unit=''
    result.declared_parts[0].quantity=2
    assert 'QUANTITY_MISMATCH' in reconcile(result)[0]['issues']


def test_supplier_note_is_evidence_and_only_material_changes_reopen_decisions():
    result=project([item()], [item(identifier='listed')])
    original=result.reconciliation[0]
    old_facts=[[{k:v for k,v in f.items() if k not in ('supply_note','remark')} for f in facts]
               for facts in [original['schematic'],original['declared']]]
    legacy=hashlib.sha256(json.dumps(old_facts+[original['status'],original['issues']],
                                     ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    original.update(signature=legacy,decision='CONFIRM')
    result.declared_parts[0].remark='液压站部件'
    neutral=reconcile(result)[0]
    assert neutral['signature']==legacy and neutral['decision']=='CONFIRM'
    assert neutral['declared'][0]['remark']=='液压站部件' and not neutral['declared'][0]['supply_note']
    result.declared_parts[0].remark='液压站部件，业主自备'
    changed=reconcile(result)[0]
    assert changed['status']=='DIFFERENCE' and changed['decision']=='PENDING' and changed['previous_decision']=='CONFIRM'
    assert changed['declared'][0]['supply_note']=='业主自备' and 'SUPPLY_SCOPE_REVIEW' in changed['issues']
    assert result.itemized_bom[0].quantity==1


@pytest.mark.parametrize(('text','supply'),[('甲供','甲供'),('资料由甲方提供',''),('待提供',''),('客户自备','客户自备'),
    ('变频柜；由某某机电供货','由某某机电供货'),('安装尺寸待核对',''),('扬程20m，水泵厂提供','水泵厂提供')])
def test_supply_note_preserves_only_the_explicit_supply_clause(text,supply):
    assert supply_note(text)==supply


@pytest.mark.parametrize('cause',['placement','alternate_main','duplicate_part','known_conflict','multi_scope'])
def test_ambiguous_table_rows_are_not_adoptable_as_accessories(cause):
    result=accessory_project()
    main,base=result.declared_parts
    if cause=='placement':base.placement='=B/8.1:D'
    if cause=='alternate_main':base.name='继电器'
    if cause=='duplicate_part':base.model=main.model
    if cause=='known_conflict':result.independent_bom[0].model='COIL-175+BASE-822'
    if cause=='multi_scope':
        for row in result.declared_parts:row.tag='-K12';row.tags=['-K12']
        other=copy.deepcopy(result.independent_bom[0]);other.id='B';other.tag='+B-K12';other.tags=[other.tag]
        result.independent_bom.append(other)
    assert {r['status'] for r in reconcile(result)}=={'AMBIGUOUS'}
