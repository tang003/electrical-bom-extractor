"""Procurement failures tested with unfamiliar tags, not a sample engineer BOM."""
import copy
from pathlib import Path
import pymupdf
import openpyxl
import pytest
from bom_extractor.ai_agent import merge_observations
from bom_extractor.models import BOMItem, PageInfo, SourceType
from bom_extractor.notation import extract_model_strings
from bom_extractor.procurement import item_readiness, flag_template_quantities
from bom_extractor.rules import build_aggregated_from_itemized, load_rules_config
from bom_extractor.schematic_extractor import extract_from_schematic_pdf
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.vision import VisionRecognizer
from tests.test_ai_agent import project, record, observation
from tests.test_regressions import register


def test_cable_and_terminal_counts_are_unknown_not_zero_or_one(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,'extract_dual_channel_spans',lambda page,spans,**kw:spans)
    monkeypatch.setattr(VisionRecognizer,'analyze_vector_regions_on_page',lambda *a:[])
    with pymupdf.open() as doc:
        p=doc.new_page(width=600,height=400)
        for x,y,t in [(60,70,'-79W3'),(60,90,'4G2.5'),(250,70,'-XT19'),(450,70,'-L9')]:p.insert_text((x,y),t)
        items,_=extract_from_schematic_pdf(doc,[PageInfo(1,'TERMINAL_DIAGRAM','terminals',width=600,height=400)],load_rules_config())
    bytag={i.tag:i for i in items}
    assert bytag['-79W3'].quantity is None and bytag['-79W3'].unit=='米'
    assert bytag['-79W3'].specification=='4G2.5'
    assert bytag['-XT19'].quantity is None
    assert '-L9' in bytag


def test_short_literal_models_do_not_depend_on_catalogue_family():
    assert extract_model_strings('VCF6')==['VCF6']
    assert extract_model_strings('ZEB8')==['ZEB8']


def test_semantic_name_and_cited_parameters_survive_without_a_model():
    item=BOMItem('old',1,'+PANEL-A9',tags=['+PANEL-A9'],model='未提取到型号(待核实)',source_type='UNRESOLVED')
    r=project([item]);obs=observation('-A9',None)
    obs.update(name='温度采集器',specification='DC24V',specification_span_ids=[0])
    data=record(observations=[obs]);data['spans'][0]['text']='DC24V'
    merge_observations(r,[data])
    assert item.name=='温度采集器' and item.specification=='DC24V'
    assert item.model=='未提取到型号(待核实)'
    assert item.observations[0]['specification']=='DC24V'


def test_distant_parameter_claim_is_preserved_as_observation_not_fact():
    r=project();obs=observation(model=None)
    obs.update(specification='DC48V',specification_span_ids=[0])
    data=record(observations=[obs]);data['spans'][0].update(text='DC48V',norm_bbox=[.8,.8,.9,.9])
    merge_observations(r,[data])
    assert not r.itemized_bom[0].specification
    assert r.itemized_bom[0].observations[0]['specification']=='DC48V'


def test_ai_cited_connection_names_do_not_pollute_a_display_rating():
    r=project()
    obs=observation(model=None)
    obs.update(specification='供电端',specification_span_ids=[0,1,2,3])
    data=record(observations=[obs])
    data['spans']=[dict(id=i,text=s,norm_bbox=obs['bbox_normalized'])
                   for i,s in enumerate(['V2+','V2-','FE','DC48V'])]
    merge_observations(r,[data])
    assert r.itemized_bom[0].specification=='DC48V'


def test_terminal_numbers_do_not_split_purchases_by_specification():
    r = project()
    obs = observation(model=None)
    obs.update(specification='端子 11、12、14', specification_span_ids=[0, 1, 2])
    data = record(observations=[obs])
    box = obs['bbox_normalized']
    data['spans'] = [dict(id=i, text=number, norm_bbox=box) for i, number in enumerate(['11','12','14'])]
    merge_observations(r, [data])
    assert not r.itemized_bom[0].specification
    assert r.itemized_bom[0].observations[0]['specification'] == '端子 11、12、14'


def test_ungrounded_model_never_becomes_original_drawing_fact():
    r=project();obs=observation(model='INVENTED-77')
    merge_observations(r,[record(observations=[obs])])
    assert r.itemized_bom[0].raw_model==''
    assert r.itemized_bom[0].observations[0]['model_raw']=='INVENTED-77'
    obs['name']='功能已识别';merge_observations(r,[record(observations=[obs])])
    assert r.itemized_bom[0].name=='功能已识别'
    assert r.itemized_bom[0].model!='INVENTED-77'


def test_contacts_alone_and_template_ranges_do_not_become_one_purchase():
    r=project();merge_observations(r,[record(observations=[observation('-KA7',None,'contact')])])
    assert r.itemized_bom[0].quantity is None
    r=project();obs=observation('-S9',None)
    obs.update(quantity_basis='repeated_template',quantity_candidates=[1,9],quantity_evidence='DV02-10')
    merge_observations(r,[record(observations=[obs])])
    assert r.itemized_bom[0].quantity is None
    assert r.itemized_bom[0].quantity_candidates==[1,9]


def test_native_range_review_preserves_manual_confirmation():
    r=project([BOMItem('a',1,'-S8'),BOMItem('b',2,'-S9',source_type='MANUAL_EDITED',quantity=9)])
    r.pages[0].extraction_diagnostics={'range_annotations':[{'text':'DV02-10','count':9}]}
    for i in r.itemized_bom:
        from bom_extractor.models import EvidenceLocation
        i.evidences=[EvidenceLocation(1,[0,0,1,1],[0,0,.1,.1])]
    flag_template_quantities(r.itemized_bom,r.review_items,r.pages)
    assert r.itemized_bom[0].quantity is None
    assert r.itemized_bom[1].quantity==9


def test_aggregation_does_not_mix_ratings_or_scope_or_hide_unknown_quantity():
    a=BOMItem('a',1,'-F7',model='SERIES',specification='10A',procurement_scope='cabinet')
    b=copy.deepcopy(a);b.id='b';b.tag='-F8';b.specification='16A'
    c=copy.deepcopy(a);c.id='c';c.procurement_scope='field'
    cable=BOMItem('d',4,'-W8',model='CABLE',quantity=None,entity_role='cable',category='auxiliary')
    other=copy.deepcopy(cable);other.id='e';other.tag='-W9'
    assert len(build_aggregated_from_itemized([a,b,c,cable,other]))==5


def test_procurement_groups_same_catalog_item_used_for_different_functions():
    a = BOMItem('panel-a', 1, '-P21', name='泵出口显示', procurement_name='面板仪表',
                model='CTRL-82B', raw_model='CTRL-82B', brand='示例厂商')
    b = copy.deepcopy(a)
    b.id, b.tag, b.name = 'panel-b', '-M22', '主机负载显示'
    spare = copy.deepcopy(a)
    spare.id, spare.tag, spare.name = 'spare', '-U23', '原厂成套备件'
    spare.source_type = 'MANUAL_EDITED'
    summary = build_aggregated_from_itemized([a, b, spare])
    assert [(r.name, r.quantity) for r in summary] == [('面板仪表', 2), ('原厂成套备件', 1)]
    assert summary[0].tags == ['-P21', '-M22']
    assert (a.name, b.name) == ('泵出口显示', '主机负载显示')


def test_unknown_count_survives_edit_export_and_reload(tmp_path):
    import backend.app as backend
    r=project([BOMItem('cable',1,'-W8',tags=['-W8'],quantity=None,entity_role='cable',unit='米')])
    r.aggregated_bom=build_aggregated_from_itemized(r.itemized_bom);pid=register(r)
    row=backend.store.results[pid].aggregated_bom[0]
    data=backend.update_bom_item(pid,backend.BOMItemUpdatePayload(id=row.id,tag=row.tag,quantity=None,procurement_scope='field',specification='5G6'))
    assert data['aggregated_bom'][0]['quantity'] is None
    restored=backend.ProjectStore();restored.initialize_defaults()
    assert restored.results[pid].itemized_bom[0].quantity is None
    wb=openpyxl.load_workbook(data['summary']['excel_path'],data_only=True)
    assert wb['物料清单']['E7'].value is None
    assert '5G6' in wb['物料清单']['C7'].value
    wb.close()


def test_selection_acceptance_preserves_original_model_and_human_decision():
    import backend.app as backend
    item=BOMItem('cpu',1,'+PANEL-A9',tags=['+PANEL-A9'],model='CPU-123',raw_model='CPU-123',selection_suggestions=[{'model':'CPU-123+MEM-9','reason':'可选存储卡','status':'PENDING'}])
    r=project([item]);pid=register(r)
    backend.selection_action(pid,'cpu',backend.SelectionActionPayload(model='CPU-123+MEM-9',action='ACCEPT'))
    assert item.raw_model=='CPU-123' and item.model=='CPU-123+MEM-9'
    merge_observations(r,[record(observations=[observation('-A9','NEW-123')])])
    assert item.model=='CPU-123+MEM-9'
    assert backend.agent_context(pid)['schema_version']==1


def test_model_only_edit_does_not_confirm_or_invent_an_unknown_quantity():
    import backend.app as backend
    item=BOMItem('x',1,'-W9',tags=['-W9'],quantity=None,quantity_status='unknown')
    r=project([item]);r.aggregated_bom=build_aggregated_from_itemized([item]);pid=register(r)
    backend.update_bom_item(pid,backend.BOMItemUpdatePayload(id=item.id,view_mode='itemized',tag=item.tag,model='CABLE-7'))
    assert item.quantity is None and item.quantity_status=='unknown'


def test_explicit_scope_exclusion_and_restore_are_reversible():
    import backend.app as backend
    item=BOMItem('x',1,'-A9',tags=['-A9'],model='MODEL-7')
    r=project([item]);r.aggregated_bom=build_aggregated_from_itemized([item]);pid=register(r)
    backend.update_bom_item(pid,backend.BOMItemUpdatePayload(id=item.id,view_mode='itemized',tag=item.tag,model=item.model,procurement_scope='excluded'))
    assert not r.itemized_bom and r.filtered_items[0].id=='x'
    backend.restore_filtered_item(pid,'x')
    assert r.itemized_bom[0].id=='x' and not r.filtered_items
    assert r.itemized_bom[0].procurement_scope=='unknown'


def test_untagged_material_can_be_accepted_without_inventing_quantity():
    import backend.app as backend
    r=project();obs=observation(None,None);obs.update(name='保护地汇流排')
    merge_observations(r,[record(observations=[obs])]);pid=register(r)
    assert not r.itemized_bom
    rev=next(x for x in r.review_items if x.action_type=='ADD_ITEM')
    backend.handle_review_action(pid,rev.id,backend.ReviewActionPayload(action='ACCEPT'))
    assert len(r.itemized_bom)==1 and r.itemized_bom[0].quantity is None
    assert r.itemized_bom[0].source_type=='MANUAL_EDITED'


def test_contact_uncertainty_does_not_override_a_coil_seen_on_another_page():
    r=project();contact=observation('-K9',None,'contact');contact['quantity_basis']='unknown'
    coil=observation('-K9',None,'coil')
    merge_observations(r,[record(observations=[contact]),record(2,[coil])])
    assert r.itemized_bom[0].quantity==1
    partial=project();merge_observations(partial,[record(observations=[contact])])
    assert partial.itemized_bom[0].quantity is None
    merge_observations(partial,[record(2,[coil])])
    assert partial.itemized_bom[0].quantity==1


def test_internal_contact_with_same_tag_is_not_a_parent_of_the_main_device():
    r=project();main=observation('-K9',None,'device');contact=observation('-K9',None,'contact')
    contact['parent_tag']='-K9'
    merge_observations(r,[record(observations=[main,contact])])
    assert len(r.itemized_bom)==1
    assert r.itemized_bom[0].parent_tag=='' and r.itemized_bom[0].quantity==1


def test_ai_field_scope_does_not_depend_on_a_known_tag_prefix():
    r=project();obs=observation('-UNFAMILIAR9',None,'field_load');obs['scope']='field'
    merge_observations(r,[record(observations=[obs])])
    assert not r.itemized_bom
    assert r.filtered_items[0].procurement_scope=='field'


def test_template_recheck_preserves_an_accepted_review_even_for_legacy_rows():
    from bom_extractor.models import EvidenceLocation, ReviewItem
    item=BOMItem('legacy',1,'+PANEL-S9',tags=['+PANEL-S9'],quantity=9)
    item.evidences=[EvidenceLocation(1,[0,0,1,1],[0,0,.1,.1])]
    r=project([item]);r.pages[0].extraction_diagnostics={'range_annotations':[{'text':'DV02-10','count':9}]}
    r.review_items=[ReviewItem('ok','INFO','CONFIRM',item.tag,'','','',1,status='ACCEPTED')]
    flag_template_quantities(r.itemized_bom,r.review_items,r.pages)
    assert item.quantity==9


def test_filtered_backup_export_keeps_parameters_and_unknown_lengths(tmp_path):
    from bom_extractor.exporter import export_to_excel
    r=project();r.filtered_items=[BOMItem('field',1,'-W9',quantity=None,unit='米',specification='5G6',procurement_scope='field')]
    path=tmp_path/'backup.xlsx';export_to_excel(r,path,diagnostic=True)
    wb=openpyxl.load_workbook(path,data_only=True);s=wb['已过滤物料备查']
    headers={c.value:c.column for c in s[1]}
    assert s.cell(2,headers['原图参数']).value=='5G6'
    assert s.cell(2,headers['物料单位']).value=='米'
    assert s['D2'].value is None
    wb.close()


@pytest.mark.parametrize('labels,expected',[
    (['6ES7 288-1ST30-0AA1','CPU ST30 DC/DC/DC (6ES7 288-1ST30-0AA1)'],1),
    (['ACS510','ACS510-01-157A-4'],1),
    (['ACS510-01-157A-4','ACS510-01-087A-4'],2),
    (['ABC-123','ABC-1234'],2),
    (['ABC-123','ABC-123+BASE-456'],2),
    (['ABC-123','ABC-123-OTHER'],2),
    (['VCF6 175A','VCF6 250A'],2),
    (['SERIES99 2P C10','SERIES99 2P C16'],2),
])
def test_description_and_series_labels_are_not_false_model_conflicts(labels,expected):
    from bom_extractor.ai_agent import canonical_models
    assert len(canonical_models(labels))==expected


def test_repeated_ai_merges_do_not_duplicate_parameter_values():
    r=project();obs=observation(model=None);obs['specification_span_ids']=[0,1]
    data=record(observations=[obs]);data['spans']=[{'id':i,'text':text,'norm_bbox':[.2,.2,.4,.4]} for i,text in enumerate(['24V','2A'])]
    merge_observations(r,[data]);merge_observations(r,[data])
    assert r.itemized_bom[0].specification=='24V；2A'


@pytest.mark.parametrize('complete',[True,False])
def test_only_a_complete_reconciliation_clears_an_old_machine_model_conflict(complete):
    from bom_extractor.models import ReviewItem
    item=BOMItem('known',1,'+PANEL-K1',tags=['+PANEL-K1'],model='NEW-123')
    r=project([item]);r.review_items=[ReviewItem('old','CONFLICT','AI_MODEL_CONFLICT',item.tag,'','','',1)]
    merge_observations(r,[record(),record(2)] if complete else [record()])
    assert any(x.id=='old' for x in r.review_items) is (not complete)
