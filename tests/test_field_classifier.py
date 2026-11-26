"""A targeted visual review must cite real text and never change a BOM before acceptance."""
import importlib
import json
import copy

import pymupdf
import pytest
from fastapi import HTTPException

from bom_extractor.field_classifier import (DeviceFields, build_context, decision_summary, preview_images, request_qwen,
                                            validate_fields, test_qwen_connection as check_connection)
from bom_extractor.models import BOMItem, EvidenceLocation, ExtractionResult, PageInfo, ReviewItem


def _context():
    return {'tag':'-F10','page':1,'anchor_box':[.10,.20,.15,.24],
            'focus_box':[.06,.16,.30,.39],'context_box':[.02,.10,.40,.46],
            'text_spans':[
                {'id':1,'text':'CK3M电源','norm_bbox':[.10,.25,.20,.28]},
                {'id':2,'text':'SE.iC65N-C10A/2P','norm_bbox':[.08,.30,.28,.33]},
                {'id':3,'text':'10 AMP','norm_bbox':[.11,.24,.18,.26]}],
            'neighbor_tags':[], 'existing_fields':{'name':'CK3M电源','model':'未提取到型号(待核实)','specification':''},
            'existing_evidence':[]}


def _decision():
    return {'tag':'-F10','assignments':[
        {'span_id':1,'role':'circuit_purpose','owner_tag':'-F10'},
        {'span_id':2,'role':'model','owner_tag':'-F10'},
        {'span_id':3,'role':'rating','owner_tag':'-F10'}],
        'model_raw':'iC65N-C10A/2P','model_span_ids':[2],
        'material_name':'断路器','name_source':'inferred','material_name_span_ids':[],
        'specification':'10 AMP','specification_span_ids':[3],
        'brand_explicit':None,'brand_span_ids':[],'visual_brand_candidate':None,
        'uncertain':False,'reason':'CK3M电源是回路用途，订货号在下一行。'}


def test_qwen_call_uses_two_crops_and_grounded_json():
    context = _context()
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        def fake_request(url, *, json: dict, headers, timeout):
            assert url.endswith('/chat/completions')
            assert json['model']=='qwen3-vl-plus'
            assert json['enable_thinking'] is False
            assert sum(part['type']=='image_url' for part in json['messages'][0]['content'])==2
            assert headers['Authorization']=='Bearer test-key'
            assert json['response_format']['json_schema']['schema']['$defs']['TextAssignment']['properties']['span_id']['enum']==[1,2,3]
            class Response:
                ok=True
                def json(self):
                    return {'choices':[{'finish_reason':'stop','message':{'content':__import__('json').dumps(_decision())}}],
                            'usage':{'prompt_tokens':1200,'completion_tokens':200}}
            return Response()
        result,usage=request_qwen(context,page,api_key='test-key',request=fake_request)
    assert result.model_raw=='iC65N-C10A/2P'
    assert usage['prompt_tokens']==1200
    assert decision_summary(result,context)['can_propose_update']
    wrong=_decision();wrong['model_span_ids']=[1]
    with pytest.raises(ValueError,match='原文证据'):
        validate_fields(wrong,context)
    damaged=copy.deepcopy(context)
    damaged['text_spans'][0]['text']='CK3M\ufffd\ufffd\ufffd'
    damaged['existing_evidence']=[{'text':'CK3M电源','box':context['text_spans'][0]['norm_bbox']}]
    assert decision_summary(result,damaged)['assignments'][0]['text']=='CK3M电源'


def test_qwen_corrects_invalid_evidence_once_and_accounts_for_both_requests():
    calls=[]
    wrong=_decision();wrong['assignments'][1]['role']='other'
    def fake_request(url, **kwargs):
        calls.append(copy.deepcopy(kwargs['json']))
        answer=wrong if len(calls)==1 else _decision()
        class Response:
            ok=True
            def json(self):
                return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}],
                        'usage':{'prompt_tokens':1200,'completion_tokens':200,'total_tokens':1400}}
        return Response()
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        result,usage=request_qwen(_context(),page,api_key='test-key',request=fake_request)
    assert result.model_raw=='iC65N-C10A/2P' and len(calls)==2
    assert len(calls[0]['messages'])==1 and len(calls[1]['messages'])==3
    assert 'assignments 必须为 model' in calls[1]['messages'][-1]['content']
    assert usage['prompt_tokens']==2400 and usage['total_tokens']==2800
    assert usage['validation_retries']==1


def test_qwen_stops_after_one_correction_if_evidence_is_still_invalid():
    calls=[];wrong=_decision();wrong['model_raw']='INVENTED-9'
    def fake_request(url, **kwargs):
        calls.append(kwargs['json'])
        class Response:
            ok=True
            def json(self):
                return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(wrong)}}]}
        return Response()
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        with pytest.raises(ValueError,match='自动校正后仍未通过原文校验'):
            request_qwen(_context(),page,api_key='test-key',request=fake_request)
    assert len(calls)==2


def test_qwen_does_not_retry_service_errors():
    calls=[]
    def fake_request(url, **kwargs):
        calls.append(kwargs['json'])
        class Response:ok=False;status_code=429
        return Response()
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        with pytest.raises(RuntimeError,match='HTTP 429'):
            request_qwen(_context(),page,api_key='test-key',request=fake_request)
    assert len(calls)==1


def test_field_citation_schema_excludes_reference_page_context_and_neighbor_spans():
    context=_context()
    context['text_spans'] += [
        {'id':90,'text':'OTHER-1','norm_bbox':[.08,.30,.28,.33],'page':2,'image_role':'reference'},
        {'id':91,'text':'OTHER-2','norm_bbox':[.32,.30,.38,.33],'page':1},
        {'id':92,'text':'OTHER-3','norm_bbox':[.07,.18,.09,.20],'page':1}]
    context['neighbor_tags']=[{'tag':'-F2','page':1,'box':[.06,.17,.10,.21]}]
    def fake_request(url, **kwargs):
        schema=kwargs['json']['response_format']['json_schema']['schema']
        assert schema['$defs']['TextAssignment']['properties']['span_id']['enum']==[1,2,3,90,91,92]
        for name in ('model_span_ids','specification_span_ids','brand_span_ids','material_name_span_ids'):
            assert schema['properties'][name]['items']['enum']==[1,2,3]
        for name in ('model_span_ids','specification_span_ids','kind_span_ids','relationship_span_ids'):
            assert schema['$defs']['DeviceComponent']['properties'][name]['items']['enum']==[1,2,3]
        class Response:
            ok=True
            def json(self):
                return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(_decision())}}]}
        return Response()
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        result,_=request_qwen(context,page,api_key='test-key',request=fake_request)
    assert result.model_raw=='iC65N-C10A/2P'


def test_complete_enclosure_crop_excludes_neighbor_text():
    with pymupdf.open() as doc:
        page=doc.new_page(width=800,height=600)
        for start,end in [((108,200),(292,200)),((108,420),(292,420)),
                          ((100,208),(100,412)),((300,208),(300,412))]:
            page.draw_line(start,end)
        page.insert_text((65,210),'-V1')
        page.insert_text((150,285),'MEAN WELL')
        page.insert_text((150,305),'NDR-120-24')
        page.insert_text((380,210),'-V2')
        page.insert_text((380,305),'Other device model')
        page.insert_text((10,570),'Unrelated page footer ' * 7)
        def item(tag,box,id):
            return BOMItem(id=id,seq_no=id,tag=tag,primary_page=1,evidences=[
                EvidenceLocation(1,[box[0]*800,box[1]*600,box[2]*800,box[3]*600],box,
                                 raw_text=tag,role='main_circuit',device_tag=tag)])
        target=item('-V1',[.075,.31,.145,.37],1)
        neighbor=item('-V2',[.46,.31,.53,.37],2)
        context=build_context(page,target,1,[target,neighbor])
        assert context['crop_mode']=='enclosure'
        assert len(preview_images(context,page))==1
        texts=[s['text'] for s in context['text_spans']]
        assert 'NDR-120-24' in texts and 'MEAN WELL' in texts
        assert '-V2' not in texts and 'Other device model' not in texts


def test_open_symbol_crop_reaches_model_below_symbol():
    with pymupdf.open() as doc:
        page=doc.new_page(width=800,height=600)
        for x,label in ((50,'0'),(150,'1'),(250,'2')):
            page.insert_text((x,10),label)
        page.insert_text((125,145),'-F1')
        page.insert_text((125,310),'40 AMP')
        page.insert_text((110,330),'SE.iC65N-C40A/3P')
        page.insert_text((215,430),'-F2')
        page.insert_text((10,570),'Unrelated page footer ' * 7)
        def item(tag,box,id):
            return BOMItem(id=id,seq_no=id,tag=tag,primary_page=1,evidences=[
                EvidenceLocation(1,[box[0]*800,box[1]*600,box[2]*800,box[3]*600],box,
                                 raw_text=tag,role='main_circuit',device_tag=tag)])
        target=item('-F1',[.14,.20,.22,.27],1)
        neighbor=item('-F2',[.26,.68,.34,.75],2)
        context=build_context(page,target,1,[target,neighbor])
        assert context['crop_mode']=='layout_cell'
        assert context['image_boxes'][0]['box'][3] > .55
        assert 'SE.iC65N-C40A/3P' in [s['text'] for s in context['text_spans']]


def test_stacked_open_symbols_keep_lower_labels_with_their_own_device(monkeypatch):
    from bom_extractor import field_classifier
    with pymupdf.open() as doc:
        page=doc.new_page(width=840,height=595)
        page.draw_line((0,563),(840,563))  # title-block boundary
        spans=[]
        def add(text,box):
            spans.append({'text':text,'norm_bbox':box})
        for x,label in ((.06,'0'),(.21,'1'),(.40,'2')):
            add(label,[x-.003,.006,x+.003,.015])
        add('SE.RGZE1S35M',[.190,.732,.229,.741])
        add('SE.RXG12BD',[.193,.742,.226,.751])
        add('-KA19',[.198,.753,.221,.766])
        add('24VDC',[.201,.803,.217,.812])
        add('schneider',[.195,.813,.224,.823])
        add('SE.RGZE1S35M',[.190,.840,.229,.849])
        add('SE.RXG12BD',[.193,.850,.226,.859])
        add('-KA20',[.198,.861,.221,.874])
        add('24VDC',[.201,.911,.217,.920])
        add('schneider',[.195,.921,.224,.930])
        add('项目名称',[.190,.949,.220,.958])
        monkeypatch.setattr(field_classifier,'page_evidence',lambda _:spans)
        def device(tag,id,box):
            return BOMItem(id=id,seq_no=id,tag=tag,primary_page=1,evidences=[
                EvidenceLocation(1,[box[0]*840,box[1]*595,box[2]*840,box[3]*595],box,
                                 raw_text=tag,role='contact_or_symbol',device_tag=tag)])
        ka19=device('-KA19',19,[.1863,.7383,.2325,.7810])
        ka20=device('-KA20',20,[.1863,.8461,.2325,.8888])
        for item,own_tag,other_tag in ((ka19,'-KA19','-KA20'),(ka20,'-KA20','-KA19')):
            context=build_context(page,item,1,[ka19,ka20])
            texts=[entry['text'] for entry in context['text_spans']]
            assert context['crop_mode']=='layout_cell'
            assert own_tag in texts and other_tag not in texts
            assert texts.count('24VDC')==1 and texts.count('schneider')==1
            assert '项目名称' not in texts
            assert context['focus_box'][3] > (.83 if item is ka19 else .93)
            assert len(preview_images(context,page))==1


def test_unbounded_last_row_uses_context_instead_of_claiming_complete_crop(monkeypatch):
    from bom_extractor import field_classifier
    with pymupdf.open() as doc:
        page=doc.new_page(width=840,height=595)
        spans=[{'text':'SE.RXG12BD','norm_bbox':[.193,.850,.226,.859]},
               {'text':'-KA20','norm_bbox':[.198,.861,.221,.874]},
               {'text':'schneider','norm_bbox':[.195,.921,.224,.930]}]
        monkeypatch.setattr(field_classifier,'page_evidence',lambda _:spans)
        anchor=[.1863,.8461,.2325,.8888]
        item=BOMItem(id=20,seq_no=20,tag='-KA20',primary_page=1,evidences=[
            EvidenceLocation(1,[anchor[0]*840,anchor[1]*595,anchor[2]*840,anchor[3]*595],
                             anchor,raw_text='-KA20',role='contact_or_symbol',device_tag='-KA20')])
        context=build_context(page,item,1,[item])
        assert context['crop_mode']=='legacy_context'
        assert len(context['image_boxes'])==2
        assert 'schneider' in [entry['text'] for entry in context['text_spans']]


def test_model_bearing_appearance_beats_first_contact_and_follows_page_column(monkeypatch):
    from bom_extractor import field_classifier
    with pymupdf.open() as doc:
        page=doc.new_page(width=840,height=595)
        spans=[{'text':text,'norm_bbox':box} for text,box in [
            ('SE.RGZE1S35M',[.190,.517,.229,.526]),('SE.RXG12BD',[.193,.527,.226,.536]),
            ('-KA9',[.198,.540,.221,.555]),('24VDC',[.201,.588,.218,.597]),
            ('-KA9',[.575,.540,.603,.555]),('/1.1',[.580,.576,.600,.586])]]
        monkeypatch.setattr(field_classifier,'page_evidence',lambda _:spans)
        def evidence(text,role,box):
            return EvidenceLocation(1,[box[0]*840,box[1]*595,box[2]*840,box[3]*595],box,
                                    raw_text=text,role=role,device_tag='-KA9')
        item=BOMItem(id='ka9',seq_no=1,tag='-KA9',primary_page=1,evidences=[
            evidence('-KA9','contact_or_symbol',[.5695,.5278,.6114,.5705]),
            evidence('-KA9','contact_or_symbol',[.1885,.5228,.2304,.5655]),
            evidence('SE.RXG12BD','model_text',[.193,.527,.226,.536]),
            evidence('SE.RGZE1S35M','model_text',[.190,.517,.229,.526])])
        info=PageInfo(1,'CONTROL_CIRCUIT','控制',extraction_diagnostics={
            'drawing_grid_columns':[{'number':n,'x':.066+n*.1} for n in range(10)]})
        refs=[{'source_page':1,'source_bbox':[.580,.576,.600,.586],
               'target_page':1,'grid_x':.166,'sheet':1,'column':1,'column_verified':True}]
        context=build_context(page,item,1,[item],doc,[info],refs)
        assert context['anchor_box']==item.evidences[1].norm_bbox
        assert [(image['role'],image['column']) for image in context['image_boxes']]==[
            ('primary',1),('reference',5)] or [
            (image['role'],image['column']) for image in context['image_boxes']]==[
            ('primary',1),('primary',1),('reference',5)]
        assert {'SE.RXG12BD','SE.RGZE1S35M'} <= {span['text'] for span in context['text_spans']}
        assert context['reference_links'][0]['column']==1
        assert context['selection_status']=='resolved'


def test_cross_page_reference_uses_main_page_and_keeps_source_page_evidence(monkeypatch):
    from bom_extractor import field_classifier
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        doc.new_page(width=600,height=400)
        spans_by_page={0:[{'text':'-KA9','norm_bbox':[.60,.4,.64,.44]},
                          {'text':'/2.1','norm_bbox':[.61,.46,.64,.48]}],
                       1:[{'text':'-KA9','norm_bbox':[.18,.4,.22,.44]},
                          {'text':'RXG12BD','norm_bbox':[.18,.35,.25,.38]}]}
        monkeypatch.setattr(field_classifier,'page_evidence',lambda page:spans_by_page[page.number])
        def evidence(page,role,box,text):
            return EvidenceLocation(page,[box[0]*600,box[1]*400,box[2]*600,box[3]*400],box,
                                    raw_text=text,role=role,device_tag='-KA9')
        item=BOMItem(id='ka9',seq_no=1,tag='-KA9',primary_page=1,evidences=[
            evidence(1,'contact_or_symbol',[.60,.4,.64,.44],'-KA9'),
            evidence(2,'coil',[.18,.4,.22,.44],'-KA9'),
            evidence(2,'model_text',[.18,.35,.25,.38],'RXG12BD')])
        infos=[PageInfo(n,'CONTROL_CIRCUIT','控制',extraction_diagnostics={
            'drawing_grid_columns':[{'number':i,'x':.065+i*.095} for i in range(10)]}) for n in (1,2)]
        refs=[{'source_page':1,'source_bbox':[.61,.46,.64,.48],
               'target_page':2,'grid_x':.16,'sheet':2,'column':1,'column_verified':True}]
        context=build_context(doc[0],item,1,[item],doc,infos,refs)
        assert context['page']==2
        assert [image['page'] for image in context['image_boxes']][-1]==1
        assert context['reference_links'][0]['target_page']==2
        assert {span['page'] for span in context['text_spans']}=={1,2}
        rendered=preview_images(context,doc[1],doc)
        assert rendered[0]['page']==2 and rendered[-1]['page']==1
        def fake_request(url, *, json: dict, headers, timeout):
            labels=[part['text'] for part in json['messages'][0]['content'] if part['type']=='text']
            assert any('PDF P2.1' in label for label in labels)
            assert any('PDF P1' in label for label in labels)
            assert sum(part['type']=='image_url' for part in json['messages'][0]['content'])==len(rendered)
            class Response:
                ok=True
                def json(self):
                    return {'choices':[{'finish_reason':'stop','message':{'content':__import__('json').dumps({
                        'tag':'-KA9','assignments':[],'model_raw':None,'model_span_ids':[],
                        'material_name':None,'name_source':'unknown','material_name_span_ids':[],
                        'specification':None,'specification_span_ids':[],
                        'brand_explicit':None,'brand_span_ids':[],'visual_brand_candidate':None,
                        'uncertain':True,'reason':'型号待核'})}}]}
            return Response()
        result,_=request_qwen(context,doc[1],doc,api_key='test-key',request=fake_request)
        assert result.tag=='-KA9'


def test_neighbor_owner_is_corrected_and_visual_logo_stays_candidate():
    context=_context()
    context['text_spans'].append({'id':4,'text':'低压电源','norm_bbox':[.32,.25,.37,.28]})
    context['neighbor_tags']=[{'tag':'-V2','box':[.30,.20,.40,.35]}]
    raw=_decision()
    raw['assignments'].append({'span_id':4,'role':'circuit_purpose','owner_tag':'-F10'})
    raw['visual_brand_candidate']='pilz'
    fields=validate_fields(raw,context)
    summary=decision_summary(fields,context)
    assert next(a for a in summary['assignments'] if a['span_id']==4)['owner_tag']=='-V2'
    assert summary['visual_brand_candidate']=='pilz'


def test_empty_printed_name_does_not_discard_grounded_model():
    raw=_decision()
    raw['material_name']=None
    raw['name_source']='printed'
    raw['visual_brand_candidate']=None
    fields=validate_fields(raw,_context())
    assert fields.name_source=='unknown'
    assert decision_summary(fields,_context())['model']=='iC65N-C10A/2P'


@pytest.mark.parametrize('name', ['0,5 mm²', '0.5 mm2', 'OUT07_1', '24VDC'])
def test_wire_or_rating_cannot_be_a_material_name(name):
    context=_context()
    context['tag']='-KA12'
    context['text_spans'].append({'id':4,'text':name,'norm_bbox':[.10,.35,.20,.37]})
    raw=_decision()
    raw.update(tag='-KA12',material_name=name,name_source='printed',material_name_span_ids=[4])
    raw['assignments'].append({'span_id':4,'role':'material_name','owner_tag':'-KA12'})
    for assignment in raw['assignments'][:-1]:
        assignment['owner_tag']='-KA12'
    result=validate_fields(raw,context)
    assert result.material_name is None
    assert result.name_source=='unknown'
    assert result.uncertain is True
    assert result.assignments[-1].role=='other'
    assert result.assignments[-1].owner_tag is None
    assert result.model_raw=='iC65N-C10A/2P'


def test_ka_context_supplies_a_cautious_device_hint():
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        box=[.2,.3,.3,.4]
        item=BOMItem(id='ka12',seq_no=1,tag='-KA12',primary_page=1,evidences=[
            EvidenceLocation(1,[120,120,180,160],box,raw_text='-KA12',role='coil',device_tag='-KA12')])
        context=build_context(page,item,1,[item])
    assert '继电器类' in context['device_class_hint']
    assert context['symbol_role_hint']=='coil'
    assert context['symbol_class_hint'] is None


def test_wire_gauge_cannot_be_a_model_either():
    context=_context()
    context['text_spans'].append({'id':4,'text':'0,5 mm²','norm_bbox':[.10,.35,.20,.37]})
    raw=_decision()
    raw['assignments'].append({'span_id':4,'role':'model','owner_tag':'-F10'})
    raw['model_raw']='0,5 mm²'
    raw['model_span_ids']=[4]
    result=validate_fields(raw,context)
    assert result.model_raw is None
    assert result.model_span_ids==[]
    assert result.assignments[-1].role=='other'
    assert decision_summary(result,context)['can_propose_update'] is False


def test_connection_check_sends_a_small_image_and_reports_key_errors():
    def success(url, *, json: dict, headers, timeout):
        assert json['model']=='qwen3-vl-plus'
        assert sum(part['type']=='image_url' for part in json['messages'][0]['content'])==1
        assert json['max_tokens']==80
        class Response:
            ok=True
            def json(self):
                return {'model':'qwen3-vl-plus','choices':[{'finish_reason':'stop',
                    'message':{'content':'{"color":"red"}'}}],
                    'usage':{'prompt_tokens':80,'completion_tokens':12}}
        return Response()
    result=check_connection(api_key='sample-key',request=success)
    assert result['ok'] is True
    assert result['usage']['completion_tokens']==12
    class Unauthorized:
        ok=False
        status_code=401
    with pytest.raises(RuntimeError,match='Key 无效'):
        check_connection(api_key='bad-key',request=lambda *args,**kwargs: Unauthorized())


def test_review_acceptance_reclassifies_purpose_without_changing_quantity(tmp_path,monkeypatch):
    server=importlib.import_module('backend.app')
    pdf=tmp_path/'drawing.pdf'
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        doc.save(pdf)
    context=_context()
    def evidence(text,role,box):
        return EvidenceLocation(1,[box[0]*600,box[1]*400,box[2]*600,box[3]*400],box,
                                raw_text=text,role=role,device_tag='-F10')
    item=BOMItem(id='f10',seq_no=1,tag='-F10',tags=['-F10'],name='CK3M电源',
                 model='未提取到型号(待核实)',quantity=1,primary_page=1,
                 evidences=[evidence('-F10','main_circuit',context['anchor_box']),
                            evidence('CK3M电源','model_text',context['text_spans'][0]['norm_bbox']),
                            evidence('CK3M电源','device_description',context['text_spans'][0]['norm_bbox']),
                            evidence('SE.iC65N-C10A/2P','model_text',context['text_spans'][1]['norm_bbox'])])
    conflict=ReviewItem('rev_conflict_-F10','CONFLICT','PARAM_MISMATCH','-F10',
                        '同位号不同型号','CK3M / iC65N','核对',1)
    project=ExtractionResult('field-test','drawing.pdf','test','', 'SCHEMATIC_MODE',1,
                             pages=[PageInfo(1,'MAIN_CIRCUIT','主回路',width=600,height=400)],
                             itemized_bom=[item],review_items=[conflict])
    monkeypatch.setattr(server.store,'results',{'field-test':project})
    monkeypatch.setattr(server.store,'pdf_paths',{'field-test':pdf})
    monkeypatch.setattr(server.store,'sync_and_persist',lambda project_id: project)
    monkeypatch.setattr(server.ai_jobs,'jobs',{})
    monkeypatch.setattr(server,'OUTPUT_DIR',tmp_path)
    monkeypatch.setattr(server,'analysis_source',lambda *args: pdf)
    monkeypatch.setattr(server,'build_context',lambda *args: context)
    monkeypatch.setattr(server,'analysis_fingerprint',lambda *args: 'f10fixture')
    monkeypatch.setattr(server,'request_qwen',lambda *args: (DeviceFields.model_validate(_decision()),{'prompt_tokens':1200}))
    response=server.classify_item_text('field-test','f10')
    assert response['review_status']=='PENDING'
    assert item.model=='未提取到型号(待核实)'
    assert len([r for r in project.review_items if r.issue_type=='AI_FIELD_CLASSIFICATION'])==1
    assert server.classify_item_text('field-test','f10')['cache_hit'] is True
    server.handle_review_action('field-test',response['review_id'],server.ReviewActionPayload(action='ACCEPT'))
    assert item.model=='iC65N-C10A/2P'
    assert item.name=='断路器'
    assert item.specification=='10 AMP'
    assert item.quantity==1
    assert conflict.status=='ACCEPTED'
    assert all(ev.role=='device_annotation' for ev in item.evidences if ev.raw_text=='CK3M电源')


def test_saved_ai_history_includes_legacy_result_and_review_status(tmp_path,monkeypatch):
    server=importlib.import_module('backend.app')
    fingerprint='a'*64
    item=BOMItem(id='ka12',seq_no=1,tag='-KA12',primary_page=1)
    review=ReviewItem('rev_ai_fields_'+fingerprint[:20],'WARNING','AI_FIELD_CLASSIFICATION',
                      '-KA12','旧判读','名称 0,5 mm²','核对',1)
    project=ExtractionResult('history-test','drawing.pdf','test','','SCHEMATIC_MODE',1,
                             itemized_bom=[item],review_items=[review])
    monkeypatch.setattr(server.store,'results',{'history-test':project})
    monkeypatch.setattr(server,'OUTPUT_DIR',tmp_path)
    cache=tmp_path/'history-test'/'ai_field_cache'
    cache.mkdir(parents=True)
    (cache/(fingerprint+'.json')).write_text(json.dumps({
        'summary':{'tag':'-KA12','page':1,'name':'0,5 mm²','name_source':'printed'},
        'model':'qwen3-vl-plus','usage':{'prompt_tokens':100}},ensure_ascii=False),encoding='utf-8')
    history=server.item_text_history('history-test','ka12')['records']
    assert len(history)==1
    assert history[0]['analysis']['name']=='0,5 mm²'
    assert history[0]['review_status']=='PENDING'
    assert history[0]['legacy'] is True


def test_stale_review_cannot_overwrite_a_later_edit(tmp_path,monkeypatch):
    # The review action checks the item snapshot before replacing any field.
    server=importlib.import_module('backend.app')
    item=BOMItem(id='f10',seq_no=1,tag='-F10',name='旧名称',model='待核')
    review=ReviewItem('ai-review','WARNING','AI_FIELD_CLASSIFICATION','-F10','title','desc','review',1,
                      action_type='UPDATE_ITEM',proposed_data={'target_id':'f10','model':'iC65N',
                      'expected_values':{'name':'旧名称','model':'待核','specification':'','brand':''}})
    result=ExtractionResult('stale','a.pdf','test','','SCHEMATIC_MODE',1,itemized_bom=[item],review_items=[review])
    monkeypatch.setattr(server.store,'results',{'stale':result})
    item.model='人工新型号'
    with pytest.raises(HTTPException) as exc:
        server.handle_review_action('stale','ai-review',server.ReviewActionPayload(action='ACCEPT'))
    assert exc.value.status_code==409
    assert item.model=='人工新型号'


def test_visual_brand_candidate_can_be_confirmed_without_filling_model(tmp_path,monkeypatch):
    server=importlib.import_module('backend.app')
    pdf=tmp_path/'drawing.pdf'
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        doc.save(pdf)
    context=_context()
    item=BOMItem(id='sr1',seq_no=1,tag='-SR1',tags=['-SR1'],name='电气元器件',
                 model='未提取到型号(待核实)',brand='待定',quantity=1,primary_page=1,
                 needs_review=True,evidences=[EvidenceLocation(1,[60,80,90,96],
                 context['anchor_box'],raw_text='-SR1',role='main_circuit',device_tag='-SR1')])
    legacy=ReviewItem('rev_ai_fields_brandfixture','WARNING','AI_FIELD_CLASSIFICATION',
                      '-SR1','旧判读','品牌候选','请核实',1,action_type='CONFIRM')
    project=ExtractionResult('brand-test','drawing.pdf','test','','SCHEMATIC_MODE',1,
                             pages=[PageInfo(1,'MAIN_CIRCUIT','主回路',width=600,height=400)],
                             itemized_bom=[item],review_items=[legacy])
    raw=_decision()
    raw.update(tag='-SR1',assignments=[],model_raw=None,model_span_ids=[],
               material_name=None,name_source='unknown',material_name_span_ids=[],
               specification=None,specification_span_ids=[],visual_brand_candidate='Pilz',
               uncertain=True,reason='图形商标可见，未找到型号文字。')
    monkeypatch.setattr(server.store,'results',{'brand-test':project})
    monkeypatch.setattr(server.store,'pdf_paths',{'brand-test':pdf})
    monkeypatch.setattr(server.store,'sync_and_persist',lambda project_id: project)
    monkeypatch.setattr(server.ai_jobs,'jobs',{})
    monkeypatch.setattr(server,'OUTPUT_DIR',tmp_path)
    monkeypatch.setattr(server,'analysis_source',lambda *args: pdf)
    monkeypatch.setattr(server,'build_context',lambda *args: {**context,'tag':'-SR1'})
    monkeypatch.setattr(server,'analysis_fingerprint',lambda *args: 'brandfixture')
    monkeypatch.setattr(server,'request_qwen',lambda *args: (DeviceFields.model_validate(raw),{}))
    response=server.classify_item_text('brand-test','sr1')
    assert response['analysis']['can_propose_brand'] is True
    assert response['analysis']['can_propose_update'] is False
    assert legacy.action_type=='UPDATE_ITEM'
    assert legacy.proposed_data['brand']=='Pilz'
    assert 'model' not in legacy.proposed_data
    assert item.brand=='待定'
    server.handle_review_action('brand-test',response['review_id'],server.ReviewActionPayload(action='ACCEPT'))
    assert item.brand=='Pilz'
    assert item.model=='未提取到型号(待核实)'
    assert item.quantity==1
    assert item.needs_review is True
    assert '型号与其他采购信息仍待核实' in item.review_note
    server.handle_review_action('brand-test',response['review_id'],server.ReviewActionPayload(action='ACCEPT'))
    assert item.needs_review is True
