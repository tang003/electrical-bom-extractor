from __future__ import annotations

import copy
import io
from pathlib import Path

import openpyxl
from fastapi.testclient import TestClient

import backend.app as backend
from bom_extractor.customer_excel import HEADERS
from bom_extractor.exporter import export_to_excel
from bom_extractor.models import BOMItem,ExtractionResult,ReviewItem,EvidenceLocation
from bom_extractor.rules import build_aggregated_from_itemized
from bom_extractor.table_workflow import TABLE_POLICY,TABLE_MODE


def item(identifier='one',tag='-QF1',**kwargs):
    values=dict(tags=[tag],name='断路器',model='ABC-123',brand='原图品牌',quantity=1,unit='台',procurement_scope='cabinet')
    values.update(kwargs)
    return BOMItem(identifier,1,tag,**values)


def result(items=None,table=False):
    items=[item()] if items is None else items
    r=ExtractionResult('proj_export','source.pdf','测试配电柜','E-501','PART_LIST_MODE' if table else 'SCHEMATIC_MODE',1,
        itemized_bom=items,aggregated_bom=[] if table else build_aggregated_from_itemized(items))
    if table:
        r.summary={'workflow_mode':TABLE_MODE,'table_extraction':{'policy':TABLE_POLICY,'status':'loaded'}}
        r.declared_parts=copy.deepcopy(items)
    return r


def test_default_main_columns_layout_and_literal_identifiers(tmp_path):
    r=result([item(tag='=02-K1',name='=literal')])
    before=copy.deepcopy(r.to_dict())
    path=tmp_path/'compact.xlsx';export_to_excel(r,path)
    assert r.to_dict()==before
    wb=openpyxl.load_workbook(path,data_only=False);s=wb['物料清单']
    assert wb.sheetnames==['物料清单']
    assert [c.value for c in s[6]]==HEADERS
    assert s['G7'].value=='=02-K1' and s['G7'].data_type=='s'
    assert s['B7'].value=='=literal' and s['B7'].data_type=='s'
    assert s['E7'].value==1 and s['E7'].number_format=='0'
    assert s.freeze_panes=='C7' and s.auto_filter.ref=='A6:I7'
    assert '确认状态' not in HEADERS and s.max_column==9
    assert s.print_title_rows=='$6:$6'
    assert s.page_setup.orientation=='landscape'
    assert s.page_setup.fitToWidth==1 and s.page_setup.fitToHeight==0
    assert not s.sheet_view.showGridLines and s['C7'].alignment.wrap_text


def test_diagnostic_export_keeps_each_drawing_annotation_with_its_position(tmp_path):
    card=item(tag='-AX1515N',name='4轴数字量轴卡',model='CK3W_AX1515N',evidences=[
        EvidenceLocation(6,[10,10,20,20],[.1,.1,.2,.2],'-AX1515N','main_circuit'),
        EvidenceLocation(6,[10,20,30,25],[.1,.2,.3,.25],'4轴数字量轴卡','device_description'),
        EvidenceLocation(6,[10,26,30,31],[.1,.26,.3,.31],'S轴命令端口','device_annotation'),
        EvidenceLocation(6,[10,32,30,37],[.1,.32,.3,.37],'CK3W_AX1515N','model_text'),
    ])
    path=tmp_path/'diagnostic.xlsx'
    export_to_excel(result([card]),path,diagnostic=True)
    wb=openpyxl.load_workbook(path)
    try:
        lines=list(wb['图纸标注信息'].values)
        assert [line[5] for line in lines[1:]]==['-AX1515N','4轴数字量轴卡','S轴命令端口','CK3W_AX1515N']
        assert lines[3][4]=='功能 / 接口说明' and lines[3][6]=='[0.1, 0.26, 0.3, 0.31]'
    finally:
        wb.close()


def test_supply_note_on_one_member_names_only_that_device_in_group(tmp_path):
    items=[item('1','-QF9'),item('2','-QF8')]
    r=result(items)
    r.independent_bom=copy.deepcopy(items)
    ref=copy.deepcopy(items[0]);ref.id='ref';ref.remark='客户自备'
    r.declared_parts=[ref]
    from bom_extractor.reconciliation import reconcile
    reconcile(r)
    path=tmp_path/'scope.xlsx';export_to_excel(r,path)
    wb=openpyxl.load_workbook(path)
    main=wb['物料清单']
    assert main['E7'].value==2 and '-QF9 客户自备' in main['I7'].value
    assert '-QF8' not in main['I7'].value
    assert '客户自备' in wb['待确认事项']['D7'].value
    wb.close()


def test_table_mode_keeps_blank_units_without_status_and_preserves_quantities(tmp_path):
    r=result([item('complete','-K1',needs_review=True),item('unknown','-K2',model='',specification='2A',quantity=None),
        item('zero','-K3',quantity=0,unit='待核'),item('length','-W1',quantity=2.75,unit='米'),
        item('blank-unit','-K4',unit='')],table=True)
    before=copy.deepcopy(r.to_dict())
    path=tmp_path/'table.xlsx';export_to_excel(r,path)
    w=openpyxl.load_workbook(path);s=w['物料清单']
    assert w.sheetnames==['物料清单']
    assert '未经原理图核对' in s['A4'].value
    assert s['I7'].value is None, 'No generated confirmation status in the remarks column'
    assert s['C8'].value=='2A' and s['E8'].value is None
    assert s['C8'].fill.fgColor.rgb.endswith('FFF1D4')
    assert s['E8'].fill.fgColor.rgb.endswith('FFF1D4')
    assert s['E9'].value==0 and s['F9'].value is None
    assert not s['F9'].fill.fgColor.rgb.endswith('FFF1D4')
    assert s['E10'].value==2.75 and s['E10'].data_type=='n'
    assert s['E10'].number_format=='0.########'
    assert s['F10'].value=='米' and s['F11'].value is None
    assert '单位待确认' not in ' '.join(str(c.value) for row in s for c in row)
    assert r.to_dict()==before, 'Export must not rewrite historical source values or manual work'


def test_confirmation_sheet_only_contains_unresolved_cases(tmp_path):
    r=result()
    r.review_items=[ReviewItem('done','WARNING','PARAM_MISMATCH','-K1','已处理事项','旧描述','旧建议',1,status='ACCEPTED'),
        ReviewItem('pending','WARNING','PARAM_MISMATCH','-K2','待确认型号','原理图型号不同','确认最终型号',1)]
    r.reconciliation=[{'status':'MATCH','decision':'PENDING','schematic_ids':[]},
        {'status':'DIFFERENCE','decision':'KEEP_SCHEMATIC','schematic_ids':[]}]
    path=tmp_path/'check.xlsx';export_to_excel(r,path)
    w=openpyxl.load_workbook(path)
    assert w.sheetnames==['物料清单','待确认事项']
    values=' '.join(str(c.value) for row in w['待确认事项'] for c in row if c.value is not None)
    assert '待确认型号' in values and '已处理事项' not in values
    assert w['待确认事项'].max_row==7
    r.review_items[1].status='IGNORED'
    export_to_excel(r,path)
    assert openpyxl.load_workbook(path).sheetnames==['物料清单']


def test_long_remarks_wrap_and_pending_comparison_has_both_sources(tmp_path):
    r=result([item(remark='工程师说明：'+'需要核对的具体内容。'*20)])
    r.reconciliation=[{'status':'DIFFERENCE','decision':'PENDING','schematic_ids':['one'],
        'schematic':[{'tag':'-QF1','model':'ABC-123','quantity':1,'unit':'台','evidences':[{'page':1}]}],
        'declared':[{'tag':'-QF1','model':'OTHER-999','quantity':2,'unit':'台','evidences':[{'page':3}]}],
        'reasons':['型号不同','数量不同']}]
    path=tmp_path/'long.xlsx';export_to_excel(r,path)
    w=openpyxl.load_workbook(path)
    assert w['物料清单']['I7'].value==r.itemized_bom[0].remark
    assert w['物料清单'].row_dimensions[7].height>60
    assert w['物料清单']['G7'].fill.fgColor.rgb.endswith('FFF1D4')
    s=w['待确认事项'];assert 'ABC-123' in s['C7'].value and 'OTHER-999' in s['D7'].value
    assert s['F7'].value=='P1、P3'


def test_empty_table_still_exports_one_clear_sheet(tmp_path):
    r=result([],table=True);r.summary['table_extraction']['status']='absent'
    r.review_items=[ReviewItem('none','WARNING','TABLE_EXTRACTION','','没找到表格','检查表页','重新选择',1)]
    path=tmp_path/'empty.xlsx';export_to_excel(r,path)
    w=openpyxl.load_workbook(path)
    assert w.sheetnames==['物料清单'] and '未读完整' in w['物料清单']['A4'].value
    assert '未提取到' in w['物料清单']['B7'].value


def test_diagnostic_endpoint_and_default_download_preserve_saved_state(tmp_path):
    r=result(table=True)
    folder=backend.OUTPUT_DIR/r.project_id;folder.mkdir(parents=True)
    backend.store.results[r.project_id]=r
    backend.store.sync_and_persist(r.project_id)
    before=(folder/'project_state.json').read_bytes()
    with TestClient(backend.app) as client:
        response=client.get(f'/api/projects/{r.project_id}/export/excel')
        assert response.status_code==200
        assert openpyxl.load_workbook(io.BytesIO(response.content)).sheetnames==['物料清单']
        diagnostic=client.get(f'/api/projects/{r.project_id}/export/excel/diagnostics')
        assert diagnostic.status_code==200
        names=openpyxl.load_workbook(io.BytesIO(diagnostic.content)).sheetnames
        assert {'部件表原始字段','识别任务与质量','图纸页面结构分析'}<=set(names)
        assert client.get('/api/projects/missing/export/excel/diagnostics').status_code==404
    assert (folder/'project_state.json').read_bytes()==before
