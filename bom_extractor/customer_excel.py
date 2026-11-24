"""Compact employee-facing Excel views; detailed evidence stays in diagnostics."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import openpyxl
from openpyxl.styles import Alignment,Border,Font,PatternFill,Side
from openpyxl.worksheet.page import PageMargins

from .table_workflow import table_policy

HEADERS=['序号','物料名称','型号/规格','品牌','数量','单位','位号','来源图号/页码','备注']
WIDTHS=[7,19,42,13,10,8,25,18,36]
BLUE='203F58'
AMBER='FFF1D4'


def _text(cell,value):
    cell.value=value
    if isinstance(value,str):cell.data_type='s'


def _known(model):
    return bool(model and not model.startswith(('未提取','图纸未注','待定','待核','未知')))


def _page_label(item):
    pages=sorted({e.page for e in item.evidences} or {item.primary_page})
    return '、'.join(f'P{p}' for p in pages[:10])+(f'等{len(pages)}页' if len(pages)>10 else '')


def _unit_text(unit):
    unit=(unit or '').strip()
    return '' if unit in {'待核','待定','未知','待确认','未注明'} else unit


def _flags(item,table_only):
    flags=[]
    if not _known(item.model):flags.append(('型号待确认',3))
    if item.quantity is None or (not table_only and item.quantity_status in {'unknown','needs_confirmation'}):flags.append(('数量待确认',5))
    if not table_only:
        if item.procurement_scope=='unknown':flags.append(('范围待确认',None))
        if item.parent_tag or item.entity_role in {'assembly','contact'}:flags.append(('组件关系待核',7))
        if any(s.get('status')=='PENDING' for s in item.selection_suggestions):flags.append(('选型建议待确认',3))
    return flags


def _model_text(item):
    parts=[item.model] if _known(item.model) else []
    if item.specification and ''.join(item.specification.split()).upper() not in {''.join(p.split()).upper() for p in parts}:
        parts.append(item.specification)
    return '\n'.join(p.replace('+','\n+') if len(p)>38 and '+' in p else p for p in parts) or '未确定'


def _facts(items):
    if not items:return '未建立对应记录'
    models=list(dict.fromkeys(i.get('model') or '未确定' for i in items))
    tags=list(dict.fromkeys(i.get('tag') or i.get('name') or '无位号' for i in items))
    tag_text='、'.join(tags[:3])+(f'等{len(items)}项' if len(items)>3 else '')
    model_text='；'.join(models[:3])+(f'等{len(models)}种' if len(models)>3 else '')
    quantities=[i.get('quantity') for i in items];units={_unit_text(i.get('unit')) for i in items}
    if any(q is None for q in quantities):quantity='数量待确认'
    elif len(items)==1 or (len(units)==1 and '' not in units):
        quantity=f"数量 {sum(quantities):g} {next(iter(units))}".rstrip()
    else:
        quantity='各项数量：'+'、'.join(f"{i['quantity']:g} {_unit_text(i.get('unit'))}".rstrip() for i in items)
    supply='、'.join(dict.fromkeys(i['supply_note'] for i in items if i.get('supply_note')))
    return f'{tag_text}\n型号：{model_text}\n{quantity}'+(f'\n供货说明：{supply}' if supply else '')


def _pending_rows(result):
    rows=[];covered=set();seen=set()
    labels={'DIFFERENCE':'清单字段有差异','AMBIGUOUS':'对应关系待核','TABLE_ONLY':'仅部件表有记录',
            'SCHEMATIC_ONLY':'仅原理图有记录','UNLINKED_REFERENCE':'原表位号未对应',
            'TABLE_ACCESSORY':'同位号附件仅见原表'}
    def add(row):
        key=tuple(row)
        if key not in seen:seen.add(key);rows.append(row)
    for r in result.reconciliation:
        supply_pending='SUPPLY_SCOPE_REVIEW' in r.get('issues',[])
        if r.get('decision','PENDING')!='PENDING' or (r['status'] not in labels and not supply_pending):continue
        facts=r.get('schematic',[]);declared=r.get('declared',[])
        covered.update(r.get('schematic_ids',[]))
        tags=list(dict.fromkeys(i.get('tag') or i.get('name') or '无位号' for i in [*facts,*declared]))
        pages=sorted({e['page'] for i in [*facts,*declared] for e in i.get('evidences',[])})
        add(['、'.join(tags[:4])+(f'等{len(tags)}项' if len(tags)>4 else ''),labels.get(r['status'],'供货范围待核'),_facts(facts),_facts(declared),
             '\n'.join(r.get('reasons',[])),'、'.join(f'P{p}' for p in pages)])
    members={i.id:i for i in result.itemized_bom}
    for group in result.aggregated_bom:
        affected=[members[key] for key in group.member_ids if key in members and key not in covered]
        if len(affected)>1 and all(_flags(i,False)==[('范围待确认',None)] for i in affected):
            add([f'物料清单第 {group.seq_no} 项','采购范围待核',_model_text(group),'',
                 f'核实这 {len(affected)} 个位号的供货方；如有例外，可在位号明细中分别处理。',
                 _page_label(group)])
            covered.update(i.id for i in affected)
    flag_groups={}
    for item in result.itemized_bom:
        flags=_flags(item,False)
        if flags and item.id not in covered:
            key=(item.primary_page,item.name,_model_text(item),tuple(f[0] for f in flags),item.quantity_basis,item.parent_tag,item.location_code)
            flag_groups.setdefault(key,[]).append(item)
    for (_,_,model,flags,_,_,_),items in flag_groups.items():
        add(['、'.join(i.tag or i.name for i in items),'、'.join(flags),model,
             '','确认缺失字段、采购范围或组件关系。','、'.join(dict.fromkeys(_page_label(i) for i in items))])
    legacy_unbound={}
    ai_by_page={}
    missing_models={i.tag for i in result.itemized_bom if not _known(i.model)}
    for review in result.review_items:
        if review.status!='PENDING' or review.severity=='INFO' or review.issue_type=='AI_RECOGNIZED':continue
        if review.issue_type=='UNRESOLVED_MODEL' and review.tag in missing_models:
            continue
        if review.issue_type in {'AI_EVIDENCE_CHECK','AI_REGION_REVIEW','QUANTITY_UNCERTAIN'}:
            ai_by_page.setdefault((review.page,review.issue_type),[]).append(review)
            continue
        if review.issue_type=='UNBOUND_MODEL' and review.title.startswith('型号文字尚未关联元件'):
            legacy_unbound[review.page]=legacy_unbound.get(review.page,0)+1
            continue
        source='项目导入报告' if review.scope=='project' else f'P{review.page}'
        add([review.tag or '图纸范围',review.title,review.description,'',review.suggestion,source])
    for page,count in sorted(legacy_unbound.items()):
        add(['图页','检查是否漏列设备',f'{count} 处原图文字尚未关联设备','','在工作台展开本页证据核对。',f'P{page}'])
    for (page,issue),reviews in ai_by_page.items():
        if len(reviews)==1:
            r=reviews[0]
            add([r.tag or '图纸范围',r.title,r.description,'',r.suggestion,f'P{page}'])
            continue
        label={'AI_EVIDENCE_CHECK':'核对 AI 引用的原图证据','AI_REGION_REVIEW':'核实未定位的部件','QUANTITY_UNCERTAIN':'确认采购数量'}[issue]
        examples=list(dict.fromkeys((r.proposed_data or {}).get('name') or r.tag or r.description[:35] for r in reviews))
        add([f'第 {page} 页',label,'、'.join(examples[:5]),'',f'共 {len(reviews)} 条。工作台按位置保留明细，可分别处理。',f'P{page}'])
    return rows


def _base_sheet(wb,name,title,result,widths,headers,note):
    sheet=wb.create_sheet(name)
    sheet.sheet_view.showGridLines=False
    sheet.sheet_view.zoomScale=90
    _text(sheet.cell(2,1),title)
    sheet.cell(2,1).font=Font(name='Microsoft YaHei',size=15,bold=True,color=BLUE)
    _text(sheet.cell(3,1),'源文件：'+(result.source_filename or result.pdf_filename)+(f'    图号：{result.drawing_no}' if result.drawing_no else ''))
    _text(sheet.cell(4,1),note)
    for r in (3,4):sheet.cell(r,1).font=Font(name='Microsoft YaHei',size=10,color='526879')
    for index,(width,header) in enumerate(zip(widths,headers),1):
        sheet.column_dimensions[openpyxl.utils.get_column_letter(index)].width=width
        cell=sheet.cell(6,index);_text(cell,header)
        cell.font=Font(name='Microsoft YaHei',size=10,bold=True,color='FFFFFF')
        cell.fill=PatternFill('solid',fgColor=BLUE)
        cell.alignment=Alignment(horizontal='center',vertical='center')
        cell.border=Border(right=Side(style='thin',color='FFFFFF'))
    for row,height in [(1,8),(2,27),(3,20),(4,20),(5,7),(6,29)]:sheet.row_dimensions[row].height=height
    if result.summary.get('cad_import',{}).get('requires_sheet_review'):
        _text(sheet.cell(5,1),'范围提醒：CAD有效图框或图框外文字仍需核对，请查看系统中的导入提醒。')
        sheet.cell(5,1).font=Font(name='Microsoft YaHei',size=10,color='8B5B12')
        sheet.row_dimensions[5].height=20
    sheet.freeze_panes='C7'
    sheet.print_title_rows='6:6'
    sheet.sheet_properties.pageSetUpPr.fitToPage=True
    sheet.page_setup.orientation='landscape'
    sheet.page_setup.paperSize=sheet.PAPERSIZE_A4
    sheet.page_setup.fitToWidth=1;sheet.page_setup.fitToHeight=0
    sheet.page_margins=PageMargins(left=.25,right=.25,top=.35,bottom=.35,header=.15,footer=.15)
    sheet.oddFooter.center.text='第 &P / &N 页'
    return sheet


def _body_row(sheet,row,values,widths):
    lines=1
    for col,(value,width) in enumerate(zip(values,widths),1):
        cell=sheet.cell(row,col);_text(cell,value)
        cell.font=Font(name='Microsoft YaHei',size=10,color='243746')
        cell.fill=PatternFill('solid',fgColor='F2F6F9' if row%2 else 'FFFFFF')
        cell.alignment=Alignment(vertical='center',wrap_text=True,horizontal='left')
        cell.border=Border(bottom=Side(style='hair',color='E0E7EC'))
        if value is not None:
            # Estimate full-width CJK characters as two glyph units, preserving explicit lines.
            required=sum(max(1,math.ceil(sum(2 if ord(c)>127 else 1 for c in line)/max(4,width-2))) for line in str(value).split('\n'))
            lines=max(lines,required)
    sheet.row_dimensions[row].height=min(409,max(29,lines*15+10))


def export_customer_excel(result,output_path):
    from .independent import legacy_table_project
    table_only=table_policy(result)
    wb=openpyxl.Workbook();wb.remove(wb.active)
    mode='仅提取部件表，未经原理图核对' if table_only else '历史清单，未按当前独立流程核对' if legacy_table_project(result) else '原理图识别并核对，采购范围与选型以人工确认为准'
    note=f'处理方式：{mode}。数量空白表示待确认，0数量保留。'
    status=result.summary.get('table_extraction',{}).get('status')
    if table_only and status in {'absent','unreadable','partial'}:note+=' 部件表未读完整，请检查表页。'
    sheet=_base_sheet(wb,'物料清单',(result.project_name or Path(result.pdf_filename).stem)+' · 物料清单',result,WIDTHS,HEADERS,note)
    items=result.itemized_bom if result.extraction_mode=='PART_LIST_MODE' else result.aggregated_bom
    conflicts={key for r in result.reconciliation if r.get('decision','PENDING')=='PENDING' and r['status'] in {'DIFFERENCE','AMBIGUOUS','SCHEMATIC_ONLY'} for key in r.get('schematic_ids',[])}
    model_conflicts={r.tag for r in result.review_items if r.status=='PENDING'
                     and r.issue_type in {'PARAM_MISMATCH','AI_MODEL_CONFLICT'} and r.tag}
    supply_by_id={}
    for match in result.reconciliation:
        sources=match.get('declared',[])
        ids=match.get('schematic_ids',[])
        if match.get('status') in {'MATCH','DIFFERENCE'} and len(ids)==len(sources)==1 and sources[0].get('supply_note'):
            supply_by_id[ids[0]]=(sources[0].get('tag') or '',sources[0]['supply_note'])
    for row,item in enumerate(items,7):
        flags=_flags(item,table_only)
        if set(item.member_ids or [item.id]).intersection(conflicts):flags.append(('清单差异待核',7))
        notes=[item.remark] if item.remark else []
        members=item.member_ids or [item.id]
        for key in members:
            if key in supply_by_id:
                tag,supply=supply_by_id[key]
                evidence=(tag+' ' if len(members)>1 and tag else '')+supply
                if evidence not in notes and not (len(members)==1 and supply in (item.remark or '')):
                    notes.append('原表供货说明：'+evidence)
        if not table_only and item.needs_review and model_conflicts.intersection(item.tags or [item.tag]):
            notes.append('同位号有不同型号标注，订购前请查阅待确认事项')
            flags.append(('型号有冲突',3))
        if item.model_origin=='drawing_parts_list_confirmed':notes.append('人工采纳部件表型号')
        if item.is_quantity_adjustment:notes.append('人工采购数量调整')
        if item.quantity is None and item.quantity_candidates:notes.append('候选数量：'+' / '.join(map(str,item.quantity_candidates)))
        if item.parent_tag:notes.append('所属组件：'+item.parent_tag)
        _body_row(sheet,row,[item.seq_no,item.name,_model_text(item),item.brand,item.quantity,_unit_text(item.unit),
                  item.tag,_page_label(item),'；'.join(notes)],WIDTHS)
        sheet.cell(row,1).alignment=Alignment(horizontal='center',vertical='center')
        sheet.cell(row,5).number_format='0' if item.quantity is None or float(item.quantity).is_integer() else '0.########'
        sheet.cell(row,5).alignment=Alignment(horizontal='right',vertical='center',indent=1)
        sheet.cell(row,6).alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
        for col in {c for _,c in flags if c is not None}:
            sheet.cell(row,col).fill=PatternFill('solid',fgColor=AMBER)
            sheet.cell(row,col).font=Font(name='Microsoft YaHei',size=10,color='8B5B12')
    if not items:
        _text(sheet.cell(7,2),'未提取到可用物料，请核对图页。')
        sheet.cell(7,2).font=Font(name='Microsoft YaHei',size=10,color='8B5B12')
    last_column=openpyxl.utils.get_column_letter(len(HEADERS))
    sheet.auto_filter.ref=f'A6:{last_column}{max(6,6+len(items))}'
    sheet.print_area=f'A1:{last_column}{max(7,6+len(items))}'
    if not table_only:
        pending=_pending_rows(result)
        if pending:
            widths=[25,27,48,48,62,18]
            checks=_base_sheet(wb,'待确认事项','待确认事项',result,widths,
                ['位号','问题','原理图/当前清单','部件表','需要确认什么','来源图号/页码'],
                '仅列出尚未解决的问题；已处理记录及完整证据可在系统中查看。')
            for row,values in enumerate(pending,7):_body_row(checks,row,values,widths)
            checks.freeze_panes='B7'
            checks.auto_filter.ref=f'A6:F{6+len(pending)}'
            checks.print_area=f'A1:F{6+len(pending)}'
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.xlsx',delete=False) as f:temporary=Path(f.name)
    try:wb.save(temporary);os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True);wb.close()
    return str(path)
