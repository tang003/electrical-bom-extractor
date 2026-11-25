from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .models import ExtractionResult, SourceType
from .procurement import item_readiness
from .table_workflow import table_policy


SOURCE_TYPE_CN = {
    SourceType.EXPLICIT.value: "原图明文",
    SourceType.UNRESOLVED.value: "型号待核实",
    SourceType.VISION_RECOGNIZED.value: "视觉识别待核实",
    SourceType.TOPOLOGY_INFERRED.value: "拓扑推导",
    SourceType.RULE_COMPLETED.value: "规则补全",
    SourceType.ESTIMATED_AUX.value: "工程估算",
    SourceType.MANUAL_EDITED.value: "人工已改",
}

DRAWING_ANNOTATION_ROLES = {
    'main_circuit': '位号', 'device_description': '器件说明',
    'device_annotation': '功能 / 接口说明', 'model_text': '原图型号',
    'model_candidate': '型号线索', 'specification': '技术参数',
}


def _style_worksheet(ws: openpyxl.worksheet.worksheet.Worksheet, header_row: int = 1) -> None:
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name="Microsoft YaHei", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Microsoft YaHei", size=10, color="1E293B")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    for col in range(1, ws.max_column + 1):
        cell = ws.cell(header_row, col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws.row_dimensions[header_row].height = 26

    for r in range(header_row + 1, ws.max_row + 1):
        ws.row_dimensions[r].height = 21
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(r, c)
            # Schematic identifiers such as '=02-K1' are literal text, never Excel formulas.
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.font = body_font
            cell.border = thin_border
            cell.alignment = Alignment(vertical="center", wrap_text=False)

    # Auto-fit column widths
    for col in range(1, ws.max_column + 1):
        max_len = 8
        for r in range(header_row, ws.max_row + 1):
            val = str(ws.cell(r, col).value or "")
            # Chinese characters count ~1.8 width
            char_len = sum(1.8 if ord(ch) > 127 else 1.0 for ch in val)
            if char_len > max_len:
                max_len = min(52, char_len)
        ws.column_dimensions[get_column_letter(col)].width = max_len + 4


def _export_diagnostics_to_excel(result: ExtractionResult, output_path: Path | str) -> str:
    """
    Export ExtractionResult into a multi-sheet standard Excel workbook:
    - Sheet 1: Primary standard BOM (matches customer/company format for direct use & benchmark)
    - Sheet 2: Secondary view (Aggregated procurement table or Itemized tag table)
    - Sheet 3: 待人工审核与校验报告 (Review & Anomaly Report with page & bbox coordinates)
    - Sheet 4: 已过滤非采购件备查 / 页面分类概览
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)
    table_only=table_policy(result)
    source_label=lambda it:('人工已改（未经原理图核对）' if it.source_type==SourceType.MANUAL_EDITED.value else '部件表原文（未经原理图核对）') if table_only else SOURCE_TYPE_CN.get(it.source_type,it.source_type)

    if result.extraction_mode == "PART_LIST_MODE":
        # Sheet 1: 部件清单 (Exact match to 双钱L9_部件清单_20261020.xlsx structure)
        ws1 = wb.create_sheet('部件表提取清单' if table_only else '部件清单')
        ws1.append(["序号", "名称/描述", "数量", "订货编号", "制造商", "备注", "放置位置", "PDF页码", "设备标识符(位号)", "数据来源", "坐标框(norm_bbox)"])
        for it in result.itemized_bom:
            ws1.append([
                it.seq_no,
                it.name or None,
                it.quantity,
                it.model or None,
                it.brand or None,
                it.remark or None,
                it.placement or None,
                it.pdf_page_label,
                it.tag,
                source_label(it),
                json.dumps(it.primary_bbox),
            ])
        _style_worksheet(ws1, 1)

        # Sheet 2: 合并采购汇总表
        ws2 = wb.create_sheet("合并采购汇总表")
        ws2.append(["序号", "品牌", "名称", "型号/订货编号", "单位", "汇总数量", "包含位号", "PDF页码", "数据来源"])
        for it in result.aggregated_bom:
            ws2.append([
                it.seq_no,
                it.brand,
                it.name,
                it.model,
                it.unit,
                it.quantity,
                it.tag,
                it.pdf_page_label,
                source_label(it),
            ])
        _style_worksheet(ws2, 1)
        if table_only:wb.remove(ws2)

    else:
        # SCHEMATIC_MODE: Sheet 1 matches C26368 bom20261102.xlsx structure
        ws1 = wb.create_sheet("BOM采购清单")
        ws1.append(["序号", "品牌", "名称", "型号", "单位", "数量", "对应位号", "来源页码", "数据来源", "坐标框(norm_bbox)", "工程说明"])
        for it in result.aggregated_bom:
            ws1.append([
                it.seq_no,
                it.brand or None,
                it.name or None,
                it.model or None,
                it.unit or "个",
                it.quantity,
                it.tag,
                it.pdf_page_label,
                source_label(it),
                json.dumps(it.primary_bbox),
                it.review_note or "",
            ])
        _style_worksheet(ws1, 1)

    # Original columns stay stable for integrations; decision fields are separate columns.
    def append_decisions(sheet, items):
        start=sheet.max_column+1
        labels=['图纸原始型号','原图参数','采购范围','数量状态','数量依据','候选数量','实体角色','所属组件','选型建议（待确认）','采购待办','型号采用来源','图内清单引用ID']
        for col,label in enumerate(labels,start):sheet.cell(1,col,label)
        for row,item in enumerate(items,2):
            values=[item.raw_model,item.specification,item.procurement_scope,item.quantity_status,item.quantity_basis,
                ', '.join(map(str,item.quantity_candidates)),item.entity_role,item.parent_tag,
                '；'.join(s.get('model','')+'：'+s.get('reason','') for s in item.selection_suggestions if s.get('status')=='PENDING'),
                '；'.join(item_readiness(item)['issues']),item.model_origin,item.parts_reference_id]
            for col,value in enumerate(values,start):sheet.cell(row,col,value)
        sheet.freeze_panes='D2';sheet.auto_filter.ref=sheet.dimensions
        _style_worksheet(sheet)
    append_decisions(ws1,result.itemized_bom if result.extraction_mode=='PART_LIST_MODE' else result.aggregated_bom)
    if result.extraction_mode=='PART_LIST_MODE' and not table_only:append_decisions(ws2,result.aggregated_bom)

    if result.extraction_mode=='SCHEMATIC_MODE':
        ws=wb.create_sheet('图纸标注信息')
        ws.append(['记录ID','位号','名称','图纸页','标注类型','原图文字','坐标框(norm_bbox)','采购范围'])
        for item in [*result.itemized_bom,*result.filtered_items]:
            seen=set()
            for ev in sorted(item.evidences,key=lambda e:(e.page,e.norm_bbox[1],e.norm_bbox[0])):
                if ev.role not in DRAWING_ANNOTATION_ROLES or not ev.raw_text.strip():
                    continue
                if ev.role=='main_circuit' and ev.raw_text.strip().upper().lstrip('-')!=item.tag.strip().upper().lstrip('-'):
                    continue
                key=(ev.page,ev.raw_text.strip(),tuple(ev.norm_bbox))
                if key in seen:continue
                seen.add(key)
                ws.append([item.id,item.tag,item.name,ev.page,DRAWING_ANNOTATION_ROLES[ev.role],
                           ev.raw_text,json.dumps(ev.norm_bbox),item.procurement_scope])
        _style_worksheet(ws);ws.freeze_panes='F2';ws.auto_filter.ref=ws.dimensions

    # Sheet 3: 待人工审核与校验报告
    ws_rev = wb.create_sheet("待审核与校验报告")
    ws_rev.append(["审核ID", "级别", "异常类型", "涉及位号", "问题标题", "详细证据说明", "人工处理建议", "图纸页码", "归一化坐标(norm_bbox)", "状态"])
    for rev in result.review_items:
        ws_rev.append([
            rev.id,
            rev.severity,
            rev.issue_type,
            rev.tag,
            rev.title,
            rev.description,
            rev.suggestion,
            f"第{rev.page}页",
            json.dumps(rev.norm_bbox),
            rev.status,
        ])
    _style_worksheet(ws_rev, 1)

    # Sheet 4: 已过滤非采购件备查 (if any) + 页面结构分析
    if result.filtered_items:
        ws_flt = wb.create_sheet("已过滤物料备查")
        ws_flt.append(["原序号", "设备标识符", "名称/描述", "原数量", "原订货编号", "制造商", "原备注", "过滤原因", "所在页码"])
        for fit in result.filtered_items:
            ws_flt.append([
                fit.seq_no,
                fit.tag,
                fit.name,
                fit.quantity,
                fit.model,
                fit.brand,
                fit.remark,
                fit.review_note,
                fit.pdf_page_label,
            ])
        unit_col=ws_flt.max_column+1
        ws_flt.cell(1,unit_col,'物料单位')
        for row,item in enumerate(result.filtered_items,2):ws_flt.cell(row,unit_col,item.unit)
        append_decisions(ws_flt,result.filtered_items)

    ws_pages = wb.create_sheet("图纸页面结构分析")
    ws_pages.append(["PDF页码", "图纸类型代码", "图纸类型名称", "图框标题", "文本块数", "矢量图元数", "含无文本层曲线字区域"])
    for p in result.pages:
        ws_pages.append([
            p.page_num,
            p.page_type,
            p.page_type_label,
            p.title,
            p.text_block_count,
            p.drawing_count,
            "是 (已调用视觉切片)" if p.has_vector_text_region else "否",
        ])
    _style_worksheet(ws_pages, 1)

    if table_only:
        ws=wb.create_sheet('提取范围与说明')
        ws.append(['项目','值'])
        ws.append(['处理方式','仅提取部件表；未经原理图核对'])
        ws.append(['来源文件',result.source_filename or result.pdf_filename])
        for key,value in result.summary['table_extraction'].items():
            ws.append([key,json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value])
        ws.append(['范围说明','不分析原理图，不自动增加附件或辅材；缺失数量留空，0数量原样保留；不同表页不自动相加。'])
        _style_worksheet(ws)
        ws=wb.create_sheet('部件表原始字段')
        ws.append(['记录ID','所在页','原序号','记号/位号原文','名称原文','型号/规格原文','数量原文','单位原文','品牌原文','备注原文'])
        for item in result.declared_parts:
            cells=next((o['cells'] for o in item.observations if 'cells' in o),{})
            ws.append([item.id,item.primary_page,cells.get('seq',item.seq_no),cells.get('tag',item.tag),cells.get('name',item.name),
                cells.get('model',item.raw_model),cells.get('quantity',item.quantity),cells.get('unit',item.unit),
                cells.get('brand',item.brand),cells.get('remark',item.remark)])
        _style_worksheet(ws);ws.freeze_panes='D2';ws.auto_filter.ref=ws.dimensions

    if result.summary.get('independent_analysis'):
        for label,items in [('原理图独立快照',result.independent_bom),('图内清单原文',result.declared_parts)]:
            ws=wb.create_sheet(label)
            ws.append(['记录ID','位号','名称','原文型号','型号','原文参数','数量','单位','页码','证据','型号来源'])
            for i in items:
                ws.append([i.id,i.tag,i.name,i.raw_model,i.model,i.specification,i.quantity,i.unit,i.pdf_page_label,
                           json.dumps([e.to_dict() for e in i.evidences],ensure_ascii=False),i.model_origin])
            _style_worksheet(ws);ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
        ws=wb.create_sheet('原理图与清单核对')
        ws.append(['核对ID','核对状态','原理图位号','原理图型号','原理图数量','原表位号','原表型号','原表数量','差异原因','人工处理','处理说明'])
        for r in result.reconciliation:
            fmt=lambda key,field:'；'.join(str(i.get(field,'')) for i in r[key])
            ws.append([r['id'],r['status'],fmt('schematic','tag'),fmt('schematic','model'),fmt('schematic','quantity'),
                       fmt('declared','tag'),fmt('declared','model'),fmt('declared','quantity'),'；'.join(r['reasons']),r['decision'],r.get('decision_note','')])
        _style_worksheet(ws);ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions

    ws_run=wb.create_sheet('识别任务与质量')
    ws_run.append(['项目','值'])
    for key,value in {'源图文件':result.source_filename or result.pdf_filename,'总页数':result.page_count,
        'AI任务':result.summary.get('ai_last_run',{}),'提取覆盖':result.summary.get('extraction_quality',{}),
        '采购审核':result.summary.get('procurement',{}),
        '口径':'候选覆盖与采购待办不是准确率；未知长度/数量为空；选型建议未改写原图型号。'}.items():
        ws_run.append([key,json.dumps(value,ensure_ascii=False) if isinstance(value,dict) else value])
    _style_worksheet(ws_run)

    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix=".xlsx", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        wb.save(str(temp_path))
        os.replace(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)
        wb.close()
    return str(out_path)


def export_to_excel(result: ExtractionResult, output_path: Path | str, *, diagnostic: bool=False) -> str:
    if diagnostic:return _export_diagnostics_to_excel(result,output_path)
    from .customer_excel import export_customer_excel
    return export_customer_excel(result,output_path)


def export_to_json(result: ExtractionResult, output_path: Path | str) -> str:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=out_path.parent, suffix=".json", encoding="utf-8", delete=False) as f:
        temp_path = Path(f.name)
        try:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return str(out_path)
