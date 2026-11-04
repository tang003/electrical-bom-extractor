from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .models import ExtractionResult, SourceType


SOURCE_TYPE_CN = {
    SourceType.EXPLICIT.value: "原图明文",
    SourceType.TOPOLOGY_INFERRED.value: "拓扑推导",
    SourceType.RULE_COMPLETED.value: "规则补全",
    SourceType.ESTIMATED_AUX.value: "工程估算",
    SourceType.MANUAL_EDITED.value: "人工已改",
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


def export_to_excel(result: ExtractionResult, output_path: Path | str) -> str:
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

    if result.extraction_mode == "PART_LIST_MODE":
        # Sheet 1: 部件清单 (Exact match to 双钱L9_部件清单_20261020.xlsx structure)
        ws1 = wb.create_sheet("部件清单")
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
                SOURCE_TYPE_CN.get(it.source_type, it.source_type),
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
                SOURCE_TYPE_CN.get(it.source_type, it.source_type),
            ])
        _style_worksheet(ws2, 1)

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
                SOURCE_TYPE_CN.get(it.source_type, it.source_type),
                json.dumps(it.primary_bbox),
                it.review_note or "",
            ])
        _style_worksheet(ws1, 1)

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
        _style_worksheet(ws_flt, 1)

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

    wb.save(str(out_path))
    return str(out_path)


def export_to_json(result: ExtractionResult, output_path: Path | str) -> str:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return str(out_path)
