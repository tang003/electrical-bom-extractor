from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any
import pymupdf

from .models import PageInfo, PageType


PAGE_TYPE_LABELS = {
    PageType.COVER.value: "封面/首页",
    PageType.CABINET_LAYOUT.value: "柜体外形/开孔图",
    PageType.INSTALL_LAYOUT.value: "安装板布置图",
    PageType.MAIN_CIRCUIT.value: "主电路/动力原理图",
    PageType.CONTROL_CIRCUIT.value: "控制回路原理图",
    PageType.PLC_WIRING.value: "PLC接线原理图",
    PageType.INSTRUMENT_WIRING.value: "仪表/传感器/阀门接线图",
    PageType.TOPOLOGY.value: "网络拓扑图",
    PageType.TERMINAL_DIAGRAM.value: "端子排/场内接线图",
    PageType.PART_LIST.value: "部件列表/元器件清单",
    PageType.OTHER.value: "其他图纸",
}


def extract_visual_spans(page: pymupdf.Page) -> List[Dict[str, Any]]:
    """
    Extract all text spans from a PDF page and transform their bounding boxes
    using `page.rotation_matrix` so coordinates are always in visual top-left (0,0)
    to bottom-right (width, height) space regardless of CAD rotation (0/90/180/270).
    """
    mat = page.rotation_matrix
    w = float(page.rect.width)
    h = float(page.rect.height)
    spans: List[Dict[str, Any]] = []

    raw_dict = page.get_text("dict")
    for block in raw_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                rect = pymupdf.Rect(span["bbox"]) * mat
                x0 = max(0.0, min(w, float(rect.x0)))
                y0 = max(0.0, min(h, float(rect.y0)))
                x1 = max(0.0, min(w, float(rect.x1)))
                y1 = max(0.0, min(h, float(rect.y1)))
                spans.append({
                    "text": text,
                    "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    "norm_bbox": [
                        round(x0 / w, 4) if w else 0.0,
                        round(y0 / h, 4) if h else 0.0,
                        round(x1 / w, 4) if w else 0.0,
                        round(y1 / h, 4) if h else 0.0,
                    ],
                    "cx": round((x0 + x1) / 2.0, 2),
                    "cy": round((y0 + y1) / 2.0, 2),
                    "size": round(float(span.get("size", 0.0)), 2),
                })
    return spans


def classify_pdf_pages(doc: pymupdf.Document) -> Tuple[str, str, str, List[PageInfo]]:
    """
    Analyze all pages of the PDF document.
    Returns (extraction_mode, project_name, drawing_no, pages_info).
    """
    pages_info: List[PageInfo] = []
    part_list_pages = 0
    project_name = ""
    drawing_no = ""

    for idx in range(len(doc)):
        page = doc[idx]
        page_num = idx + 1
        w = float(page.rect.width)
        h = float(page.rect.height)
        spans = extract_visual_spans(page)
        drawings = page.get_drawings()
        drawing_count = len(drawings)
        full_text = " ".join(s["text"] for s in spans)

        # Extract project metadata if available
        if not project_name:
            for s in spans:
                if "控制系统" in s["text"] or "电液控制" in s["text"]:
                    if len(s["text"]) >= 6 and "知识产权" not in s["text"]:
                        project_name = s["text"]
                        break
        if not drawing_no:
            m = re.search(r"\b(C\d{5,6})\b", full_text)
            if m:
                drawing_no = m.group(1)

        # Determine page type and title
        page_type = PageType.OTHER.value
        title = f"第 {page_num} 页"
        sheet_no = str(page_num)
        has_vector_text_region = False

        # Check EPLAN Part List table header
        if "部件列表" in full_text and ("设备标识符" in full_text or "订货编号" in full_text):
            page_type = PageType.PART_LIST.value
            part_list_pages += 1
            for s in spans:
                if s["text"].startswith("部件列表 :"):
                    title = s["text"]
                    break
            else:
                title = "部件列表"
        elif "封面" in full_text or ("Project Description" in full_text and "Drawing NO." in full_text):
            page_type = PageType.COVER.value
            title = "封面"
        elif len(spans) == 0 and drawing_count > 500:
            # Pure CAD vector drawing with 0 text layer (e.g., C26368 Page 20 cabinet layout)
            page_type = PageType.CABINET_LAYOUT.value
            title = "电柜结构尺寸图 (纯矢量无文本层)"
            has_vector_text_region = True
        elif "柜体外形" in full_text or "柜体底部开孔" in full_text or "正视图" in full_text:
            page_type = PageType.CABINET_LAYOUT.value
            title = "柜体外形/开孔图"
        elif "安装版布置图" in full_text or "安装板布置图" in full_text:
            page_type = PageType.INSTALL_LAYOUT.value
            title = "安装板布置图"
        elif "拓扑图" in full_text:
            page_type = PageType.TOPOLOGY.value
            title = "网络拓扑图"
        elif "场内接线图" in full_text or ("=+-X1" in full_text and "=+-X2" in full_text):
            page_type = PageType.TERMINAL_DIAGRAM.value
            title = "端子排/场内接线图"
        elif "PLC接线" in full_text or "PLC模块" in full_text:
            page_type = PageType.PLC_WIRING.value
            for s in spans:
                if "PLC" in s["text"] and len(s["text"]) <= 15:
                    title = s["text"]
                    break
        elif "动力线原理图" in full_text or any(s["text"] in ("-QF0", "-QF1", "-QF3", "-ES680-1", "-ES680-2") for s in spans):
            page_type = PageType.MAIN_CIRCUIT.value
            title = "主电路/动力原理图"
        elif any(k in full_text for k in ("NHR-1100C", "温度液位", "电磁阀接线")):
            page_type = PageType.INSTRUMENT_WIRING.value
            title = "仪表/电磁阀接线图"
        else:
            page_type = PageType.CONTROL_CIRCUIT.value
            title = "控制回路原理图"

        # Detect pages with embedded CAD vector-rendered text (like C26368 Page 5 REXL2TMP7 or Page 9 LF60G)
        if ("定时器" in full_text and "-KT1" in full_text and "REXL" not in full_text) or (
            "最低液位" in full_text and "14脚" in full_text and "LF60" not in full_text
        ):
            has_vector_text_region = True

        pages_info.append(
            PageInfo(
                page_num=page_num,
                page_type=page_type,
                page_type_label=PAGE_TYPE_LABELS.get(page_type, "其他图纸"),
                title=title,
                sheet_no=sheet_no,
                width=round(w, 1),
                height=round(h, 1),
                text_block_count=len(spans),
                drawing_count=drawing_count,
                has_vector_text_region=has_vector_text_region,
            )
        )

    extraction_mode = "PART_LIST_MODE" if part_list_pages > 0 else "SCHEMATIC_MODE"
    if not project_name:
        project_name = drawing_no or "电气控制柜项目"
    return extraction_mode, project_name, drawing_no, pages_info
