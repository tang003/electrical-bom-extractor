from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any
import pymupdf

from .models import PageInfo, PageType
from .notation import COMPONENT_TAG_RE, PDF_DEVICE_RE, device_code, is_connection_tag


PAGE_TYPE_LABELS = {
    PageType.COVER.value: "封面/首页",
    PageType.CONTENTS.value: "目录/图纸索引",
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
    from .independent import independent_spans
    isolated = independent_spans(page)
    if isolated is not None:
        return isolated
    from .cad_import import cad_page_spans
    cad_spans = cad_page_spans(page)
    if cad_spans is not None:
        return [dict(span) for span in cad_spans]
    mat = page.rotation_matrix
    w = float(page.rect.width)
    h = float(page.rect.height)
    spans: List[Dict[str, Any]] = []

    raw_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
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


def _right_value(spans, label, *, footer=False):
    labels = [s for s in spans if s["text"].strip().rstrip("：:") == label and (not footer or s["cy"] > .9*max(p["bbox"][3] for p in spans))]
    for anchor in labels:
        nearby = [s for s in spans if s["bbox"][0] > anchor["bbox"][2]+2 and abs(s["cy"]-anchor["cy"]) <= max(anchor["size"],s["size"])*.65]
        if label == "图号":
            nearby=[s for s in nearby if re.search(r"[A-Za-z]",s["text"]) and re.search(r"\d",s["text"])]
        if nearby:
            return min(nearby,key=lambda s:s["bbox"][0])["text"]
    return ""


def classify_pdf_pages(doc: pymupdf.Document) -> Tuple[str, str, str, List[PageInfo]]:
    pages_info = []
    project_name = drawing_no = ""
    part_list_pages = 0
    for idx, page in enumerate(doc):
        w,h = float(page.rect.width),float(page.rect.height)
        native = extract_visual_spans(page)
        spans = native
        if len(spans) < 12:
            from .ocr_engine import DualChannelOCREngine
            spans = DualChannelOCREngine.extract_dual_channel_spans(page,spans)
        drawing_count = len(page.get_cdrawings())
        text = " ".join(s["text"] for s in spans)
        compact = text.replace(" ", "")
        if not project_name:
            project_name = _right_value(spans,"项目名称")
            if not project_name:
                project_name = next((s["text"] for s in spans if any(k in s["text"] for k in ("控制系统","电液控制")) and 6<=len(s["text"])<50 and "知识产权" not in s["text"]), "")
        if not drawing_no:
            number = _right_value(spans,"图号")
            if number and not any(k in number for k in ("页数","执行标准")) and len(number)<45:
                drawing_no=number
            else:
                match=re.search(r"\b(C\d{5,6})\b",text)
                drawing_no=match.group(1) if match else ""
        footer = [s for s in spans if s.get("cy",0)>.90*h]
        titles=[s for s in footer if .56*w<s["cx"]<.72*w and 2<=len(s["text"])<40 and (any('\u4e00'<=c<='\u9fff' for c in s["text"]) or re.search(r"CPU|PLC|\d+AI|\d+AQ",s["text"],re.I)) and not any(k in s["text"] for k in ("执行标准","图号","原始项目"))]
        title = max(titles,key=lambda s:s["size"])["text"] if titles else ""
        location=_right_value(footer,"+") if footer else ""
        location=("+"+location.upper()) if re.fullmatch(r"[A-Z][A-Z0-9_.]*",location,re.I) else ""
        function=_right_value(footer,"=") if footer else ""
        function=("="+function.upper()) if re.fullmatch(r"[A-Z0-9][A-Z0-9_.]*",function,re.I) else ""
        sheet=_right_value(footer,"页数") if footer else ""
        page_type=PageType.OTHER.value
        cad_table = None
        from .cad_part_list import detect_parts_columns, table_spans
        cad_table = detect_parts_columns(table_spans(page,spans),page)
        if cad_table:
            page_type=PageType.PART_LIST.value
            part_list_pages+=1
            title=title or "CAD 元件清单"
        elif any(k in compact for k in ("部件列表","部件清单","元器件清单")) and any(k in compact for k in ("设备标识符","订货编号")):
            page_type=PageType.PART_LIST.value
            part_list_pages+=1
            title=title or "部件列表"
        elif "目录" in title or ("目录" in text and not any(COMPONENT_TAG_RE.fullmatch(s["text"]) for s in spans)):
            page_type=PageType.CONTENTS.value
            title=title or "目录"
        elif title in ("标题页","封面") or ("项目名称" in text and "客户" in text and "设计工程师" in text):
            page_type=PageType.COVER.value
            title=title or "封面"
        elif any(k in text for k in ("柜体外形","柜体底部开孔","正视图")):
            page_type=PageType.CABINET_LAYOUT.value
        elif any(k in text for k in ("安装版布置图","安装板布置图")):
            page_type=PageType.INSTALL_LAYOUT.value
        elif "场内接线图" in text or ("=+-X1" in text and "=+-X2" in text) or "端子图" in title:
            page_type=PageType.TERMINAL_DIAGRAM.value
        elif (any(k in title for k in ("CPU","PLC","模拟量","数字量","2AI","I/O")) or "PLC接线" in text or "PLC模块" in text
              or len({s['text'].upper() for s in spans if re.fullmatch(r'D[IQO]\d+',s['text'],re.I)})>=4
              or ('PROFINET' in text.upper() and re.search(r'\bCPU\b|远程\s*IO',text,re.I))):
            page_type=PageType.PLC_WIRING.value
        elif "拓扑图" in text or any(k in title for k in ("交换机","通讯","通信")):
            page_type=PageType.TOPOLOGY.value
        elif any(k in title for k in ("主电源","主电路","动力","供电","进线")) or "动力线原理图" in text:
            page_type=PageType.MAIN_CIRCUIT.value
        elif any(k in title for k in ("仪表","传感器","温度液位")):
            page_type=PageType.INSTRUMENT_WIRING.value
        elif any((COMPONENT_TAG_RE.fullmatch(s["text"]) or PDF_DEVICE_RE.fullmatch(s["text"])) and not is_connection_tag(s["text"]) for s in spans):
            page_type=PageType.CONTROL_CIRCUIT.value
            if not title and any(s["text"] in ("-QF0","-QF1","-QF3") for s in spans):
                page_type=PageType.MAIN_CIRCUIT.value
        vector = not native and drawing_count > 500 or ("定时器" in text and "REXL" not in text) or ("最低液位" in text and "LF60" not in text)
        pages_info.append(PageInfo(page_num=idx+1,page_type=page_type,page_type_label=PAGE_TYPE_LABELS[page_type],
            title=title or PAGE_TYPE_LABELS[page_type],sheet_no=sheet or str(idx+1),width=w,height=h,
            text_block_count=len(spans),drawing_count=drawing_count,has_vector_text_region=bool(vector),
            location_code=location,function_code=function,
            extraction_diagnostics={"native_characters":sum(len(sp["text"]) for sp in native),**({"cad_parts_table":cad_table} if cad_table else {})}))
    mode="PART_LIST_MODE" if part_list_pages else "SCHEMATIC_MODE"
    return mode,project_name or drawing_no or "电气控制柜项目",drawing_no,pages_info
