"""Isolate embedded parts tables from schematic facts, pixels and AI inputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import uuid

import pymupdf

POLICY = "independent-schematic-v1"
MARKER = "electrical-bom-independent-v1"
ATTACHMENT = "independent_evidence_v1.json"
VIEW_NAME = "schematic_only.pdf"


def independent_policy(result):
    return result.summary.get("independent_analysis", {}).get("policy") == POLICY


def schematic_pages(result):
    policy = result.summary.get("independent_analysis", {})
    if policy.get("policy") == POLICY:
        return set(policy.get("schematic_pages", []))
    return {p.page_num for p in result.pages}


def legacy_table_project(result):
    from .table_workflow import table_policy
    return not independent_policy(result) and not table_policy(result) and result.extraction_mode == "PART_LIST_MODE"


def overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def table_region(page, spans, detected):
    columns = detected["columns"]
    from .cad_part_list import row_anchors
    anchors = row_anchors(spans,detected,page.rect.height)
    height = max((s.get("size", 6) for s in spans if abs(s["cy"] - detected["header_y"]) < 8), default=8)
    titles = [s for s in spans if any(t in s["text"].replace(" ", "") for t in ("元件清单", "元器件清单", "部件清单", "材料清单", "物料清单")) and s["cy"] < detected["header_bottom"]]
    top = min([detected["header_y"] - height, *[s["bbox"][1] - height * .5 for s in titles]])
    ordered=sorted(anchors,key=lambda s:s['cy'])
    row_half=(ordered[-1]['cy']-ordered[-2]['cy'])/2 if len(ordered)>1 else height*1.8
    bottom = max((s["bbox"][3] for s in anchors), default=page.rect.height*.88) + max(height,row_half)
    w, h = page.rect.width, page.rect.height
    return [max(0, min(c["left"] for c in columns.values()) / w - .008), max(0, top / h),
            min(1, max(c["right"] for c in columns.values()) / w + .008), min(1, bottom / h)]


def _schematic_outside(page, spans, region):
    from .notation import COMPONENT_TAG_RE
    for s in spans:
        if overlaps(s["norm_bbox"], region):
            continue
        value = s["text"].strip()
        if COMPONENT_TAG_RE.fullmatch(value) or re.fullmatch(r"(?:NFB|MCB|MCCB|MC|OCR|INV|CR|PB|PL)\d+(?:[.-]\d+)*[A-Z]?", value, re.I):
            return True
    count = 0
    w, h = page.rect.width, page.rect.height
    for drawing in page.get_drawings():
        r = pymupdf.Rect(drawing["rect"]) * page.rotation_matrix
        box = [r.x0/w,r.y0/h,r.x1/w,r.y1/h]
        if .04 < box[0] < box[2] < .94 and .04 < box[1] < box[3] < .87 and r.width < .2*w and r.height < .2*h and not overlaps(box,region):
            count += 1
            if count >= 8:
                return True
    return False


def build_isolation_plan(doc, infos):
    from .classifier import extract_visual_spans
    from .ocr_engine import DualChannelOCREngine
    from .cad_part_list import detect_parts_columns, table_spans
    from types import SimpleNamespace
    pages = []
    source_spans = []
    for info, page in zip(infos, doc):
        native = extract_visual_spans(page)
        spans = native if len(native) >= 12 else DualChannelOCREngine.extract_dual_channel_spans(page, native)
        source_spans.append(spans)
        physical=page.rect*page.derotation_matrix
        normalized=table_spans(page,spans)
        detected = detect_parts_columns(normalized,page)
        table = info.page_type == "PART_LIST" or detected is not None
        region = table_region(SimpleNamespace(rect=physical),normalized,detected) if detected else [0,0,1,1]
        if detected and page.rotation:
            r=pymupdf.Rect(region[0]*physical.width,region[1]*physical.height,region[2]*physical.width,region[3]*physical.height)*page.rotation_matrix
            region=[r.x0/page.rect.width,r.y0/page.rect.height,r.x1/page.rect.width,r.y1/page.rect.height]
        full = table and (not detected or not _schematic_outside(page,spans,region))
        regions = [[0,0,1,1]] if full else [region] if table else []
        kept = [s for s in spans if not any(overlaps(s["norm_bbox"],box) for box in regions)]
        pages.append({"page":info.page_num,"eligible":not full,"regions":regions,"table_detected":table,
                      "full_table_page":full,"masked_spans":len(spans)-len(kept),"kept_spans":len(kept),
                      "table_bounds":region if table else None,"columns":detected})
    return {"policy":POLICY,"pages":pages,"schematic_pages":[p["page"] for p in pages if p["eligible"]],
            "excluded_table_pages":[p["page"] for p in pages if p["full_table_page"]],
            "mixed_table_pages":[p["page"] for p in pages if p["table_detected"] and not p["full_table_page"]]}, source_spans


def prepare_independent_view(source_path, destination, infos):
    """Keep original page indexes while masking both table pixels and native text.

    AI receives rendered page images and the filtered evidence attachment only;
    no full PDF or unfiltered attachment is submitted to a model.
    """
    source_path, destination = Path(source_path), Path(destination)
    with pymupdf.open(source_path) as original:
        plan, spans = build_isolation_plan(original,infos)
        with pymupdf.open(source_path) as view:
            for name in list(view.embfile_names()):
                view.embfile_del(name)
            records = []
            for page, part, native in zip(view, plan["pages"], spans):
                for box in part["regions"]:
                    visual = pymupdf.Rect(box[0]*page.rect.width,box[1]*page.rect.height,box[2]*page.rect.width,box[3]*page.rect.height)
                    rectangle = visual * page.derotation_matrix
                    page.add_redact_annot(rectangle,fill=(1,1,1))
                if part["regions"]:
                    page.apply_redactions(images=2,graphics=1,text=0)
                    # An opaque topmost mask also covers CAD path-based glyphs.
                    for box in part["regions"]:
                        visual=pymupdf.Rect(box[0]*page.rect.width,box[1]*page.rect.height,box[2]*page.rect.width,box[3]*page.rect.height)
                        page.draw_rect(visual*page.derotation_matrix,color=None,fill=(1,1,1),overlay=True)
                clean=[copy.deepcopy(s) for s in native if not any(overlaps(s["norm_bbox"],box) for box in part["regions"])]
                records.append({"page":page.number+1,"spans":clean})
            view.set_metadata({"creator":"Electrical BOM independent analysis","keywords":MARKER})
            view.embfile_add(ATTACHMENT,json.dumps({"policy":POLICY,"pages":records},ensure_ascii=False).encode("utf-8"),filename=ATTACHMENT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name(destination.stem+'_'+uuid.uuid4().hex+'.pdf')
            try:
                view.save(temporary,deflate=True,garbage=3)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    plan["source_sha256"]=hashlib.sha256(source_path.read_bytes()).hexdigest()
    plan["view_sha256"]=hashlib.sha256(destination.read_bytes()).hexdigest()
    plan["view_filename"]=destination.name
    (destination.parent/"isolation_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    return plan


def independent_spans(page):
    doc=page.parent
    if MARKER not in (doc.metadata or {}).get("keywords",""):
        return None
    if not hasattr(doc,"_independent_evidence_cache"):
        data=json.loads(doc.embfile_get(ATTACHMENT))
        if data.get("policy")!=POLICY or len(data["pages"])!=len(doc):
            raise ValueError("独立识别证据文件损坏，请重新建立分析项目")
        doc._independent_evidence_cache=data
    return [dict(s) for s in doc._independent_evidence_cache["pages"][page.number]["spans"]]


def analysis_source(result, original, folder):
    """Never silently fall back from a missing isolated view to the original PDF."""
    from .table_workflow import table_policy
    if table_policy(result):
        raise ValueError('当前为仅提取部件表模式。请另建原理图核对项目后启动 AI，原表结果会保留。')
    if legacy_table_project(result):
        raise ValueError("此历史项目由图内清单生成，请先按新流程建立独立分析项目，原审核结果将保留。")
    if not independent_policy(result):
        return Path(original)
    root=Path(folder).resolve()
    target=(Path(result.summary.get('output_dir',folder))/VIEW_NAME).resolve()
    if not target.is_relative_to(root):
        raise ValueError('独立识别视图路径无效，请重新建立分析项目。')
    if not target.is_file():
        raise ValueError("独立识别视图缺失，请重新建立分析项目，未将原始清单送入 AI。")
    expected=result.summary['independent_analysis'].get('view_sha256')
    if not expected or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        raise ValueError("独立识别视图已变化，请重新建立分析项目。")
    return target
