from __future__ import annotations
import copy
import json
import shutil
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List

import pymupdf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bom_extractor.models import ExtractionResult, BOMItem, EvidenceLocation, SourceType
from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.rules import build_aggregated_from_itemized, load_rules_config
from bom_extractor.exporter import export_to_excel, export_to_json
from bom_extractor.evaluator import evaluate_extraction
from bom_extractor.vision import VisionRecognizer


ROOT_DIR = Path(__file__).resolve().parent.parent
REF_DIR = ROOT_DIR / "参考文件"
OUTPUT_DIR = ROOT_DIR / "output"
FRONTEND_DIR = ROOT_DIR / "frontend"
RULES_PATH = ROOT_DIR / "bom_extractor" / "rules_config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="工业电气原理图 BOM 自动提取与图文审核系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


class ProjectStore:
    def __init__(self) -> None:
        self.results: Dict[str, ExtractionResult] = {}
        self.pdf_paths: Dict[str, Path] = {}
        self.gt_paths: Dict[str, Path] = {}

    def initialize_defaults(self) -> None:
        defaults = [
            {
                "proj_id": "C26368",
                "pdf": REF_DIR / "C26368.pdf",
                "gt": REF_DIR / "C26368 bom20261102.xlsx",
            },
            {
                "proj_id": "ShuangQian_L9",
                "pdf": REF_DIR / "双钱L9_20261020.pdf",
                "gt": REF_DIR / "双钱L9_部件清单_20261020.xlsx",
            },
        ]
        for item in defaults:
            if item["pdf"].exists():
                proj_id = item["proj_id"]
                self.pdf_paths[proj_id] = item["pdf"]
                if item["gt"].exists():
                    self.gt_paths[proj_id] = item["gt"]
                res = run_extraction_pipeline(
                    pdf_path=item["pdf"],
                    output_dir=OUTPUT_DIR / proj_id,
                    project_id=proj_id,
                )
                self.results[proj_id] = res

    def sync_and_persist(self, proj_id: str) -> ExtractionResult:
        res = self.results[proj_id]
        if res.extraction_mode == "PART_LIST_MODE":
            res.aggregated_bom = build_aggregated_from_itemized(res.itemized_bom)
        else:
            # In schematic mode, itemized_bom mirrors aggregated_bom edits
            res.itemized_bom = copy.deepcopy(res.aggregated_bom)

        res.summary["itemized_count"] = len(res.itemized_bom)
        res.summary["aggregated_count"] = len(res.aggregated_bom)
        res.summary["filtered_count"] = len(res.filtered_items)
        res.summary["review_count"] = len(res.review_items)
        res.summary["pending_review_count"] = sum(1 for r in res.review_items if r.status == "PENDING")

        proj_out = OUTPUT_DIR / proj_id
        proj_out.mkdir(parents=True, exist_ok=True)
        stem = self.pdf_paths[proj_id].stem if proj_id in self.pdf_paths else proj_id
        excel_path = export_to_excel(res, proj_out / f"{stem}_标准BOM清单.xlsx")
        json_path = export_to_json(res, proj_out / f"{stem}_提取结果.json")
        res.summary["excel_path"] = excel_path
        res.summary["json_path"] = json_path
        return res


store = ProjectStore()


@app.on_event("startup")
def on_startup() -> None:
    store.initialize_defaults()


def _serialize_project_response(proj_id: str) -> Dict[str, Any]:
    if proj_id not in store.results:
        raise HTTPException(status_code=404, detail=f"Project {proj_id} not found")
    res = store.results[proj_id]
    data = res.to_dict()
    if proj_id in store.gt_paths and store.gt_paths[proj_id].exists():
        data["evaluation"] = evaluate_extraction(res, store.gt_paths[proj_id])
    else:
        data["evaluation"] = None
    return data


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return "<h1>Frontend index.html not found</h1>"
    return index_file.read_text(encoding="utf-8")


@app.get("/api/projects")
def list_projects() -> List[Dict[str, Any]]:
    if not store.results:
        store.initialize_defaults()
    items = []
    for pid, res in store.results.items():
        eval_info = None
        if pid in store.gt_paths and store.gt_paths[pid].exists():
            eval_info = evaluate_extraction(res, store.gt_paths[pid])["metrics"]
        items.append({
            "project_id": pid,
            "pdf_filename": res.pdf_filename,
            "project_name": res.project_name,
            "drawing_no": res.drawing_no,
            "extraction_mode": res.extraction_mode,
            "extraction_mode_label": res.summary.get("extraction_mode_label", ""),
            "page_count": res.page_count,
            "itemized_count": len(res.itemized_bom),
            "aggregated_count": len(res.aggregated_bom),
            "review_count": len(res.review_items),
            "pending_review_count": sum(1 for r in res.review_items if r.status == "PENDING"),
            "evaluation_metrics": eval_info,
        })
    return items


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> Dict[str, Any]:
    if not store.results:
        store.initialize_defaults()
    return _serialize_project_response(project_id)


@app.post("/api/projects/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 格式的电气图纸文件")

    safe_stem = Path(file.filename).stem.replace(" ", "_")
    proj_id = f"proj_{safe_stem}"
    upload_dir = OUTPUT_DIR / proj_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = upload_dir / file.filename

    with open(saved_pdf, "wb") as f:
        shutil.copyfileobj(file.file, f)

    store.pdf_paths[proj_id] = saved_pdf
    # Check if uploaded file matches one of the reference ground-truth files
    if "C26368" in file.filename and (REF_DIR / "C26368 bom20261102.xlsx").exists():
        store.gt_paths[proj_id] = REF_DIR / "C26368 bom20261102.xlsx"
    elif "L9" in file.filename and (REF_DIR / "双钱L9_部件清单_20261020.xlsx").exists():
        store.gt_paths[proj_id] = REF_DIR / "双钱L9_部件清单_20261020.xlsx"

    res = run_extraction_pipeline(
        pdf_path=saved_pdf,
        output_dir=upload_dir,
        project_id=proj_id,
    )
    store.results[proj_id] = res
    return _serialize_project_response(proj_id)


@app.post("/api/projects/{project_id}/reextract")
def reextract_project(project_id: str) -> Dict[str, Any]:
    if project_id not in store.pdf_paths:
        raise HTTPException(status_code=404, detail="PDF source path not found")
    pdf_path = store.pdf_paths[project_id]
    res = run_extraction_pipeline(
        pdf_path=pdf_path,
        output_dir=OUTPUT_DIR / project_id,
        project_id=project_id,
    )
    store.results[project_id] = res
    return _serialize_project_response(project_id)


@app.get("/api/projects/{project_id}/pages/{page_num}/image")
def get_page_image(project_id: str, page_num: int, dpi: int = 150) -> FileResponse:
    if project_id not in store.pdf_paths:
        raise HTTPException(status_code=404, detail="Project PDF not found")
    pdf_path = store.pdf_paths[project_id]
    pages_cache_dir = OUTPUT_DIR / project_id / "pages_cache"
    pages_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_png = pages_cache_dir / f"page_{page_num:03d}_{dpi}.png"

    if not cached_png.exists():
        doc = pymupdf.open(str(pdf_path))
        try:
            if page_num < 1 or page_num > len(doc):
                raise HTTPException(status_code=400, detail=f"Invalid page number {page_num}")
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(cached_png))
        finally:
            doc.close()

    return FileResponse(str(cached_png), media_type="image/png")


class BOMItemUpdatePayload(BaseModel):
    view_mode: str = "primary"  # "itemized" | "aggregated" | "primary"
    id: str
    seq_no: Optional[int] = None
    tag: str = ""
    brand: str = ""
    name: str = ""
    model: str = ""
    unit: str = "个"
    quantity: float = 1.0
    remark: str = ""
    placement: str = ""
    pdf_page_label: str = ""


@app.put("/api/projects/{project_id}/bom/item")
def update_bom_item(project_id: str, payload: BOMItemUpdatePayload) -> Dict[str, Any]:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.results[project_id]

    target_list = (
        res.itemized_bom
        if (payload.view_mode == "itemized" or (payload.view_mode == "primary" and res.extraction_mode == "PART_LIST_MODE"))
        else res.aggregated_bom
    )

    found = next((it for it in target_list if it.id == payload.id), None)
    if not found and payload.seq_no is not None:
        found = next((it for it in target_list if it.seq_no == payload.seq_no), None)
    if not found:
        raise HTTPException(status_code=404, detail=f"BOM item {payload.id} not found")

    found.tag = payload.tag
    found.brand = payload.brand
    found.name = payload.name
    found.model = payload.model
    found.unit = payload.unit
    found.quantity = payload.quantity
    found.remark = payload.remark
    found.placement = payload.placement
    if payload.pdf_page_label:
        found.pdf_page_label = payload.pdf_page_label
    found.source_type = SourceType.MANUAL_EDITED.value
    found.needs_review = False

    # If user edited aggregated row in PART_LIST_MODE, also sync matching itemized rows
    if res.extraction_mode == "PART_LIST_MODE" and target_list is res.aggregated_bom:
        for it in res.itemized_bom:
            if it.tag in found.tags or it.model == found.raw_model:
                it.brand = payload.brand
                it.name = payload.name
                it.model = payload.model
                it.source_type = SourceType.MANUAL_EDITED.value

    store.sync_and_persist(project_id)
    return _serialize_project_response(project_id)


class BOMItemCreatePayload(BaseModel):
    tag: str = "-NEW1"
    brand: str = ""
    name: str = ""
    model: str = ""
    unit: str = "个"
    quantity: float = 1.0
    remark: str = "人工网页新增"
    primary_page: int = 1
    primary_bbox: List[float] = [0.1, 0.1, 0.3, 0.2]


@app.post("/api/projects/{project_id}/bom/item")
def create_bom_item(project_id: str, payload: BOMItemCreatePayload) -> Dict[str, Any]:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.results[project_id]
    target_list = res.itemized_bom if res.extraction_mode == "PART_LIST_MODE" else res.aggregated_bom
    next_seq = max((it.seq_no for it in target_list), default=0) + 1
    new_id = f"manual_{next_seq}_{len(target_list)+1}"

    new_item = BOMItem(
        id=new_id,
        seq_no=next_seq,
        tag=payload.tag,
        tags=[payload.tag] if payload.tag else [],
        name=payload.name,
        model=payload.model,
        raw_model=payload.model,
        brand=payload.brand,
        unit=payload.unit,
        quantity=payload.quantity,
        remark=payload.remark,
        pdf_page_label=f"第{payload.primary_page}页",
        primary_page=payload.primary_page,
        primary_bbox=payload.primary_bbox,
        evidences=[
            EvidenceLocation(
                page=payload.primary_page,
                bbox=[0, 0, 100, 100],
                norm_bbox=payload.primary_bbox,
                raw_text=f"{payload.tag} {payload.model}",
                role="manual",
            )
        ],
        source_type=SourceType.MANUAL_EDITED.value,
        category="primary",
    )
    target_list.append(new_item)
    store.sync_and_persist(project_id)
    return _serialize_project_response(project_id)


@app.delete("/api/projects/{project_id}/bom/item/{item_id}")
def delete_bom_item(project_id: str, item_id: str) -> Dict[str, Any]:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.results[project_id]

    if res.extraction_mode == "PART_LIST_MODE":
        res.itemized_bom = [it for it in res.itemized_bom if it.id != item_id]
    else:
        res.aggregated_bom = [it for it in res.aggregated_bom if it.id != item_id]

    store.sync_and_persist(project_id)
    return _serialize_project_response(project_id)


@app.post("/api/projects/{project_id}/filtered/{item_id}/restore")
def restore_filtered_item(project_id: str, item_id: str) -> Dict[str, Any]:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.results[project_id]
    found = next((it for it in res.filtered_items if it.id == item_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Filtered item not found")

    res.filtered_items = [it for it in res.filtered_items if it.id != item_id]
    if found.quantity <= 0:
        found.quantity = 1
    found.source_type = SourceType.MANUAL_EDITED.value
    found.review_note = "已由工程师从过滤列表恢复"
    res.itemized_bom.append(found)
    res.itemized_bom.sort(key=lambda x: x.seq_no)

    store.sync_and_persist(project_id)
    return _serialize_project_response(project_id)


class ReviewActionPayload(BaseModel):
    action: str  # "ACCEPT" | "IGNORE" | "RESET"


@app.post("/api/projects/{project_id}/reviews/{review_id}/action")
def handle_review_action(project_id: str, review_id: str, payload: ReviewActionPayload) -> Dict[str, Any]:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.results[project_id]
    rev = next((r for r in res.review_items if r.id == review_id), None)
    if not rev:
        raise HTTPException(status_code=404, detail="Review item not found")

    if payload.action == "ACCEPT":
        if rev.status != "ACCEPTED" and rev.proposed_data:
            pdata = rev.proposed_data
            target_list = res.itemized_bom if res.extraction_mode == "PART_LIST_MODE" else res.aggregated_bom
            if rev.action_type == "ADD_ITEM":
                next_seq = max((it.seq_no for it in target_list), default=0) + 1
                new_item = BOMItem(
                    id=f"rev_added_{next_seq}",
                    seq_no=next_seq,
                    tag=pdata.get("tag", rev.tag),
                    tags=[pdata.get("tag", rev.tag)],
                    name=pdata.get("name", ""),
                    model=pdata.get("model", ""),
                    raw_model=pdata.get("model", ""),
                    brand=pdata.get("brand", ""),
                    unit=pdata.get("unit", "个"),
                    quantity=float(pdata.get("quantity", 1)),
                    remark=pdata.get("remark", "审核采纳补漏"),
                    pdf_page_label=pdata.get("pdf_page_label", f"第{rev.page}页"),
                    primary_page=rev.page,
                    primary_bbox=rev.norm_bbox,
                    evidences=[
                        EvidenceLocation(
                            page=rev.page,
                            bbox=[0, 0, 100, 100],
                            norm_bbox=rev.norm_bbox,
                            raw_text=rev.title,
                            role="review_accepted",
                        )
                    ],
                    source_type=SourceType.MANUAL_EDITED.value,
                )
                target_list.append(new_item)
            elif rev.action_type == "UPDATE_ITEM":
                t_seq = pdata.get("target_seq_no")
                target_item = next((it for it in target_list if it.seq_no == t_seq), None)
                if target_item:
                    if "brand" in pdata:
                        target_item.brand = pdata["brand"]
                    if "model" in pdata:
                        target_item.model = pdata["model"]
                    target_item.source_type = SourceType.MANUAL_EDITED.value
                    target_item.needs_review = False
        rev.status = "ACCEPTED"
    elif payload.action == "IGNORE":
        rev.status = "IGNORED"
    else:
        rev.status = "PENDING"

    store.sync_and_persist(project_id)
    return _serialize_project_response(project_id)


@app.get("/api/projects/{project_id}/export/excel")
def download_excel(project_id: str) -> FileResponse:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.sync_and_persist(project_id)
    excel_path = Path(res.summary["excel_path"])
    filename = excel_path.name
    encoded_name = urllib.parse.quote(filename)
    return FileResponse(
        path=str(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )


@app.get("/api/projects/{project_id}/export/json")
def download_json(project_id: str) -> FileResponse:
    if project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    res = store.sync_and_persist(project_id)
    json_path = Path(res.summary["json_path"])
    filename = json_path.name
    encoded_name = urllib.parse.quote(filename)
    return FileResponse(
        path=str(json_path),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )


@app.get("/api/rules")
def get_rules_config() -> Dict[str, Any]:
    return load_rules_config(RULES_PATH)


@app.put("/api/rules")
def update_rules_config(new_rules: Dict[str, Any]) -> Dict[str, Any]:
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(new_rules, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "rules": new_rules}


@app.get("/api/vlm-config")
def get_vlm_config() -> Dict[str, Any]:
    cfg = VisionRecognizer.load_saved_config()
    return cfg


@app.put("/api/vlm-config")
def update_vlm_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    saved = VisionRecognizer.save_config(cfg)
    return {"status": "ok", "config": saved}


@app.post("/api/projects/{project_id}/pages/{page_num}/ai-recognize")
def ai_recognize_page(project_id: str, page_num: int) -> Dict[str, Any]:
    if project_id not in store.pdf_paths or project_id not in store.results:
        raise HTTPException(status_code=404, detail="Project not found")
    pdf_path = store.pdf_paths[project_id]
    res = store.results[project_id]
    vision = VisionRecognizer(crops_dir=OUTPUT_DIR / project_id / "crops")

    doc = pymupdf.open(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(doc):
            raise HTTPException(status_code=400, detail="Invalid page number")
        page = doc[page_num - 1]
        try:
            ai_items = vision.extract_page_components_via_vlm(
                page, page_num, OUTPUT_DIR / project_id / "crops"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        doc.close()

    target_list = res.itemized_bom if res.extraction_mode == "PART_LIST_MODE" else res.aggregated_bom
    next_seq = max((it.seq_no for it in target_list), default=0)
    added_count = 0

    for raw in ai_items:
        next_seq += 1
        added_count += 1
        nb = raw.get("norm_bbox") or [0.15, 0.2, 0.35, 0.3]
        tag_val = str(raw.get("tag") or f"-AI{added_count}")
        model_val = str(raw.get("model") or "")
        new_item = BOMItem(
            id=f"vlm_p{page_num}_{next_seq}",
            seq_no=next_seq,
            tag=tag_val,
            tags=[tag_val],
            name=str(raw.get("name") or "未命名元器件"),
            model=model_val,
            raw_model=model_val,
            brand=str(raw.get("brand") or ""),
            unit=str(raw.get("unit") or "个"),
            quantity=float(raw.get("quantity") or 1),
            remark=str(raw.get("remark") or f"第{page_num}页AI视觉识别"),
            pdf_page_label=f"第{page_num}页",
            primary_page=page_num,
            primary_bbox=nb if len(nb) == 4 else [0.15, 0.2, 0.35, 0.3],
            evidences=[
                EvidenceLocation(
                    page=page_num,
                    bbox=[0, 0, 100, 100],
                    norm_bbox=nb if len(nb) == 4 else [0.15, 0.2, 0.35, 0.3],
                    raw_text=f"{tag_val} {model_val}",
                    role="vlm_recognized",
                )
            ],
            source_type=SourceType.MANUAL_EDITED.value,
            review_note=f"由多模态视觉模型 ({vision.model}) 从第 {page_num} 页图像识别提取",
        )
        target_list.append(new_item)

    store.sync_and_persist(project_id)
    out = _serialize_project_response(project_id)
    out["ai_added_count"] = added_count
    return out

