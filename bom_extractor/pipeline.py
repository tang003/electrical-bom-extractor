from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import pymupdf

from .classifier import classify_pdf_pages
from .part_list_extractor import extract_from_part_list_pdf
from .schematic_extractor import extract_from_schematic_pdf
from .rules import load_rules_config, apply_engineering_rules, build_aggregated_from_itemized
from .exporter import export_to_excel, export_to_json
from .models import ExtractionResult, SourceType, ReviewItem
from .cad_import import native_document, attach_cad_evidence
from .ocr_engine import DualChannelOCREngine
from .yolo_evolver import YOLOSelfEvolver
from .quality import extraction_quality
from .procurement import procurement_summary
from .device_inventory import build_device_inventory


def run_extraction_pipeline(
    pdf_path: Path | str,
    output_dir: Optional[Path | str] = None,
    rules_config_path: Optional[Path | str] = None,
    project_id: Optional[str] = None,
    workflow_mode: str = 'SCHEMATIC_CHECK',
    table_pages: Optional[list[int]] = None,
) -> ExtractionResult:
    """
    Execute the complete standardized 8-stage Electrical BOM Extraction pipeline:
    1. Analyze PDF page structure & classify page types
    2. Extract spatial text, tags, models, and bounding boxes
    3. Bind component symbols/tags to models & crop vector regions for VLM
    4. Consolidate cross-page coils, contacts, and physical devices
    5. Count physical procurement quantities
    6. Apply enterprise electrical engineering rules & validate associations
    7. Detect suspected omissions, duplicates, and model conflicts
    8. Export structured BOM (Excel + JSON) with provenance coordinates & review report
    """
    pdf_file = Path(pdf_path)
    from .table_workflow import TABLE_MODE,CHECK_MODE,table_result,read_declared_tables
    if workflow_mode not in {TABLE_MODE,CHECK_MODE}:raise ValueError('无效的处理模式')
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    proj_id = project_id or pdf_file.stem.replace(" ", "_")
    out_dir = Path(output_dir) if output_dir else (pdf_file.parent.parent / "output" / proj_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"

    rules_cfg = load_rules_config(Path(rules_config_path) if rules_config_path else None)

    doc = pymupdf.open(str(pdf_file))
    try:
        cad = native_document(doc)
        # Stage 1: Page classification
        extraction_mode, project_name, drawing_no, pages_info = classify_pdf_pages(doc)
        if cad:
            for info, raw in zip(pages_info, cad["pages"]):
                info.title = raw["title"]
                heading = raw.get("cabinet_heading")
                if heading and not info.location_code:
                    info.location_code = "+" + heading["name"]
                info.extraction_diagnostics["cad"] = {k:v for k,v in raw.items() if k!="spans"}
            drawing_no = drawing_no or next((p["attributes"].get("图号", "") for p in cad["pages"] if p["attributes"].get("图号")), "")
        if table_pages is not None and (not table_pages or any(type(p)!=int or not 1<=p<=len(pages_info) for p in table_pages)):
            raise ValueError('部件表页码无效')
        if workflow_mode==TABLE_MODE:
            result=table_result(doc,pages_info,pdf_file,proj_id,project_name,drawing_no,out_dir,table_pages)
            result.summary['excel_path']=str(out_dir/f'{pdf_file.stem}_部件表提取清单_未经原理图核对.xlsx')
            result.summary['json_path']=str(out_dir/f'{pdf_file.stem}_提取结果.json')
            export_to_json(result,result.summary['json_path'])
            export_to_excel(result,result.summary['excel_path'])
            return result
        stage1_path = out_dir / "stage1_page_structure.json"
        with open(stage1_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pdf_filename": pdf_file.name,
                    "extraction_mode": extraction_mode,
                    "project_name": project_name,
                    "drawing_no": drawing_no,
                    "page_count": len(pages_info),
                    "pages": [p.to_dict() for p in pages_info],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Independent phase: tables are masked in both pixels and native/OCR text.
        import copy
        import time
        from .independent import prepare_independent_view, VIEW_NAME, POLICY
        from .reconciliation import reconcile
        original_pages = copy.deepcopy(pages_info)
        view_path = out_dir / VIEW_NAME
        isolation = prepare_independent_view(pdf_file, view_path, original_pages)
        extraction_mode = "SCHEMATIC_MODE"
        with pymupdf.open(view_path) as schematic_doc:
            _, _, _, clean_pages = classify_pdf_pages(schematic_doc)
            if cad:
                for clean, original in zip(clean_pages, original_pages):
                    clean.location_code = original.location_code
            eligible = [p for p in clean_pages if p.page_num in isolation['schematic_pages']]
            device_inventory = build_device_inventory(schematic_doc, eligible)
            device_inventory['view_sha256']=isolation['view_sha256']
            (out_dir/'device_inventory.json').write_text(
                json.dumps(device_inventory, ensure_ascii=False, indent=2), encoding='utf-8')
            raw_items, review_items = extract_from_schematic_pdf(
                schematic_doc, eligible, rules_cfg, crops_dir=crops_dir, inventory=device_inventory)
        clean_by_page = {p.page_num:p for p in clean_pages}
        for original, mask in zip(original_pages, isolation['pages']):
            original.extraction_diagnostics['independent_analysis'] = mask
            clean=clean_by_page[original.page_num]
            original.extraction_diagnostics.update(clean.extraction_diagnostics)
            original.text_block_count=clean.text_block_count
        pages_info = original_pages

        stage2_path = out_dir / "stage2_raw_extracted.json"
        with open(stage2_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_item_count": len(raw_items),
                    "raw_items": [it.to_dict() for it in raw_items],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Stage 5-7: Deterministic engineering rules & anomaly validation
        itemized_bom, aggregated_bom, filtered_items, review_items = apply_engineering_rules(
            raw_items, review_items, extraction_mode, rules_cfg
        )
        for item in itemized_bom + filtered_items:
            if not item.model_origin:
                item.model_origin = 'schematic'
        if cad:
            from types import SimpleNamespace
            attach_cad_evidence(SimpleNamespace(summary={},pages=pages_info,itemized_bom=itemized_bom,
                aggregated_bom=aggregated_bom,filtered_items=filtered_items,independent_bom=[],declared_parts=[]),cad)
        independent_bom = copy.deepcopy(itemized_bom + filtered_items)
        isolation['filtered_ids'] = [i.id for i in filtered_items]
        snapshot = {'policy':POLICY, 'created_at':time.time(), 'source_sha256':isolation['source_sha256'],
                    'view_sha256':isolation['view_sha256'], 'items':[i.to_dict() for i in independent_bom]}
        snapshot_bytes = json.dumps(snapshot,ensure_ascii=False,indent=2).encode('utf-8')
        (out_dir/'independent_snapshot.json').write_bytes(snapshot_bytes)
        isolation['snapshot_sha256'] = __import__('hashlib').sha256(snapshot_bytes).hexdigest()
        isolation['snapshot_saved_at'] = snapshot['created_at']

        # Reference phase: raw parts tables are a separate source, never input to stage 2.
        declared_parts = []
        reference_status = 'absent'
        reference_error = ''
        try:
            declared_parts,reference_reviews,reference_report=read_declared_tables(doc,original_pages)
            review_items.extend(reference_reviews)
            reference_status=reference_report['status']
            isolation['reference_report']=reference_report
        except Exception as exc:
            reference_status,reference_error = 'failed',str(exc)[:300]
            review_items.append(ReviewItem('declared_read_failed','WARNING','REFERENCE_TABLE_READ','','图内清单读取未完成',
                                          '原理图独立结果已保留，图内清单读取失败，未将其内容填入主 BOM。','核实原表与导入记录。',1))
        isolation['reference_loaded_at'] = time.time()
        (out_dir/'declared_parts.json').write_text(json.dumps([i.to_dict() for i in declared_parts],ensure_ascii=False,indent=2),encoding='utf-8')

        if cad:
            for index, warning in enumerate(cad["source"]["warnings"]):
                review_items.append(ReviewItem(
                    f"cad_import_{index}", "WARNING", "CAD_IMPORT_REVIEW", "", "CAD 导入待核对", warning,
                    "核对 CAD 原文件、图框有效性和导入报告，再确认采购清单。", 1, scope='project',
                ))

        # Refresh page diagnostics after native/OCR extraction and spatial binding.
        structure=json.loads(stage1_path.read_text(encoding="utf-8"))
        structure["pages"]=[p.to_dict() for p in pages_info]
        structure["extraction_mode"]=extraction_mode
        structure["independent_analysis"]=isolation
        stage1_path.write_text(json.dumps(structure,ensure_ascii=False,indent=2),encoding="utf-8")
        quality=extraction_quality(eligible,itemized_bom,filtered_items,review_items)
        # Build summary metrics
        primary_items = itemized_bom if extraction_mode == "PART_LIST_MODE" else aggregated_bom
        explicit_count = sum(1 for it in primary_items if it.source_type == SourceType.EXPLICIT.value)
        inferred_count = sum(1 for it in primary_items if it.source_type == SourceType.TOPOLOGY_INFERRED.value)
        rule_count = sum(1 for it in primary_items if it.source_type == SourceType.RULE_COMPLETED.value)
        aux_count = sum(1 for it in primary_items if it.source_type == SourceType.ESTIMATED_AUX.value)

        summary = {
            'workflow_mode':CHECK_MODE,
            "extraction_quality": quality,
            "total_pages": len(pages_info),
            "extraction_mode": extraction_mode,
            "extraction_mode_label": "原理图独立识别 + 图内清单隔离核对",
            "independent_analysis": isolation,
            "declared_reference_status": reference_status,
            "declared_reference_error": reference_error,
            "itemized_count": len(itemized_bom),
            "aggregated_count": len(aggregated_bom),
            "filtered_count": len(filtered_items),
            "review_count": len(review_items),
            "conflict_count": sum(1 for r in review_items if r.severity == "CONFLICT"),
            "warning_count": sum(1 for r in review_items if r.severity == "WARNING"),
            "info_count": sum(1 for r in review_items if r.severity == "INFO"),
            "source_breakdown": {
                "explicit": explicit_count,
                "unresolved": sum(it.source_type == SourceType.UNRESOLVED.value for it in primary_items),
                "vision_recognized": sum(it.source_type == SourceType.VISION_RECOGNIZED.value for it in primary_items),
                "manual_edited": sum(it.source_type == SourceType.MANUAL_EDITED.value for it in primary_items),
                "topology_inferred": inferred_count,
                "rule_completed": rule_count,
                "estimated_aux": aux_count,
            },
            "output_dir": str(out_dir),
        }

        result = ExtractionResult(
            project_id=proj_id,
            pdf_filename=pdf_file.name,
            project_name=project_name,
            drawing_no=drawing_no or (pdf_file.stem if pdf_file.stem != "source" else ""),
            extraction_mode=extraction_mode,
            page_count=len(pages_info),
            pages=pages_info,
            itemized_bom=itemized_bom,
            aggregated_bom=aggregated_bom,
            filtered_items=filtered_items,
            review_items=review_items,
            summary=summary,
            independent_bom=independent_bom,
            declared_parts=declared_parts,
        )

        result.summary['procurement']=procurement_summary(result)
        attach_cad_evidence(result,cad)
        reconcile(result)
        if cad:
            result.pdf_filename = cad["source"]["source_filename"]
        # Stage 8: Export Excel + JSON & Archive YOLOv11 Self-Evolution Dataset
        yolo_status = {"status": "awaiting_human_verified_symbol_boxes"}
        result.summary["yolo_dataset_status"] = yolo_status
        result.summary["rapidocr_available"] = DualChannelOCREngine.is_available()

        result.summary["excel_path"] = str(out_dir / f"{pdf_file.stem}_标准BOM清单.xlsx")
        result.summary["json_path"] = str(out_dir / f"{pdf_file.stem}_提取结果.json")
        export_to_json(result, result.summary["json_path"])
        export_to_excel(result, result.summary["excel_path"])

        return result
    finally:
        doc.close()
