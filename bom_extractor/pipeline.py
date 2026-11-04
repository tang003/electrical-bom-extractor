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
from .models import ExtractionResult, SourceType


def run_extraction_pipeline(
    pdf_path: Path | str,
    output_dir: Optional[Path | str] = None,
    rules_config_path: Optional[Path | str] = None,
    project_id: Optional[str] = None,
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
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    proj_id = project_id or pdf_file.stem.replace(" ", "_")
    out_dir = Path(output_dir) if output_dir else (pdf_file.parent.parent / "output" / proj_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"

    rules_cfg = load_rules_config(Path(rules_config_path) if rules_config_path else None)

    doc = pymupdf.open(str(pdf_file))
    try:
        # Stage 1: Page classification
        extraction_mode, project_name, drawing_no, pages_info = classify_pdf_pages(doc)
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

        # Stage 2-4: Raw extraction & cross-page consolidation
        if extraction_mode == "PART_LIST_MODE":
            raw_items, review_items = extract_from_part_list_pdf(doc, pages_info)
        else:
            raw_items, review_items = extract_from_schematic_pdf(doc, pages_info, rules_cfg, crops_dir=crops_dir)

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

        # Build summary metrics
        primary_items = itemized_bom if extraction_mode == "PART_LIST_MODE" else aggregated_bom
        explicit_count = sum(1 for it in primary_items if it.source_type == SourceType.EXPLICIT.value)
        inferred_count = sum(1 for it in primary_items if it.source_type == SourceType.TOPOLOGY_INFERRED.value)
        rule_count = sum(1 for it in primary_items if it.source_type == SourceType.RULE_COMPLETED.value)
        aux_count = sum(1 for it in primary_items if it.source_type == SourceType.ESTIMATED_AUX.value)

        summary = {
            "total_pages": len(pages_info),
            "extraction_mode": extraction_mode,
            "extraction_mode_label": (
                "模式A：自带《部件列表》精准提取 + 原理图交叉校验"
                if extraction_mode == "PART_LIST_MODE"
                else "模式B：纯原理图空间提取 + 跨页归并 + 视觉局部切片"
            ),
            "itemized_count": len(itemized_bom),
            "aggregated_count": len(aggregated_bom),
            "filtered_count": len(filtered_items),
            "review_count": len(review_items),
            "conflict_count": sum(1 for r in review_items if r.severity == "CONFLICT"),
            "warning_count": sum(1 for r in review_items if r.severity == "WARNING"),
            "info_count": sum(1 for r in review_items if r.severity == "INFO"),
            "source_breakdown": {
                "explicit": explicit_count,
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
            drawing_no=drawing_no or pdf_file.stem,
            extraction_mode=extraction_mode,
            page_count=len(pages_info),
            pages=pages_info,
            itemized_bom=itemized_bom,
            aggregated_bom=aggregated_bom,
            filtered_items=filtered_items,
            review_items=review_items,
            summary=summary,
        )

        # Stage 8: Export Excel + JSON
        excel_path = export_to_excel(result, out_dir / f"{pdf_file.stem}_标准BOM清单.xlsx")
        json_path = export_to_json(result, out_dir / f"{pdf_file.stem}_提取结果.json")
        result.summary["excel_path"] = excel_path
        result.summary["json_path"] = json_path
        export_to_json(result, json_path)

        return result
    finally:
        doc.close()
