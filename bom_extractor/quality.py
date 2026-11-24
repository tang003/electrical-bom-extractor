"""Extraction diagnostics are coverage of observed evidence, never ground-truth accuracy."""
from __future__ import annotations
from .models import SourceType


def extraction_quality(pages, items, filtered, reviews):
    candidates={tag for page in pages for tag in page.extraction_diagnostics.get("device_tags", [])}
    represented={tag for item in items+filtered for tag in (item.tags or [item.tag])}
    missed=sorted(candidates-represented)
    devices=[item for item in items if item.category!="auxiliary" and not item.is_quantity_adjustment]
    unresolved=[item for item in devices if item.source_type==SourceType.UNRESOLVED.value or not item.model or item.model.startswith(("未提取", "图纸未注"))]
    readable=sum(page.text_block_count>0 for page in pages)
    unbound=sum(page.extraction_diagnostics.get("unbound_models",0) for page in pages)
    native=sum(page.extraction_diagnostics.get("native_characters",0) for page in pages)
    low=bool(missed) or (not devices and unbound>0) or (len(devices)>=5 and len(unresolved)==len(devices)) or readable<len(pages)
    status="LOW_COVERAGE" if low else "NEEDS_REVIEW" if unresolved or any(r.status=="PENDING" for r in reviews) else "EXTRACTED_UNVERIFIED"
    quality={"status":status,"total_pages":len(pages),"readable_text_pages":readable,"native_text_characters":native,
        "detected_device_candidates":len(candidates or represented),"retained_device_candidates":len(devices),
        "filtered_device_candidates":sum(it.category=="field_load" for it in filtered),
        "with_model_count":len(devices)-len(unresolved),"unresolved_model_count":len(unresolved),
        "unbound_model_occurrences":unbound,"unrepresented_device_tags":missed,
        "material_candidates":sum(it.category=='auxiliary' for it in items),
        "unknown_purchase_quantities":sum(it.quantity is None for it in items),
        "coverage_definition":"位号覆盖仅针对已检测候选；不代表完整 BOM 召回率或型号准确率。"}
    quality["message"]=quality_message(quality)
    return quality


def quality_message(quality):
    return (f"文字可读 {quality['readable_text_pages']}/{quality['total_pages']} 页；"
            f"当前 BOM 候选 {quality['retained_device_candidates']} 项，其中有型号 {quality['with_model_count']} 项、"
            f"型号待核实 {quality['unresolved_model_count']} 项；另有 {quality['unbound_model_occurrences']} 处型号候选未关联。")
