from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .models import BOMItem, ReviewItem, SourceType


def load_rules_config(config_path: Path | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "rules_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_aggregated_from_itemized(itemized_items: List[BOMItem]) -> List[BOMItem]:
    """
    Group itemized BOM rows by (brand, name, model, unit) to produce a consolidated
    procurement summary table (`合并采购汇总表`), preserving all tags and evidences.
    """
    grouped: Dict[tuple, BOMItem] = {}
    order_keys: List[tuple] = []

    for it in itemized_items:
        name = it.name
        if it.procurement_name and it.source_type == SourceType.EXPLICIT.value and it.raw_model == it.model:
            name = it.procurement_name
        key = (
            (it.brand or "").strip(),
            (name or "").strip(),
            (it.model or "").strip(),
            (it.unit or "个").strip(),
            it.specification.strip(), it.procurement_scope, it.entity_role, it.parent_tag,
            it.quantity_status, it.quantity_basis,
            it.model_origin if it.entity_role == 'terminal' and it.model_origin == 'terminal_strip_inference' else '',
            (it.group_identity or it.id) if it.quantity is None or (it.category != "auxiliary" and (it.source_type == SourceType.UNRESOLVED.value or not it.model or it.model.startswith(("未提取", "图纸未注")))) else "",
        )
        if key not in grouped:
            cloned = copy.deepcopy(it)
            cloned.name = name
            cloned.tags = list(it.tags) if it.tags else ([it.tag] if it.tag else [])
            cloned.member_ids = [it.id]
            grouped[key] = cloned
            order_keys.append(key)
        else:
            agg = grouped[key]
            agg.quantity = None if agg.quantity is None or it.quantity is None else agg.quantity + it.quantity
            agg.member_ids.append(it.id)
            for tg in (it.tags or [it.tag]):
                if tg and tg not in agg.tags:
                    agg.tags.append(tg)
            agg.evidences.extend(it.evidences)
            from .components import merge_parts
            agg.component_parts = merge_parts(agg.component_parts, it.component_parts)
            if agg.model_origin!=it.model_origin:agg.model_origin='mixed'
            if agg.procurement_scope_basis!=it.procurement_scope_basis:agg.procurement_scope_basis='mixed'
            agg.parts_reference_id=','.join(dict.fromkeys(filter(None,[*agg.parts_reference_id.split(','),it.parts_reference_id])))
            if it.needs_review:
                agg.needs_review = True

    result: List[BOMItem] = []
    for idx, key in enumerate(order_keys, start=1):
        agg = grouped[key]
        agg.seq_no = idx
        agg.id = "agg_" + hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16]
        if agg.tags:
            agg.tag = ", ".join(agg.tags)
        pages = sorted({ev.page for ev in agg.evidences}) if agg.evidences else [agg.primary_page]
        if len(pages) > 1:
            agg.pdf_page_label = ", ".join(f"第{p}页" for p in pages[:4]) + (f" 等" if len(pages) > 4 else "")
        result.append(agg)
    return result


def apply_engineering_rules(
    raw_items: List[BOMItem],
    review_items: List[ReviewItem],
    extraction_mode: str,
    rules_config: Dict[str, Any],
) -> Tuple[List[BOMItem], List[BOMItem], List[BOMItem], List[ReviewItem]]:
    """
    Apply deterministic enterprise electrical engineering rules:
    - Filter out 0-quantity and non-electrical placeholder field parts
    - Append mandatory PLC accessories (front connectors, mounting rail, memory card)
    - Return (itemized_bom, aggregated_bom, filtered_items, review_items)
    """
    filter_cfg = rules_config.get("filter_rules", {})
    patterns = [re.compile(p) for p in filter_cfg.get("exclude_placeholder_models", [])]
    kept, filtered = [], []
    completed = []
    normalize = lambda value: re.sub(r"\s+", "", value).upper()
    for original in raw_items:
        item = copy.deepcopy(original)
        if item.category == "field_load" and filter_cfg.get("exclude_field_loads_in_schematic", True):
            item.review_note = "原理图现场负载，按默认柜内采购范围过滤；可在已过滤列表恢复。"
            filtered.append(item)
            continue
        if (filter_cfg.get("exclude_zero_quantity", True) and item.quantity is not None and item.quantity <= 0) or any(p.fullmatch(item.model.strip()) for p in patterns):
            item.review_note = "按企业规则过滤的零数量或非电控采购件，可人工恢复"
            filtered.append(item)
            continue
        for rule in rules_config.get("plc_accessory_rules", []):
            if normalize(item.model.split("+")[0]) != normalize(rule["match_model"]):
                continue
            existing = {normalize(part) for part in item.model.split("+")}
            additions = [part for part in rule["append_model"].split("+") if part and normalize(part) not in existing]
            if additions:
                proposal=item.model + "+" + "+".join(additions)
                if not any(s.get('model')==proposal for s in item.selection_suggestions):
                    item.selection_suggestions.append({"model":proposal,"reason":rule["reason"],"source":"enterprise_rule","status":"PENDING"})
                item.needs_review = True
                completed.append(item)
        kept.append(item)
    reviews = copy.deepcopy(review_items)
    if completed:
        reviews.append(ReviewItem(id="rev_plc_accessories", severity="INFO", issue_type="RULE_COMPLETED",
            tag=", ".join(it.tag for it in completed), title="PLC 附件选型建议，未改写图纸型号",
            description="；".join(f"{it.tag}: {it.selection_suggestions[-1]['model']}" for it in completed),
            suggestion="核实前连接器、导轨和存储卡是否符合项目要求", page=completed[0].primary_page,
            norm_bbox=completed[0].primary_bbox))
    if filtered:
        reviews.append(ReviewItem(id="rev_filtered_items", severity="INFO", issue_type="FILTERED_ITEM",
            tag=", ".join(it.tag for it in filtered), title=f"已过滤 {len(filtered)} 项物料，可人工恢复",
            description="；".join(f"{it.tag}: {it.model}, 数量 {it.quantity}" for it in filtered),
            suggestion="请在过滤列表中核实采购范围", page=filtered[0].primary_page, norm_bbox=filtered[0].primary_bbox))
    series = [it for it in kept if it.source_type == SourceType.RULE_COMPLETED.value]
    if series and extraction_mode == "SCHEMATIC_MODE":
        reviews.append(ReviewItem(id="rev_series_mapping", severity="INFO", issue_type="RULE_COMPLETED",
            tag=", ".join(it.tag for it in series), title=f"{len(series)} 项企业选型规则补全待确认",
            description="型号或附件来自企业配置，请逐项核实", suggestion="核实电压、颜色、触点和附件规格",
            page=series[0].primary_page, norm_bbox=series[0].primary_bbox))
    if extraction_mode == 'SCHEMATIC_MODE' and filter_cfg.get('exclude_field_loads_in_schematic', True):
        from .scope_rules import apply_scope_defaults
        apply_scope_defaults(kept)
    return kept, build_aggregated_from_itemized(kept), filtered, reviews
