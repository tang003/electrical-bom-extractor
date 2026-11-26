"""Keep automatically estimated wire lengths outside the working BOM."""
from __future__ import annotations

import re
from .models import SourceType

NOTE = "自动估算导线不纳入 BOM；原始估算仅留作备查。"


def is_estimated_wire(item):
    if item.source_type != SourceType.ESTIMATED_AUX.value:
        return False
    if item.entity_role == 'terminal':
        return False
    name = (item.procurement_name or item.name).strip()
    if re.search(r"端子|接头|连接器|线槽|线管|terminal|connector|gland", name, re.I):
        return False
    return bool(re.search(r"导线|电线|电缆|\b(?:wire|cable)\b", name, re.I)
                or re.match(r"^-?WIRE\(", item.tag, re.I))


def exclude_estimated_wires(result):
    """Apply on load/save and export; retain source snapshots and terminal rows."""
    removed = [i for i in result.itemized_bom if is_estimated_wire(i)]
    removed_aggregates = [i for i in result.aggregated_bom if is_estimated_wire(i)]
    result.itemized_bom = [i for i in result.itemized_bom if not is_estimated_wire(i)]
    result.aggregated_bom = [i for i in result.aggregated_bom if not is_estimated_wire(i)]
    archived = {i.id for i in result.filtered_items}
    for item in removed:
        item.procurement_scope = 'excluded'
        item.review_note = NOTE
        if item.id not in archived:
            result.filtered_items.append(item)
            archived.add(item.id)
    # Repeated application (including restoring a historical row) is idempotent.
    excluded_tags = {t for i in [*removed, *removed_aggregates, *result.filtered_items]
                     if is_estimated_wire(i) for t in (i.tags or [i.tag]) if t}
    live_tags = {t for i in result.itemized_bom for t in (i.tags or [i.tag]) if t}
    only_excluded = excluded_tags - live_tags
    result.review_items = [r for r in result.review_items
                           if not (r.tag and set(re.split(r'[,，、]\s*', r.tag)) <= only_excluded)]
    for review in result.review_items:
        if review.id == 'rev_wires_terminals_summary':
            review.tag = '端子'
            review.title = '端子规格与数量核对'
            review.description = '端子位数与规格仍需核对原图；自动导线米数估算已停用。'
            review.suggestion = '核对端子规格和片数；不再确认估算导线米数。'
    result.summary.update(itemized_count=len(result.itemized_bom),
                          aggregated_count=len(result.aggregated_bom),
                          filtered_count=len(result.filtered_items),
                          review_count=len(result.review_items),
                          pending_review_count=sum(r.status == 'PENDING' for r in result.review_items))
    if removed or removed_aggregates or only_excluded:
        from .procurement import procurement_summary
        from .reconciliation import reconcile
        result.summary['procurement'] = procurement_summary(result)
        primary = result.itemized_bom if result.extraction_mode == 'PART_LIST_MODE' else result.aggregated_bom
        result.summary['source_breakdown'] = {s.value.lower(): sum(i.source_type == s.value for i in primary) for s in SourceType}
        if result.independent_bom or result.declared_parts:
            reconcile(result)
    return result
