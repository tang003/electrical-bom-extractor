"""Conservative project-default procurement scope for cabinet schematics."""
from __future__ import annotations

from .models import BOMItem, SourceType
from .notation import device_code, is_field_load


# These are cabinet materials in the project's default cabinet procurement list.
# Switches, sensors, cables, generic U/B/D devices and contact-only symbols stay unknown.
_CABINET_CODES = {
    'F', 'FU', 'QF', 'QS', 'KM', 'KA', 'KV', 'KF', 'KT',
    'AX', 'CPU', 'PD', 'V', 'T', 'FM', 'SX', 'SR',
}


def infer_procurement_scope(item: BOMItem) -> tuple[str, str] | None:
    """Return a rule-backed suggestion; never replace a manual or existing decision."""
    if (item.procurement_scope != 'unknown' or item.procurement_scope_basis == 'manual'
            or item.source_type == SourceType.MANUAL_EDITED.value or item.is_quantity_adjustment):
        return None
    if item.category == 'field_load' or is_field_load(item.tag):
        return 'field', 'drawing_field_load'
    if item.entity_role in {'cable', 'contact', 'assembly'}:
        return None
    code = device_code(item.tag)
    if item.entity_role == 'terminal' and code in {'X', 'XT'}:
        return 'cabinet', 'project_cabinet_default'
    if item.entity_role in {'device', 'coil', 'accessory'} and code in _CABINET_CODES:
        return 'cabinet', 'project_cabinet_default'
    return None


def apply_scope_defaults(items: list[BOMItem]) -> int:
    changed = 0
    for item in items:
        suggestion = infer_procurement_scope(item)
        if suggestion:
            item.procurement_scope, item.procurement_scope_basis = suggestion
            changed += 1
    return changed
