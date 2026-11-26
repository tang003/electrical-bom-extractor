"""Procurement scope defaults must keep manual decisions and uncertain placement intact."""

from threading import RLock

import backend.app as backend
from bom_extractor.models import BOMItem
from bom_extractor.rules import apply_engineering_rules, build_aggregated_from_itemized
from bom_extractor.scope_rules import apply_scope_defaults


def item(tag: str, role: str = 'device', **values) -> BOMItem:
    return BOMItem(id=tag, seq_no=1, tag=tag, entity_role=role, **values)


def test_scope_defaults_are_conservative_and_idempotent():
    rows = [
        item('-KA20', 'coil'),
        item('-FU1'),
        item('-X2:1~3', 'terminal'),
        item('-X2:4', 'terminal'),
        item('-S1'),
        item('-SQ1'),
        item('-W1', 'cable'),
        item('-KA21', 'contact'),
        item('-M1', category='field_load'),
        item('-KA22', procurement_scope='field'),
        item('-KA23', procurement_scope_basis='manual'),
        item('-KA24', source_type='MANUAL_EDITED'),
        item('-KA25', is_quantity_adjustment=True),
    ]
    assert apply_scope_defaults(rows) == 5
    assert [(row.procurement_scope, row.procurement_scope_basis) for row in rows[:4]] == [
        ('cabinet', 'project_cabinet_default'),
        ('cabinet', 'project_cabinet_default'),
        ('cabinet', 'project_cabinet_default'),
        ('cabinet', 'project_cabinet_default'),
    ]
    assert all(row.procurement_scope == 'unknown' for row in rows[4:8])
    assert rows[8].procurement_scope == 'field'
    assert rows[9].procurement_scope == 'field'
    assert all(row.procurement_scope == 'unknown' for row in rows[10:])
    assert apply_scope_defaults(rows) == 0


def test_schematic_pipeline_applies_default_before_aggregation():
    rows = [item('-KA1', 'coil'), item('-S1'), item('-X1:1', 'terminal')]
    itemized, aggregate, _, _ = apply_engineering_rules(
        rows, [], 'SCHEMATIC_MODE', {'filter_rules': {'exclude_field_loads_in_schematic': True}}
    )
    assert [row.procurement_scope for row in itemized] == ['cabinet', 'unknown', 'cabinet']
    assert [row.procurement_scope for row in aggregate] == ['cabinet', 'unknown', 'cabinet']
    assert all(row.procurement_scope == 'unknown' for row in rows)


def test_existing_project_scope_refresh_preserves_manual_decisions(monkeypatch):
    rows = [item('-KA1', 'coil'), item('-X1:1', 'terminal'),
            item('-S1'), item('-KA2', procurement_scope='field')]

    class Result:
        extraction_mode = 'SCHEMATIC_MODE'
        itemized_bom = rows
        aggregated_bom = []

    class Store:
        lock = RLock()
        results = {'project': Result()}
        sync_calls = 0

        def sync_and_persist(self, project_id):
            self.sync_calls += 1
            self.results[project_id].aggregated_bom = build_aggregated_from_itemized(rows)

    fake = Store()
    monkeypatch.setattr(backend, 'store', fake)
    monkeypatch.setattr(backend, '_serialize_project_response',
                        lambda project_id: {'project_id': project_id,
                                            'scopes': [row.procurement_scope for row in fake.results[project_id].aggregated_bom]})
    response = backend.infer_project_procurement_scope('project')
    assert response == {'project_id': 'project', 'scopes': ['cabinet', 'cabinet', 'unknown', 'field'], 'scope_inferred': 2}
    assert fake.sync_calls == 1
    assert backend.infer_project_procurement_scope('project')['scope_inferred'] == 0
    assert fake.sync_calls == 1
