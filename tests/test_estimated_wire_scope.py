import copy
import json

import openpyxl
import pytest

import backend.app as backend
from bom_extractor.exporter import export_to_excel
from bom_extractor.material_scope import exclude_estimated_wires
from bom_extractor.models import BOMItem, ExtractionResult, PageInfo, SourceType
from bom_extractor.rules import build_aggregated_from_itemized
from bom_extractor.schematic_extractor import _extract_wires_and_terminals


def sample(mode='SCHEMATIC_MODE'):
    items = [
        BOMItem('estimate', 1, '-WIRE(70mm²)', name='导线', model='70mm²', quantity=123, unit='米', source_type='ESTIMATED_AUX'),
        BOMItem('terminal', 2, '-X1', name='端子', model='UK5N', quantity=12, source_type='ESTIMATED_AUX'),
        BOMItem('explicit', 3, '-W1', name='电缆', model='YJV', quantity=30, unit='米'),
        BOMItem('manual', 4, '-W2', name='导线', model='BVR', quantity=20, unit='米', source_type='MANUAL_EDITED'),
    ]
    return ExtractionResult('proj_wire', 'drawing.pdf', '范围测试', '', mode, 1,
                            itemized_bom=items, aggregated_bom=build_aggregated_from_itemized(items),
                            independent_bom=copy.deepcopy(items))


def test_scope_preserves_terminals_explicit_and_manual_lengths_and_source_snapshot():
    r = sample()
    original = copy.deepcopy(r.independent_bom)
    exclude_estimated_wires(r)
    assert [i.id for i in r.itemized_bom] == ['terminal', 'explicit', 'manual']
    assert [i.id for i in r.filtered_items] == ['estimate']
    assert r.filtered_items[0].quantity == 123
    assert r.independent_bom == original
    assert not any('estimate' in x['schematic_ids'] for x in r.reconciliation)
    once = r.to_dict()
    exclude_estimated_wires(r)
    assert r.to_dict() == once


@pytest.mark.parametrize('mode', ['SCHEMATIC_MODE', 'PART_LIST_MODE'])
def test_export_of_legacy_results_does_not_include_estimated_wire_or_mutate_source(tmp_path, mode):
    r = sample(mode)
    before = r.to_dict()
    path = tmp_path / 'bom.xlsx'
    export_to_excel(r, path)
    with open(path, 'rb') as stream:
        workbook = openpyxl.load_workbook(stream)
        tags = [row[6] for row in workbook['物料清单'].iter_rows(min_row=7, values_only=True)]
        assert '-WIRE(70mm²)' not in tags
        assert set(tags) == {'-X1', '-W1', '-W2'}
        workbook.close()
    assert r.to_dict() == before


def test_existing_project_load_and_save_apply_scope_without_reextracting():
    r = sample()
    r.summary['state_schema_version'] = 3
    folder = backend.OUTPUT_DIR / r.project_id
    folder.mkdir(parents=True)
    state = folder / 'project_state.json'
    state.write_text(json.dumps(r.to_dict()), encoding='utf-8')
    original_bytes = state.read_bytes()
    backend.store.initialize_defaults()
    loaded = backend.store.results[r.project_id]
    assert [i.id for i in loaded.itemized_bom] == ['terminal', 'explicit', 'manual']
    assert state.read_bytes() == original_bytes
    backend.store.sync_and_persist(r.project_id)
    saved = json.loads(state.read_text())
    assert [i['id'] for i in saved['filtered_items']] == ['estimate']
    assert saved['summary']['itemized_count'] == 3


def test_wire_estimation_stops_but_terminal_detection_remains():
    pages = [PageInfo(1, 'MAIN_CIRCUIT', '主回路'), PageInfo(2, 'CONTROL_CIRCUIT', '控制回路')]
    spans = {1: [{'text': '70mm²', 'bbox': [20, 20, 50, 30], 'norm_bbox': [.2, .2, .5, .3]}]}
    items, _ = _extract_wires_and_terminals(pages, spans, 0,
        {'wire_terminal_estimation': {'gauge_to_terminal_rating': {'70': '200A'}, 'incoming_terminal_poles': 3}},
        control_device_count=20)
    assert items and all(i.name == '端子' for i in items)
    assert items[0].model == '200A' and items[0].quantity == 3
