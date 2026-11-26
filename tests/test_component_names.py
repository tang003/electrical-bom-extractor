import copy
import json
import pytest

from bom_extractor.component_names import owned_name, repair_saved_component_names
from bom_extractor.models import BOMItem, EvidenceLocation, ExtractionResult, ReviewItem
from bom_extractor.schematic_extractor import _infer_component_name_and_brand
from bom_extractor.spatial_binding import device_captions


def annotation(text, tag='-D1'):
    return EvidenceLocation(1, [10, 20, 30, 28], [.1, .2, .3, .28],
                            raw_text=text, role='device_annotation', device_tag=tag)


def result_for(item):
    summary = copy.deepcopy(item)
    summary.id = 'aggregate'
    summary.member_ids = [item.id]
    return ExtractionResult('names', 'names.pdf', '', '', 'SCHEMATIC_MODE', 1,
                            itemized_bom=[item], aggregated_bom=[summary],
                            independent_bom=[copy.deepcopy(item)])


def test_drive_caption_binds_to_its_tag_without_stealing_the_neighbours_name():
    def span(text, x, y):
        return dict(text=text, bbox=[x, y, x+45, y+8], cx=x+22.5, cy=y+4)
    tags = [dict(span('-D1', 10, 10), tag='-D1'), dict(span('-E1', 100, 10), tag='-E1')]
    captions = [span('X轴伺服驱动器', 10, 20), span('X轴反馈', 100, 20)]
    bound = device_captions(captions, tags, [])
    assert [entry['text'] for entry in bound['-D1']] == ['X轴伺服驱动器']
    assert '-E1' not in bound
    assert owned_name('-D1', [entry['text'] for entry in bound['-D1']]) == 'X轴伺服驱动器'


def test_feedback_or_conflicting_captions_do_not_establish_a_specific_device_class():
    assert _infer_component_name_and_brand('-E1', '', [])[0] == '电气元器件（类型待核）'
    assert owned_name('-E1', ['4倍数字量细分+1VPP', 'X轴反馈']) == '反馈器件（类型待核）'
    assert owned_name('-D1', ['X轴伺服驱动器', 'Z轴伺服驱动器']) == ''
    assert owned_name('-D1', ['220V AC 36A', 'X轴命令端口', 'COPLEY']) == ''
    assert owned_name('-E1', ['柜内照明灯']) == '柜内照明灯'


def test_saved_name_repair_preserves_quantity_model_and_frozen_facts(tmp_path):
    item = BOMItem('drive', 1, '-D1', name='电气元器件', model='XML-230-18',
                   raw_model='XML-230-18', quantity=None, evidences=[annotation('X轴伺服驱动器')])
    result = result_for(item)
    original = copy.deepcopy(result.independent_bom[0].to_dict())
    assert repair_saved_component_names(result) == ['drive']
    assert item.name == result.aggregated_bom[0].name == 'X轴伺服驱动器'
    assert item.quantity is None and item.model == item.raw_model == 'XML-230-18'
    assert result.independent_bom[0].to_dict() == original
    assert repair_saved_component_names(result) == []
    state = tmp_path / 'state.json'
    state.write_text(json.dumps(result.to_dict(), ensure_ascii=False))
    restored = ExtractionResult.from_dict(json.loads(state.read_text()))
    assert restored.itemized_bom[0].name == 'X轴伺服驱动器'
    assert restored.itemized_bom[0].observations[-1]['previous_name'] == '电气元器件'


@pytest.mark.parametrize('protection', ['manual', 'quantity', 'verified', 'accepted', 'table', 'part_list'])
def test_saved_name_repair_keeps_human_decisions_and_table_text(protection):
    item = BOMItem('drive', 1, '-D1', name='电气元器件', evidences=[annotation('X轴伺服驱动器')])
    result = result_for(item)
    if protection == 'manual': item.source_type = 'MANUAL_EDITED'
    if protection == 'quantity': item.quantity_status = 'confirmed'
    if protection == 'verified': item.evidences[0].verified = True
    if protection == 'accepted':
        result.review_items = [ReviewItem('review', 'INFO', 'MODEL', '-D1', '', '', '', 1, status='ACCEPTED')]
    if protection == 'table': result.summary['workflow_mode'] = 'TABLE_EXTRACT'
    if protection == 'part_list': result.extraction_mode = 'PART_LIST_MODE'
    assert repair_saved_component_names(result) == []
    assert item.name == '电气元器件'


def test_saved_name_repair_ignores_a_neighbour_and_a_mixed_summary():
    one = BOMItem('one', 1, '-D1', name='电气元器件', evidences=[annotation('X轴伺服驱动器', '-D2')])
    result = result_for(one)
    assert repair_saved_component_names(result) == []
    one.evidences = [annotation('X轴伺服驱动器')]
    two = BOMItem('two', 2, '-D2', name='工程师自定义名称', source_type='MANUAL_EDITED')
    result.itemized_bom.append(two)
    result.aggregated_bom[0].member_ids.append(two.id)
    assert repair_saved_component_names(result) == ['one']
    assert result.aggregated_bom[0].name == '电气元器件'
