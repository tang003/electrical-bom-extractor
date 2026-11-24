"""AI merges must use the position of a model annotation, not merely its spelling."""
from copy import deepcopy

import pytest

from bom_extractor.ai_agent import PLACEHOLDER, merge_observations
from bom_extractor.models import BOMItem, EvidenceLocation
from test_ai_agent import observation, project, record


MODEL = 'ZP-758A'
MODEL_BOX = [.23, .47, .37, .5]


def model_evidence(tag, page=1, box=MODEL_BOX, binding=''):
    return EvidenceLocation(page, box, box, raw_text=MODEL, role='model_text',
                            device_tag=tag, model_binding=binding)


def drawing():
    first = observation('-B18', MODEL)
    first['bbox_normalized'] = [.2, .2, .4, .52]
    neighbor = observation('-W9', None, 'device')
    neighbor['bbox_normalized'] = [.58, .32, .68, .40]
    page = record(observations=[first, neighbor])
    page['spans'] = [
        {'id': 0, 'text': MODEL, 'norm_bbox': MODEL_BOX},
        {'id': 1, 'text': '-B18', 'norm_bbox': [.25, .20, .27, .23]},
        {'id': 2, 'text': '-W9', 'norm_bbox': [.6, .33, .63, .35]},
    ]
    return page


def old_native():
    tag = '+PANEL-W9'
    return BOMItem('near', 1, tag, tags=[tag], model=MODEL, raw_model=MODEL,
                   evidences=[model_evidence(tag)], brand='原型号推定品牌')


def test_replay_moves_only_the_same_printed_occurrence_and_removes_stale_brand():
    neighbor = old_native()
    other_copy = BOMItem('spare', 2, '+PANEL-B27', model=MODEL,
                         evidences=[model_evidence('+PANEL-B27', box=[.81, .42, .91, .45])])
    result = project([neighbor, other_copy])
    page = drawing()
    for _ in range(2):
        merge_observations(result, [page])
        assert neighbor.model == PLACEHOLDER and neighbor.raw_model == ''
        assert neighbor.brand == '待定' and neighbor.quantity == 1
        assert not any(e.role == 'model_text' for e in neighbor.evidences)
        assert {i.tag for i in result.itemized_bom if i.model == MODEL} == {'+PANEL-B18', '+PANEL-B27'}


@pytest.mark.parametrize('other_support', ['another_page', 'verified', 'shared_region', 'missing_tag', 'accessory'])
def test_missing_or_conflicting_localization_preserves_saved_model(other_support):
    item = old_native()
    data = drawing()
    if other_support == 'another_page':
        item.evidences.append(model_evidence(item.tag, page=2))
    elif other_support == 'verified':
        item.evidences[0].verified = True
    elif other_support == 'shared_region':
        data['data']['devices'][0]['bbox_normalized'][2] = .7
    elif other_support == 'missing_tag':
        data['spans'].pop(2)
    else:
        item.parent_tag = '+PANEL-X2'
    merge_observations(project([item]), [data])
    assert item.model == MODEL


def test_new_ai_observation_cannot_reuse_another_devices_enclosed_model_span():
    first_tag = '+PANEL-B18'
    first = BOMItem('first', 1, first_tag, model=MODEL,
                    evidences=[model_evidence(first_tag, binding='enclosure')])
    data = drawing()
    wrong = data['data']['devices'][1]
    wrong.update(model_raw=MODEL, model_span_ids=[0, 2], bbox_normalized=[.2, .2, .7, .55])
    original = deepcopy(data)
    result = project([first])
    merge_observations(result, [data])
    assert first.model == MODEL
    assert next(item for item in result.itemized_bom if item.tag == '+PANEL-W9').model == PLACEHOLDER
    assert any(MODEL in review.description and '标给其他设备' in review.description for review in result.review_items)
    assert data == original
