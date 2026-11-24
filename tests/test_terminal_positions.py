import pymupdf

from bom_extractor.models import BOMItem, PageInfo, ExtractionResult, SourceType
from bom_extractor.reconciliation import _keys
from bom_extractor.rules import build_aggregated_from_itemized
from bom_extractor.terminal_positions import terminal_positions, apply_terminal_positions, vector_terminal_symbols


def span(text, x, y, width=8, height=6):
    box = [x, y, x + width, y + height]
    return {'text': text, 'cx': x + width / 2, 'cy': y + height / 2,
            'bbox': box, 'norm_bbox': [v / 900 if i % 2 == 0 else v / 700
                                     for i, v in enumerate(box)]}


def row(spans, tag, numbers, x, y, model):
    spans.append(span(tag, x - 28, y))
    for index, number in enumerate(numbers):
        px = x + index * 24
        spans.extend([span(str(number), px, y), span(model, px, y - 18)])


def column(spans, tag, numbers, x, y, model):
    spans.append(span(tag, x, y - 25))
    for index, number in enumerate(numbers):
        py = y + index * 20
        spans.extend([span(str(number), x, py), span(model, x, py - 10)])


def test_counts_drawn_positions_across_segments_and_pages():
    pages = [PageInfo(n, 'TERMINAL_DIAGRAM', '端子图', width=900, height=700)
             for n in (4, 17, 18)]
    spans = {4: [], 17: [], 18: []}
    row(spans[4], '-X8', range(1, 7), 100, 100, 'PT 2,5')
    row(spans[4], '-X8', range(1, 6), 100, 200, 'PT 2,5')
    row(spans[4], '-X9', range(1, 11), 500, 100, 'PT 2,5-PE')
    row(spans[4], '-X9', range(11, 14), 500, 200, 'PT 2,5-PE')
    column(spans[17], '-X20', range(1, 17), 300, 60, 'PT2.5')
    column(spans[18], '-X20', range(17, 33), 300, 60, 'PT2.5')
    column(spans[17], '-XT2', range(1, 17), 500, 60, 'PT2.5')
    column(spans[18], '-XT2', range(1, 17), 500, 60, 'PT2.5')

    found = terminal_positions(pages, spans)
    items = [BOMItem(tag, i, tag, tags=[tag], entity_role='terminal', quantity=None,
                     quantity_basis='terminal_positions_required')
             for i, tag in enumerate(('-X8', '-X9', '-X20', '-XT2'), 1)]
    reviews = []
    apply_terminal_positions(items, reviews, found)
    by_tag = {item.terminal_strip_tag: item for item in items}

    assert [(by_tag[tag].quantity, by_tag[tag].model) for tag in ('-X8', '-X9', '-X20')] == [
        (11, 'PT2.5'), (13, 'PT2.5-PE'), (32, 'PT2.5')]
    assert by_tag['-X9'].tag == '-X9:1~13'
    assert by_tag['-X20'].tag == '-X20:1~32'
    assert by_tag['-X20'].quantity_basis == 'drawn_terminal_positions'
    assert '第4页 6+5位' in by_tag['-X8'].remark
    assert '第4页 10+3位' in by_tag['-X9'].remark
    assert '第17页 16位、第18页 16位，合计32位' in by_tag['-X20'].remark
    assert {e.page for e in by_tag['-X20'].evidences if e.role == 'terminal_position'} == {17, 18}
    assert by_tag['-XT2'].quantity is None
    assert {(r.tag, r.issue_type) for r in reviews} == {
        ('-X8', 'TERMINAL_NUMBER_REUSED'), ('-XT2', 'TERMINAL_POSITIONS_AMBIGUOUS')}


def test_mixed_terminal_models_create_distinct_material_rows():
    pages = [PageInfo(1, 'TERMINAL_DIAGRAM', '端子图', width=900, height=700)]
    spans = []
    row(spans, '-X2', [1, 2, 3], 100, 100, 'PT 6')
    row(spans, '-X2', [4], 100, 200, 'PT 6-PE')
    item = BOMItem('x2', 1, '-X2', entity_role='terminal', quantity=None)
    items = [item]
    reviews = []
    apply_terminal_positions(items, reviews, terminal_positions(pages, {1: spans}))
    assert [(it.tag, it.name, it.model, it.quantity, it.unit) for it in items] == [
        ('-X2:1~3', '接线端子', 'PT6', 3, '个'),
        ('-X2:4', '接地端子', 'PT6-PE', 1, '个')]
    assert not any(r.issue_type == 'TERMINAL_POSITIONS_AMBIGUOUS' for r in reviews)
    assert [[e.raw_text for e in it.evidences if e.role == 'terminal_position'] for it in items] == [
        ['1', '2', '3'], ['4']]
    assert len(build_aggregated_from_itemized(items)) == 2
    result = ExtractionResult('test', 'test.pdf', '', '', 'SCHEMATIC_MODE', 1, itemized_bom=items)
    assert [it['tag_members'] for it in result.to_dict()['itemized_bom']] == [['-X2'], ['-X2']]
    assert [_keys(it) for it in items] == [['-X2'], ['-X2']]


def test_bare_terminal_header_claims_its_own_numbered_symbols():
    page = PageInfo(5, 'CONTROL_CIRCUIT', '控制', width=900, height=700)
    spans = [span('-X11', 20, 100), span('-X12', 20, 200), span('-X14', 490, 200),
             span('X13', 490, 100), span('X99', 700, 100)]
    symbols = []
    for start, numbers in ((48, range(1, 3)), (512, range(1, 12))):
        for index, number in enumerate(numbers):
            x = start + index * 16
            spans.extend([span(str(number), x + 5, 100, width=10 if number >= 10 else 8),
                          span('PT2.5', x, 84, width=5, height=16)])
            symbols.append({'cx':x + 2, 'cy':104, 'bbox':[x, 102, x + 4, 106],
                            'norm_bbox':[x / 900, 102 / 700, (x + 4) / 900, 106 / 700]})
    found = terminal_positions([page], {5:spans}, {5:symbols})
    assert len(found['-X11']) == 2
    assert len(found['-X13']) == 11
    assert '-X99' not in found
    items = [BOMItem('x11', 1, '-X11', entity_role='terminal', quantity=None)]
    reviews = []
    apply_terminal_positions(items, reviews, found)
    by_strip = {item.terminal_strip_tag:item for item in items}
    assert (by_strip['-X11'].quantity, by_strip['-X13'].quantity) == (2, 11)
    assert by_strip['-X13'].model == 'PT2.5'
    assert len([e for e in by_strip['-X13'].evidences if e.role == 'terminal_symbol']) == 11
    assert by_strip['-X13'].needs_review
    assert any(review.tag == '-X13' and review.issue_type == 'TERMINAL_TAG_NORMALIZED' for review in reviews)
    assert any(review.tag == '-X13' and '-X12' in review.description and '-X14' in review.description
               for review in reviews)


def test_repeated_bare_header_accepts_numbered_continuation_segment():
    page = PageInfo(3, 'MAIN_CIRCUIT', '主电路', width=900, height=700)
    spans = [span('X7', 100, 100), span('X7', 100, 200)]
    symbols = []
    for y, values in ((100, range(1, 4)), (200, range(4, 8))):
        for index, number in enumerate(values):
            x = 130 + index * 24
            spans.extend([span(str(number), x + 5, y), span('PT2.5', x, y - 13)])
            symbols.append({'cx':x + 2, 'cy':y + 4, 'bbox':[x, y + 2, x + 4, y + 6],
                            'norm_bbox':[x / 900, (y + 2) / 700, (x + 4) / 900, (y + 6) / 700]})
    found = terminal_positions([page], {3:spans}, {3:symbols})
    assert sorted(entry['number'] for entry in found['-X7']) == list(range(1, 8))
    items, reviews = [], []
    apply_terminal_positions(items, reviews, found)
    row, = items
    assert (row.tag, row.quantity, row.terminal_numbers) == ('-X7:1~7', 7, list(range(1, 8)))
    assert '3+4' in row.remark
    assert sum(e.role == 'terminal_symbol' for e in row.evidences) == 7


def test_long_vertical_gap_keeps_continuing_positions_on_same_strip():
    page = PageInfo(14, 'TERMINAL_DIAGRAM', '端子图', width=900, height=700)
    spans = []
    column(spans, '-X18', range(1, 10), 300, 60, 'PT2.5')
    for index, number in enumerate(range(10, 16)):
        py = 360 + index * 20
        spans.extend([span(str(number), 300, py), span('PT2.5', 300, py - 10)])
    column(spans, '-XT4', range(1, 16), 500, 60, 'PT2.5')
    found = terminal_positions([page], {14: spans})
    assert [entry['number'] for entry in found['-X18']] == list(range(1, 16))
    assert len(found['-XT4']) == 15
    items = [BOMItem('x18', 1, '-X18', entity_role='terminal', quantity=None)]
    apply_terminal_positions(items, [], found)
    assert items[0].quantity == 15
    assert '第14页 9+6位' in items[0].remark


def test_diagonal_terminal_row_uses_its_own_axis():
    page = PageInfo(1, 'TERMINAL_DIAGRAM', '端子图', width=900, height=700)
    spans = [span('-X7', 100, 100)]
    for number in range(1, 4):
        x = 120 + number * 20
        y = 120 + number * 20
        spans.extend([span(str(number), x, y), span('PT4', x + 6, y - 10)])
    found = terminal_positions([page], {1: spans})
    assert [entry['number'] for entry in found['-X7']] == [1, 2, 3]


def test_vector_symbol_required_when_checked():
    document = pymupdf.open()
    page = document.new_page(width=900, height=700)
    page.draw_circle((104, 103), 3)
    symbols = vector_terminal_symbols(page)
    assert len(symbols) == 1
    spans = []
    row(spans, '-X1', [1, 2], 100, 100, 'PT2.5')
    info = PageInfo(1, 'TERMINAL_DIAGRAM', '端子图', width=900, height=700)
    found = terminal_positions([info], {1: spans}, {1: symbols})
    item = BOMItem('x1', 1, '-X1', entity_role='terminal', quantity=None)
    reviews = []
    apply_terminal_positions([item], reviews, found)
    assert item.quantity is None
    assert item.quantity_candidates == [2]
    assert any(r.issue_type == 'TERMINAL_SYMBOL_UNVERIFIED' for r in reviews)


def test_x14_inherits_single_printed_model_across_six_drawn_positions():
    document = pymupdf.open()
    pdf_page = document.new_page(width=900, height=700)
    spans = [span('-X14', 70, 100), span('1 PT 2,5', 98, 100, width=24),
             span('2', 140, 100), span('3', 350, 100), span('4', 400, 100),
             span('-X14', 330, 300), span('5', 350, 300), span('6', 400, 300)]
    for x, y in [(103, 103), (144, 103), (354, 103), (404, 103), (354, 303), (404, 303)]:
        pdf_page.draw_circle((x, y), 3)
    page = PageInfo(7, 'CONTROL_CIRCUIT', '控制回路', width=900, height=700)
    found = terminal_positions([page], {7: spans}, {7: vector_terminal_symbols(pdf_page)})
    assert sorted(position['number'] for position in found['-X14']) == list(range(1, 7))
    assert [position['model_source'] for position in found['-X14']].count('same_strip_continuity') == 5
    items = [BOMItem('x14', 1, '-X14', entity_role='terminal', quantity=None)]
    reviews = []
    apply_terminal_positions(items, reviews, found)
    row = items[0]
    assert (row.tag, row.model, row.quantity, row.terminal_numbers) == ('-X14:1~6', 'PT2.5', 6, list(range(1, 7)))
    assert row.source_type == SourceType.TOPOLOGY_INFERRED.value
    assert row.needs_review
    assert len([e for e in row.evidences if e.role == 'terminal_symbol']) == 6
    assert len([e for e in row.evidences if e.role == 'model_text']) == 1
    assert len([e for e in row.evidences if e.role == 'model_inference']) == 5
    assert any(review.issue_type == 'TERMINAL_MODEL_INFERRED' for review in reviews)
    explicit_row = BOMItem('other', 2, '-X10:1~2', name='接线端子', model='PT2.5', raw_model='PT2.5',
                           quantity=2, entity_role='terminal', category='auxiliary',
                           quantity_status='by_positions', quantity_basis='drawn_terminal_positions')
    assert len(build_aggregated_from_itemized([row, explicit_row])) == 2


def test_unlabelled_positions_do_not_inherit_when_strip_has_two_models():
    page = PageInfo(1, 'TERMINAL_DIAGRAM', '端子图', width=900, height=700)
    spans = [span('-X2', 70, 100), span('1', 100, 100), span('PT6', 100, 82),
             span('2', 140, 100), span('3', 180, 100), span('PT6-PE', 180, 82)]
    symbols = [{'cx':x, 'cy':103, 'bbox':[x-3,100,x+3,106],
                'norm_bbox':[(x-3)/900,100/700,(x+3)/900,106/700]} for x in (104,144,184)]
    found = terminal_positions([page], {1:spans}, {1:symbols})
    assert [(p['number'],p['model']) for p in found['-X2']] == [(1,'PT6'),(2,''),(3,'PT6-PE')]
    items = [BOMItem('x2', 1, '-X2', entity_role='terminal', quantity=None)]
    reviews = []
    apply_terminal_positions(items, reviews, found)
    assert any(row.quantity is None and row.terminal_numbers == [2] for row in items)
    assert any(review.issue_type == 'TERMINAL_MODEL_UNBOUND' for review in reviews)
