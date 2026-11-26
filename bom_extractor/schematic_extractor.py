from __future__ import annotations
import math
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pymupdf

from .classifier import extract_visual_spans
from .cad_import import cad_join_tag_fragments
from .cad_rack import read_rack
from .component_names import owned_name, printed_device_caption, resistor_parameters
from .models import BOMItem, ReviewItem, EvidenceLocation, SourceType, PageInfo, PageType
from .vision import VisionRecognizer
from .ocr_engine import DualChannelOCREngine


from .notation import (BREAKER_FAMILIES, COMPONENT_TAG_RE, MODEL_TOKEN_RE, POWER_CURRENT_LABEL, tagged_model_series, with_component_anchors, device_code, local_tag, qualify_tag,
                       is_connection_tag, is_field_load, model_compatible, model_family)
from .spatial_binding import bind_models, device_captions, stacked_device_annotations
from .procurement import range_annotations, flag_template_quantities, suggest
from .terminal_positions import terminal_positions, apply_terminal_positions, vector_terminal_symbols
from .reconciliation import norm

IGNORE_TAG_PREFIXES = ("-X", "=+-X", "-W", "-PE", "-M", "-YV", "DT", "-DT", "SP", "-SP")


def _dist(s1: Dict[str, Any], s2: Dict[str, Any]) -> float:
    return math.hypot(s1["cx"] - s2["cx"], s1["cy"] - s2["cy"])


def _reference_gap(anchor_box, marker_box):
    gap = marker_box[1] - anchor_box[3]
    height = anchor_box[3] - anchor_box[1]
    aligned = min(anchor_box[2], marker_box[2]) >= max(anchor_box[0], marker_box[0])
    return gap if height > 0 and aligned and -.3 * height <= gap <= 2.0 * height else None


def _is_model_compatible_with_tag(tag: str, model_text: str) -> bool:
    return model_compatible(tag, model_text)


def _expand_norm_bbox(norm_bbox: List[float], pad_x: float = 0.02, pad_y: float = 0.02) -> List[float]:
    return [
        round(max(0.0, norm_bbox[0] - pad_x), 4),
        round(max(0.0, norm_bbox[1] - pad_y), 4),
        round(min(1.0, norm_bbox[2] + pad_x), 4),
        round(min(1.0, norm_bbox[3] + pad_y), 4),
    ]


def _normalize_tag(raw_tag: str) -> str:
    t = raw_tag.strip()
    if t.upper().startswith(("HL", "FR", "SQ")):
        return t.upper()
    if not t.startswith(("-", "=", "+")):
        return f"-{t.upper()}"
    return t.upper() if t.startswith("-") else t


def _preferred_device_caption(labels: List[str]) -> str:
    """Resolve a short caption and a repeated-character plot/OCR variant."""
    if not labels:
        return ''
    def clean(value):
        return re.sub(r'([\u4e00-\u9fff])\1+', r'\1', value)
    canonical = max((clean(label) for label in labels), key=len)
    if not all(clean(label) in canonical for label in labels):
        return ''
    return next((label for label in labels if label == canonical), canonical)


def _infer_component_name_and_brand(tag: str, model: str, nearby_texts: List[str]) -> Tuple[str, str]:
    t = device_code(tag)
    t = {'NFB':'QF','MCCB':'QF','MCB':'QF','MC':'KM','OCR':'FR','TH':'FR','INV':'TB','CR':'KA'}.get(t,t)
    m = model.upper()
    near_str = " ".join(nearby_texts)

    # 1. Infer brand from model prefix (Industrial Brand Gene Dictionary)
    brand = ""
    if m.startswith("ACS"):
        return "变频器", "ABB"
    if m.startswith("EDS-") or "MOXA" in m:
        return "工业以太网交换机", "MOXA"
    if m.startswith("FBOX-"):
        return "远程通信模块", "FLEXEM" if "FLEXEM" in near_str.upper() else "待定"
    if m.startswith(("SMART", "6AV")):
        return "触摸屏", "西门子"
    if m.startswith("6ES7"):
        return ("PLC CPU" if "CPU" in near_str.upper() or "1ST" in m else "PLC 模块"), "西门子"
    if m.startswith("RM22"):
        return "相序保护继电器", "施耐德"
    if m.startswith("AMC"):
        return "多功能电表", "待定"
    if m.startswith("SXC-"):
        return "脉冲控制器", "待定"
    if any(k in m for k in ("LC1", "EZD", "NSX", "CVS", "GV2", "LRD", "IC65", "C65", "XB2", "ZB2", "RXM", "RXZE", "REXL", "ATV", "EA9")):
        brand = "施耐德"
    elif any(k in m for k in ("6ES7", "6EP", "6AV", "3RT", "3RV", "5SU", "5SY", "SIEMENS")):
        brand = "西门子"
    elif any(k in m for k in ("HDR-", "NDR-", "EDR-", "LRS-", "MW")):
        brand = "明纬"
    elif any(k in m for k in ("ES680", "MD500", "RXLG", "MEG", "INOVANCE")):
        brand = "汇川"
    elif "NHR-" in m:
        brand = "虹润"
    elif "LF60" in m:
        brand = "PAKU"
    elif any(k in m for k in ("UK-", "QUINT", "CLIPLINE", "PHOENIX")):
        brand = "菲尼克斯"
    elif any(k in m for k in ("CHNT", "NB1", "NXB", "CJX2", "BH-0.66", "SG-")):
        brand = "正泰"

    # 2. Infer standard component name
    if tagged_model_series(tag, model) and any(POWER_CURRENT_LABEL.fullmatch(text.strip()) for text in nearby_texts):
        return "功率设备（型式待核）", brand
    if t == "HMI":
        return "人机界面", brand
    if t == "KPH":
        return "相序保护继电器", brand
    if t == "TB":
        return "变频器/驱动器", brand
    if t == "FD":
        return "阀门/执行机构", brand
    if t == "BP":
        return "传感器/变送器", brand
    if t == "M":
        return "电机/现场负载", brand
    if t in ("Y", "YV"):
        return "电磁阀", brand
    if t == "P":
        return "仪表", brand
    if t == "E":
        return "电气元器件（类型待核）", brand
    if t == 'L':
        return ('电流互感器' if any('S1' in text or 'S2' in text for text in nearby_texts) else '电感/互感器（待核实）'), brand
    if t.startswith('X'):
        return ('调试插座' if 'POL' in near_str.upper() else '端子排/分配端子'), brand
    if t == 'W':
        return '电缆', brand
    if t == "S":
        return "急停按钮" if any("急停" in tok for tok in nearby_texts[:4]) else "开关/按钮", brand
    if t == "Q" and any("选择开关" in tok for tok in nearby_texts[:4]):
        return "转换开关", brand
    if t == "G":
        return "开关电源", brand
    if t.startswith(("LI", "TI", "PI", "FI")):
        name = "显示仪表"
    elif t.startswith("FAN") or "风扇" in near_str:
        name = "风扇+过滤网"
    elif t.startswith("QF") or t.startswith("QS") or t == "Q" or t.startswith("F"):
        if t == 'QS':
            name = "隔离开关"
        elif device_code(tag) == 'QF' and model and not model.startswith(('未提取', '图纸未注')) and not (
            m.startswith(BREAKER_FAMILIES) or re.search(r'\b(?:MCCB|MCB|BREAKER)\b',m)
        ):
            name = "开关器件（型式待核）"
        elif "GV2" in m or "马达" in near_str:
            name = "马达保护断路器"
        elif t.startswith("FU"):
            name = "熔断器"
        elif t.startswith("FR"):
            name = "热继电器"
        else:
            name = "断路器"
    elif t.startswith("KM"):
        name = "交流接触器"
    elif t.startswith("FR"):
        name = "热继电器"
    elif t.startswith("KT"):
        name = "时间继电器"
    elif t.startswith("KA") or t.startswith("K"):
        name = "中间继电器"
    elif t.startswith("HL") or t.startswith("H") or t.startswith("WL"):
        if "B1" in m or "白" in near_str[:6]:
            name = "白色指示灯"
        elif "M3" in m or "绿" in near_str[:6]:
            name = "绿色指示灯"
        elif "M4" in m or "红" in near_str[:6]:
            name = "红色指示灯"
        elif "M5" in m or "黄" in near_str[:6]:
            name = "黄色指示灯"
        else:
            name = "指示灯"
    elif t.startswith("SB"):
        name = "启动/控制按钮"
    elif t.startswith("SA"):
        name = "旋转开关"
    elif t.startswith("ES"):
        name = "急停按钮"
    elif t.startswith("BZ"):
        name = "蜂鸣器"
    elif t.startswith("V") or t.startswith("PS") or "电源" in near_str:
        name = "开关电源"
    elif t.startswith(("TC", "TM", "T")) and not t.startswith("TI"):
        name = "控制变压器"
    elif t.startswith("SQ"):
        name = "行程开关"
    else:
        name = "电气元器件"

    return name, brand


def extract_from_schematic_pdf(doc, pages_info, rules_config, crops_dir=None, inventory=None):
    """Extract devices by scoped IEC identity and bind model text using drawn geometry."""
    vision=VisionRecognizer(crops_dir=crops_dir)
    page_spans_map={}
    terminal_spans_map={}
    terminal_symbols_map={}
    observations={}
    bindings={}
    descriptions={}
    annotation_stacks={}
    cross_references={}
    vector_findings=[]
    cabinet_modules=[]
    review_items=[]
    title_numbers={}
    page_grids={}
    for info in pages_info:
        page=doc[info.page_num-1]
        terminal_symbols_map[info.page_num]=vector_terminal_symbols(page)
        native=extract_visual_spans(page)
        grid_labels={int(span['text']):span['cx'] for span in native
                     if re.fullmatch(r'[0-9]',span['text'].strip()) and span['cy'] < .04 * info.height}
        if len(grid_labels) >= 8:
            page_grids[info.page_num]=grid_labels
            info.extraction_diagnostics['drawing_grid_columns'] = [
                {'number':number, 'x':round(cx / info.width, 5)}
                for number,cx in sorted(grid_labels.items())]
        spans=DualChannelOCREngine.extract_dual_channel_spans(page,native,dpi=200)
        body=[sp for sp in spans if .02*info.height<sp['cy']<.90*info.height and .02*info.width<sp['cx']<.98*info.width]
        # Terminal numbers can sit inside the right drawing margin (X9:10).
        terminal_spans_map[info.page_num]=with_component_anchors(cad_join_tag_fragments([
            sp for sp in spans if .02*info.height<sp['cy']<.90*info.height
            and .005*info.width<sp['cx']<.995*info.width]))
        # Native CAD often omits the IEC leading hyphen. Keep the literal label
        # as evidence; canonicalize only the temporary binding input.
        body=with_component_anchors(cad_join_tag_fragments(body))
        page_spans_map[info.page_num]=body
        if info.page_type in (PageType.COVER.value,PageType.CONTENTS.value):
            info.extraction_diagnostics={**info.extraction_diagnostics,'native_characters':sum(len(sp['text']) for sp in native),'device_candidates':0,'model_candidates':0,'unbound_models':0,'excluded_connections':0}
            continue
        anchors=[]
        connections=[]
        for span in body:
            raw=span['text'].strip()
            if raw.startswith('-') and is_connection_tag(raw):
                connections.append(raw)
                continue
            if not COMPONENT_TAG_RE.fullmatch(raw):
                continue
            tag=qualify_tag(_normalize_tag(raw), info.location_code, info.function_code)
            anchor={**span,'tag':tag,'raw_tag':raw}
            anchors.append(anchor)
            near=sorted((o for o in body if o is not span and _dist(span,o)<75),key=lambda o:_dist(span,o))
            coil=any(o['text'].upper() in ('A1','A2') for o in near)
            role='coil' if coil else 'main_circuit' if info.page_type==PageType.MAIN_CIRCUIT.value else 'contact_or_symbol'
            evidence=EvidenceLocation(info.page_num,span['bbox'],_expand_norm_bbox(span['norm_bbox'],.012,.015),raw_text=span.get('source_text',raw),role=role,device_tag=tag)
            observations.setdefault(tag,[]).append({'evidence':evidence,'anchor':anchor,'near':[o['text'] for o in near],'location':info.location_code})
            # An EPLAN /sheet.column marker immediately under a symbol points to
            # another representation of this device. It is evidence, not a part.
            for other in near:
                reference = re.fullmatch(r'/([1-9]\d*)\.([0-9])', other['text'].strip())
                if not reference:
                    continue
                # Contact terminals can sit between the tag and its sheet/column
                # marker; the old 1.5-line limit missed e.g. -KA14 /17.1.
                if _reference_gap(span['bbox'], other['bbox']) is not None:
                    cross_references.setdefault(tag, []).append((info.page_num, other, int(reference[1]), int(reference[2])))
        modules,reserved_handles=read_rack(page,body,info)
        cabinet_modules.extend(modules)
        binding_spans=[s for s in body if s.get('handle') not in reserved_handles] if reserved_handles else body
        bound,models,unbound,enclosures=bind_models(page,binding_spans,anchors)
        # A code printed in the same title block across sheets is traceability
        # evidence. Retain its position and avoid creating a purchasing task for it.
        for candidate in unbound:
            text=candidate['model_text']
            if not re.fullmatch(r'[A-Z]{2,}\d{3,}',text,re.I):
                continue
            x,y=candidate['cx']/info.width,candidate['cy']/info.height
            if min(x,y,1-x,1-y)>.18:
                continue
            nearby={s['text'].replace(' ','') for s in body
                    if abs(s['cx']/info.width-x)<.12 and abs(s['cy']/info.height-y)<.12}
            has_number='图号' in nearby or {'图','号'}<=nearby
            has_customer='客户' in nearby or {'客','户'}<=nearby
            if has_number and has_customer:
                title_numbers.setdefault(text,[]).append((info.page_num,x,y))
        for tag, labels in device_captions(body, anchors, enclosures).items():
            descriptions.setdefault(tag, []).extend((info.page_num, sp) for sp in labels)
        for tag, labels in stacked_device_annotations(body, anchors).items():
            annotation_stacks.setdefault(tag, []).extend((info.page_num, sp) for sp in labels)
        for tag,matches in bound.items():
            bindings.setdefault(tag,[]).extend({**match,'page':info.page_num} for match in matches)
        info.extraction_diagnostics={**info.extraction_diagnostics,'range_annotations':range_annotations(body, info.title),'native_characters':sum(len(sp['text']) for sp in native),
            'device_candidates':len({a['tag'] for a in anchors}),'device_tags':sorted({a['tag'] for a in anchors}),'model_candidates':len(models),
            'unbound_models':len(unbound),'excluded_connections':len(set(connections))}
        info.extraction_diagnostics['unbound_evidence']=[{'text':m['model_text'],'norm_bbox':m['norm_bbox']} for m in unbound]
        info.extraction_diagnostics['rack_slots']=len(modules)
        findings=vision.analyze_vector_regions_on_page(page,native)
        for finding in findings:
            finding=dict(finding)
            if finding.get('tag'):
                finding['tag']=','.join(qualify_tag(_normalize_tag(t.strip()),info.location_code,info.function_code) for t in finding['tag'].split(','))
            vector_findings.append(finding)

    for info in pages_info:
        diag=info.extraction_diagnostics
        locations=[]
        for evidence in diag.get('unbound_evidence',[]):
            x0,y0,x1,y1=evidence['norm_bbox']
            repeats={p for p,x,y in title_numbers.get(evidence['text'],[]) if abs(x-(x0+x1)/2)<.015 and abs(y-(y0+y1)/2)<.015}
            if info.page_num in repeats and len(repeats)>=3:
                diag.setdefault('title_block_evidence',[]).append(evidence)
            else:
                locations.append(evidence)
        diag['unbound_evidence']=locations
        diag['unbound_models']=len(locations)
        if locations:
            distinct=list(dict.fromkeys(m['text'] for m in locations))
            signature=hashlib.sha256(str(locations).encode()).hexdigest()[:12]
            examples='、'.join(distinct[:3])
            review_items.append(ReviewItem(id=f'rev_unbound_p{info.page_num}_{signature}',severity='WARNING',issue_type='UNBOUND_MODEL',
                tag='',title=f'第 {info.page_num} 页核对物料归属 · {examples}',
                description=f'{len(locations)} 处原图文字尚未对应到设备，共 {len(distinct)} 种写法。请对照图上器件检查是否漏列采购件。',
                suggestion='查看本页原图和文字位置，补录遗漏的实物或型号。',page=info.page_num,norm_bbox=locations[0]['norm_bbox']))
        elif not diag.get('device_candidates') and not diag.get('rack_slots') and info.page_type in (PageType.PLC_WIRING.value,PageType.MAIN_CIRCUIT.value,PageType.CONTROL_CIRCUIT.value):
            review_items.append(ReviewItem(id=f'rev_coverage_p{info.page_num}',severity='WARNING',issue_type='PAGE_COVERAGE',
                tag='',title=f'第 {info.page_num} 页尚未定位设备',description='识别到了回路或 PLC 接线，但未建立设备明细。',
                suggestion='核对本页是否有遗漏的设备，可使用本页 AI 识别。',page=info.page_num,norm_bbox=[.02,.02,.98,.90]))
    # Resolve unscoped spelling variants before constructing any BOM rows.
    # A scoped tag such as +B-FR1 identifies a separate device.
    for alias in list(observations):
        canonical = alias[1:] if re.fullmatch(r'-FR\d+', alias) else 'SQ' if alias == 'SQ-1' else None
        if canonical not in observations:
            continue
        occurrences = observations.pop(alias)
        for entry in occurrences:
            entry['anchor']['tag'] = canonical
            entry['evidence'].device_tag = canonical
        observations[canonical].extend(occurrences)
        observations[canonical].sort(key=lambda entry: entry['evidence'].page)
        cross_references.setdefault(canonical, []).extend(cross_references.pop(alias, []))
        bindings.setdefault(canonical, []).extend(bindings.pop(alias, []))
        descriptions.setdefault(canonical, []).extend(descriptions.pop(alias, []))
        annotation_stacks.setdefault(canonical, []).extend(annotation_stacks.pop(alias, []))
        for finding in vector_findings:
            if finding.get('tag') == alias:
                finding['tag'] = canonical
    vector_by_tag={f['tag']:f for f in vector_findings if f.get('type')=='VECTOR_TEXT_MODEL'}
    for candidate in (inventory or {}).get('devices', []):
        tag=candidate['tag']
        if tag in observations or not candidate['occurrences']:
            continue
        first=candidate['occurrences'][0]
        review_items.append(ReviewItem(
            id=f'rev_inventory_{tag}',severity='WARNING',issue_type='INVENTORY_UNBOUND',
            tag=tag,title=f'全图位号尚未建立设备：{tag}',
            description='位号盘点找到了该标记，设备符号或采购身份尚未建立。',
            suggestion='核对图上位置，确认是独立设备、触点还是连接标记。',
            page=first['page'],norm_bbox=first['bbox']))
    page_by_number={info.page_num:info for info in pages_info}
    sheets={}
    for info in pages_info:
        if info.sheet_no.isdigit():
            sheets.setdefault(int(info.sheet_no), []).append(info.page_num)
    items=[]
    panel=rules_config.get('panel_series_rules',{})
    use_defaults=bool(panel.get('enable_automatic_series_selection',False))
    placeholder='未提取到型号(待核实)'
    for tag,occurrences in sorted(observations.items(),key=lambda pair:(pair[1][0]['evidence'].page,pair[0])):
        lt=local_tag(tag)
        models = []
        load_references = []
        scope = tag.removesuffix(lt)
        for match in bindings.get(tag, []):
            referenced_tag = scope + '-' + match['model_text'].upper()
            if (match['association'] == 'spatial' and referenced_tag in observations
                    and referenced_tag != tag and is_field_load(referenced_tag)):
                load_references.append(EvidenceLocation(
                    match['page'], match['bbox'], match['norm_bbox'], raw_text=match['text'],
                    role='field_load_reference', device_tag=referenced_tag))
            else:
                models.append(match)
        concrete=[m for m in models if not model_family(m['model_text'])]
        chosen=concrete or models
        # A nearby label may belong to an adjacent device. Prefer a code printed
        # on the tag or its own symbol. Equally supported codes require selection.
        candidates={norm(m['model_text']):m['model_text'] for m in chosen}
        owned={norm(m['model_text']) for m in chosen if m['association'] in ('inline','enclosure','cad_label','label_stack')}
        possible=owned or set(candidates)
        selected=candidates[next(iter(possible))] if len(possible)==1 else ''
        distinct=list(candidates.values())
        evidences=[entry['evidence'] for entry in occurrences] + load_references
        for source_page, marker, sheet_number, column in cross_references.get(tag, []):
            target_pages=sheets.get(sheet_number, [])
            matches=[entry for entry in occurrences if entry['evidence'].page in target_pages
                     and (column not in page_grids.get(entry['evidence'].page, {})
                          or abs(entry['anchor']['cx'] - page_grids[entry['evidence'].page][column])
                          <= .1 * page_by_number[entry['evidence'].page].width)]
            column_verified=any(column in page_grids.get(page, {}) for page in target_pages)
            evidences.append(EvidenceLocation(
                source_page, marker['bbox'], marker['norm_bbox'], raw_text=marker['text'],
                role='cross_reference', device_tag=tag,
                cad_refs=[{'sheet':sheet_number,'column':column,'target_pages':target_pages,
                           'resolved':bool(matches),'column_verified':column_verified}]))
            if target_pages and not matches:
                review_items.append(ReviewItem(
                    id=f'rev_crossref_{tag}_{source_page}_{sheet_number}_{column}',severity='WARNING',
                    issue_type='CROSS_REFERENCE_MISMATCH',tag=tag,title=f'跨页引用未对应到同位号：{tag}',
                    description=f'第 {source_page} 页标注 {marker["text"]}，目标栏位附近未找到同一位号。',
                    suggestion='核对引用页、栏位和目标位号。',page=source_page,norm_bbox=marker['norm_bbox'],
                    related_page=target_pages[0]))
        near=list(dict.fromkeys(t for entry in occurrences for t in entry['near']))
        near.extend(m["text"] for m in models)
        model=selected
        source=SourceType.EXPLICIT.value if model else SourceType.UNRESOLVED.value
        needs_review=not model or len(distinct)>1 or any(m.get('needs_review') for m in chosen) or bool(model and model_family(model))
        note=''
        if len(distinct)>1:
            note=('同位号另有不同型号标注，请核实主件和附件。' if model else
                  '同位号有多个型号候选，尚未确定本次采购型号。')
            review_items.append(ReviewItem(id=f'rev_conflict_{tag}',severity='CONFLICT',issue_type='PARAM_MISMATCH',tag=tag,
                title=f'同位号检出不同型号：{tag}',description=' / '.join(distinct),suggestion='请核实主件、附件及不同页标注是否一致',
                page=evidences[0].page,norm_bbox=evidences[0].norm_bbox))
        for match in chosen:
            evidences.append(EvidenceLocation(match['page'],match['bbox'],match['norm_bbox'],raw_text=match['text'],
                                              role='model_text' if norm(match['model_text']) in possible else 'model_candidate',
                                              device_tag=tag,model_binding=match['association']))
        if not distinct and tag in vector_by_tag:
            finding=vector_by_tag[tag]
            model=finding['model']
            source=SourceType.VISION_RECOGNIZED.value
            needs_review=True
            note=finding.get('note','局部 OCR 识别，待核实')
        proposed_model=''
        if not distinct and not model and use_defaults:
            code=device_code(tag)
            if code=='KA':
                proposed_model=panel.get('intermediate_relay',{}).get('model','')
            elif code in ('ES','SB','BZ'):
                key={'ES':'emergency_stop','SB':'push_button','BZ':'buzzer'}[code]
                proposed_model=panel.get('buttons_and_switches',{}).get(key,{}).get('model','')
            elif code=='HL':
                for color,cfg in panel.get('indicator_lights',{}).items():
                    if color in near:
                        proposed_model=cfg.get('model','');break
        if not model:
            model=placeholder
            if not distinct:
                note='系统尚未可靠提取到型号；不代表原图没有标注，请核实原图或补充选型。'
                review_items.append(ReviewItem(id=f'rev_unresolved_{tag}',severity='WARNING',issue_type='UNRESOLVED_MODEL',tag=tag,
                    title=f'位号已识别，型号待核实：{tag}',description=note,suggestion='核对型号文字和关联位置，不能把未提取等同于图纸未注。',
                    page=evidences[0].page,norm_bbox=evidences[0].norm_bbox))
        elif needs_review and len(distinct)<2:
            review_items.append(ReviewItem(id=f'rev_association_{tag}',severity='WARNING',issue_type='MODEL_ASSOCIATION',tag=tag,
                title=f'型号关联/选型待确认：{tag}',description=note or f'识别到 {model}，请核实其与位号的对应关系及订货规格。',
                suggestion='根据型号文字证据定位核实',page=evidences[0].page,norm_bbox=evidences[0].norm_bbox))
        name,brand=_infer_component_name_and_brand(tag,model,near)
        labels = descriptions.get(tag, [])
        explicit_names = {"漏电保护断路器" if re.fullmatch(r'(?:[A-Z]+\+)?RCBO', span['text'], re.I)
                          else printed_device_caption(span['text']) or span['text'] for _,span in labels}
        preferred_name = _preferred_device_caption(list(explicit_names))
        if preferred_name:
            name = preferred_name
            for page_num, span in labels:
                evidences.append(EvidenceLocation(page_num, span['bbox'], span['norm_bbox'],
                                                  raw_text=span['text'], role='device_description', device_tag=tag))
        for page_num, span in annotation_stacks.get(tag, []):
            text=span['text'].strip()
            if not any(ev.page==page_num and ev.raw_text.strip()==text
                       and ev.bbox==span['bbox'] for ev in evidences):
                evidences.append(EvidenceLocation(page_num, span['bbox'], span['norm_bbox'],
                                                  raw_text=text, role='device_annotation', device_tag=tag))
        owned = [ev.raw_text for ev in evidences if ev.role in ('device_description', 'device_annotation')
                 and ev.device_tag == tag]
        printed_name = owned_name(tag, owned)
        if printed_name:
            name = printed_name
        if name.endswith('（类型待核）'):
            needs_review = True
            note = (note + ' 器件类型尚未由明确名称确认，请核对原图。').strip()
        if name == "开关器件（型式待核）":
            needs_review = True
            note = "已读到开关型号，原图文字未明确具体开关型式，按订货号核实。"
        if name == "功率设备（型式待核）":
            needs_review = True
            note = "图纸按型号系列和实例序号标识并给出功率电流，具体设备型式需核对。"
        suggestions=[]
        for rule in rules_config.get('breaker_normalization_rules',[]):
            if rule.get('model') and re.fullmatch(rule['pattern'],model,re.I):
                suggestions.append((rule['model'],rule.get('reason','企业型号选型建议')))
        pages=sorted({ev.page for ev in evidences})
        items.append(BOMItem(id=f'sch_{len(items)+1}',seq_no=len(items)+1,tag=tag,tags=[tag],local_tag=lt,
            location_code=occurrences[0]['location'],placement=occurrences[0]['location'],name=name,brand=brand or '待定',
            model=model,raw_model=selected,quantity=1,unit='个',
            primary_page=evidences[0].page,primary_bbox=evidences[0].norm_bbox,evidences=evidences,
            pdf_page_label=', '.join(f'第{p}页' for p in pages),source_type=source,
            category='field_load' if is_field_load(tag) else 'primary',needs_review=needs_review,review_note=note))
        item=items[-1]
        if len(distinct) == 1 and concrete and name != '电气元器件' and any(
            match['association'] == 'enclosure' for match in chosen
        ):
            item.procurement_name = name
        if any(entry['anchor'].get('unindexed_tag') for entry in occurrences):
            item.quantity=None;item.quantity_status='unknown';item.quantity_basis='unknown'
            item.review_note+=' 原图位号未带数字编号，采购数量保留待核。'
        code=device_code(tag)
        item.procurement_scope='field' if is_field_load(tag) else 'unknown'
        # Preserve literal parameters, owned by the closest printed tag, separately from a model.
        parameters=[]
        parameter_pattern=re.compile(r'^(?:(?:AC|DC)?\s*\d+(?:[.,]\d+)?\s*(?:A|V|KW|W|HZ)(?:\s*(?:AC|DC))?(?:(?:\s*[,，/]\s*|\s+)\d+P)?|'
                                     r'\d+P(?:\s*[,，/]\s*|\s+)\d+(?:\.\d+)?\s*A|[BCD]\d+A/\d+P|'
                                     r'\d+(?:[.,]\d+)?(?:[x×*G]\d+(?:[.,]\d+)?)+(?:\+\d+[x×*]\d+)?|.*POL.*)$',re.I)
        for match in chosen:
            if not selected or norm(match['model_text'])!=norm(selected):
                continue
            # CAD and PDF both print e.g. a contactor series and its current in one text entity.
            suffix=re.fullmatch(re.escape(match['model_text'])+r'\s+[-–]?\s*(.+)',match['text'].strip(),re.I)
            if suffix and parameter_pattern.fullmatch(suffix[1]) and suffix[1] not in parameters:
                parameters.append(suffix[1])
        for entry in occurrences:
            pno=entry['evidence'].page;anchor=entry['anchor']
            for sp in page_spans_map[pno]:
                text=sp['text'].strip()
                if not (parameter_pattern.fullmatch(text) or POWER_CURRENT_LABEL.fullmatch(text)) or _dist(anchor,sp)>65:
                    continue
                other_anchors=[o['anchor'] for obs in observations.values() for o in obs if o['evidence'].page==pno]
                if any(_dist(a,sp)+2<_dist(anchor,sp) for a in other_anchors if a['tag']!=tag):
                    continue
                if sp['text'] not in parameters:
                    parameters.append(sp['text'])
                    item.evidences.append(EvidenceLocation(pno,sp['bbox'],sp['norm_bbox'],raw_text=sp['text'],role='specification',device_tag=tag))
        if code == 'R' or re.search(r'电阻器?$', name):
            for ev in list(item.evidences):
                if ev.device_tag != tag or ev.role not in ('device_description', 'device_annotation'):
                    continue
                rating = resistor_parameters(ev.raw_text)
                if rating and rating not in parameters:
                    parameters.append(rating)
                    item.evidences.append(EvidenceLocation(ev.page, ev.bbox, ev.norm_bbox,
                                                          raw_text=rating, role='specification', device_tag=tag))
        item.specification='；'.join(parameters)
        if code.startswith('X'):
            item.entity_role='device' if 'POL' in item.specification.upper() else 'terminal'
            item.category='primary' if item.entity_role=='device' else 'auxiliary'
            if item.entity_role=='terminal':
                item.unit='片';item.quantity=None;item.quantity_status='unknown';item.quantity_basis='terminal_positions_required'
            else:item.name='调试插座'
        elif code=='W':
            item.entity_role='cable';item.category='auxiliary';item.unit='米';item.quantity=None
            item.quantity_status='unknown';item.quantity_basis='length_required'
        if proposed_model:suggestions.append((proposed_model,'企业系列选型建议，未作为图纸事实'))
        for proposed,reason in suggestions:suggest(item,proposed,reason)
    repeated_slots = {}
    for module in cabinet_modules:
        repeated_slots.setdefault(module.group_identity, []).append(module)
    for group in repeated_slots.values():
        for module in group:
            if len(group)>1:
                module.quantity=None
                module.quantity_status='needs_confirmation'
                module.quantity_basis='repeated_slot'
                module.review_note+=' 同一安装范围出现相同槽号，请核实机架身份或重复图页。'
            module.seq_no=len(items)+1
            items.append(module)
    # Attach GV2 fault-contact accessories only when the associated breaker and contact both exist.
    by_tag={it.tag:it for it in items}
    remove=set()
    for item in items:
        if 'GV2ME' in item.model.upper():
            peer=item.tag.replace('QF','FR')
            if peer in by_tag:
                item.evidences.extend(by_tag[peer].evidences)
                if '+GVAE11' not in item.model.upper():
                    suggest(item,item.model+'+GVAE11','控制触点提示可能需辅助模块，请核对主件已含配置')
                # A distinct FR tag is not deleted merely because its number resembles QF.
    items=[it for it in items if it.tag not in remove]
    review_items=[r for r in review_items if r.tag not in remove]
    for finding in vector_findings:
        if finding['type']=='UNRESOLVED_VECTOR_REGION':
            review_items.append(ReviewItem(id=f"rev_vector_p{finding['page']}",severity='WARNING',issue_type='VECTOR_VISION',tag='',
                title='矢量区域识别未完成',description=finding['note'],suggestion='请框选核实',page=finding['page'],norm_bbox=finding['norm_bbox']))
        elif finding.get('tag') not in observations:
            nb=finding['norm_bbox'];pno=finding['page']
            sheet = doc[pno - 1].rect
            bbox = [nb[0]*sheet.width, nb[1]*sheet.height, nb[2]*sheet.width, nb[3]*sheet.height]
            items.append(BOMItem(id=f'ocr_{len(items)+1}',seq_no=len(items)+1,tag=finding['tag'],tags=[finding['tag']],
                local_tag=local_tag(finding['tag']),name=finding['name'],model=finding['model'],raw_model=finding['model'],
                brand=finding.get('brand','待定'),quantity=finding.get('quantity',1),primary_page=pno,primary_bbox=nb,
                evidences=[EvidenceLocation(pno,bbox,nb,raw_text=finding['model'],role='rapidocr_vector',device_tag=finding['tag'])],
                source_type=SourceType.VISION_RECOGNIZED.value,category='auxiliary' if finding['type']=='CABINET_DIMENSION' else 'primary',needs_review=True,review_note=finding.get('note','')))
    if rules_config.get('auxiliary_estimation_rules',{}).get('enable_automatic_quantity_estimation',False):
        aux,reviews=_extract_wires_and_terminals(pages_info,page_spans_map,max((it.seq_no for it in items),default=0),rules_config,vector_findings,len(items))
        items.extend(aux);review_items.extend(reviews)
    apply_terminal_positions(items, review_items, terminal_positions(
        pages_info, terminal_spans_map, terminal_symbols_map))
    flag_template_quantities(items,review_items,pages_info)
    return items,review_items


def _extract_wires_and_terminals(
    pages_info: List[PageInfo],
    page_spans_map: Dict[int, List[Dict[str, Any]]],
    start_seq: int,
    rules_config: Optional[Dict[str, Any]] = None,
    vector_findings: Optional[List[Dict[str, Any]]] = None,
    control_device_count: int = 0,
) -> Tuple[List[BOMItem], List[ReviewItem]]:
    """
    PDF-driven wire gauge (mm²) + terminal strip extractor. No per-drawing constants:
      1. Scan MAIN_CIRCUIT pages for explicit gauge annotations ('70 mm²', '16' + 'mm²', '4mm²4mm²4mm²').
      2. Scan TERMINAL_DIAGRAM pages for strip headers (=+-X1 / -X2 ...), count the pins of each strip
         (stopping at horizontal gaps so strips on the same row are not merged) and read the wire label
         printed above every pin.
      3. Resolve each pin's wire label on MAIN_CIRCUIT pages and attach the nearest gauge annotation
         (same column first, then Euclidean). Pins without a resolvable gauge are control pins.
      4. Map gauge -> terminal rating via `wire_terminal_estimation` in rules_config.json.
         Terminal estimates stay tagged ESTIMATED_AUX; wire lengths are never estimated.
    """
    cfg = (rules_config or {}).get("wire_terminal_estimation", {})
    rating_map: Dict[str, str] = cfg.get("gauge_to_terminal_rating", {})
    control_rating: str = cfg.get("control_terminal_rating", "16A")
    term_spare: float = float(cfg.get("terminal_spare_ratio", 0.0))
    incoming_poles: int = int(cfg.get("incoming_terminal_poles", 0))
    terminal_brand: str = cfg.get("terminal_brand", "")

    items: List[BOMItem] = []
    reviews: List[ReviewItem] = []
    seq = start_seq

    def _gv(g: str) -> float:
        try:
            return float(g.lower().replace("mm²", "").replace("mm2", "").strip())
        except ValueError:
            return 0.0

    def _norm_gauge(num: str) -> str:
        v = float(num)
        return f"{int(v)}mm²" if v.is_integer() else f"{v}mm²"

    # ---- 1. Gauge annotations on main-circuit pages (keep point coordinates for label matching) ----
    gauge_marks: List[Dict[str, Any]] = []  # {gauge, page, bbox, norm_bbox, raw}
    for pinfo in pages_info:
        if pinfo.page_type not in (PageType.MAIN_CIRCUIT.value,):
            continue
        spans = page_spans_map.get(pinfo.page_num, [])
        for s in spans:
            txt = s["text"].strip()
            m_all = re.findall(r"(\d+(?:\.\d+)?)\s*mm[²2]", txt, re.IGNORECASE)
            if m_all:
                for num in m_all:
                    gauge_marks.append({
                        "gauge": _norm_gauge(num), "page": pinfo.page_num, "bbox": s["bbox"],
                        "norm_bbox": _expand_norm_bbox(s["norm_bbox"], 0.015, 0.015), "raw": txt,
                    })
                continue
            if txt.lower() not in ("mm²", "mm2"):
                continue
            best, best_d = None, 22.0
            for o in spans:
                if o is s:
                    continue
                if re.match(r"^(?:0\.5|0\.75|1|1\.5|2\.5|4|6|10|16|25|35|50|70|95|120|150|185|240)$", o["text"].strip()):
                    d = _dist(s, o)
                    if d < best_d:
                        best, best_d = o, d
            if best is not None:
                bx = [min(s["bbox"][0], best["bbox"][0]), min(s["bbox"][1], best["bbox"][1]),
                      max(s["bbox"][2], best["bbox"][2]), max(s["bbox"][3], best["bbox"][3])]
                nb = [min(s["norm_bbox"][0], best["norm_bbox"][0]), min(s["norm_bbox"][1], best["norm_bbox"][1]),
                      max(s["norm_bbox"][2], best["norm_bbox"][2]), max(s["norm_bbox"][3], best["norm_bbox"][3])]
                gauge_marks.append({
                    "gauge": _norm_gauge(best["text"].strip()), "page": pinfo.page_num, "bbox": bx,
                    "norm_bbox": _expand_norm_bbox(nb, 0.015, 0.015), "raw": f"{best['text'].strip()} mm²",
                })

    # ---- 2. Terminal strips: pins + wire label above each pin ----
    strips: Dict[str, Dict[str, Any]] = {}
    for pinfo in pages_info:
        if pinfo.page_type != PageType.TERMINAL_DIAGRAM.value:
            continue
        spans = page_spans_map.get(pinfo.page_num, [])
        for hdr in spans:
            htxt = hdr["text"].strip()
            if not re.match(r"^(?:=\+?-|-)?X\d+$", htxt, re.IGNORECASE):
                continue
            name = "-" + re.search(r"X\d+", htxt.upper()).group(0)
            row = sorted(
                [o for o in spans
                 if o["text"].strip().isdigit() and 1 <= int(o["text"].strip()) <= 200
                 and -4.0 <= o["cy"] - hdr["cy"] <= 22.0 and o["cx"] > hdr["cx"]],
                key=lambda o: o["cx"],
            )
            pins: List[Dict[str, Any]] = []
            last_x = hdr["cx"]
            for o in row:
                if o["cx"] - last_x > 30.0:  # gap -> next strip on the same row
                    break
                pins.append(o)
                last_x = o["cx"]
            if not pins:
                continue
            pin_info = []
            for p in pins:
                label = None
                best_dy = 50.0
                for o in spans:
                    if o is p or o in pins:
                        continue
                    dy = p["cy"] - o["cy"]
                    if abs(o["cx"] - p["cx"]) <= 4.5 and 0 < dy <= best_dy:
                        best_dy, label = dy, o["text"].strip()
                pin_info.append({"pin": int(p["text"].strip()), "label": label})
            nbs = [hdr["norm_bbox"]] + [p["norm_bbox"] for p in pins]
            strips[name] = {
                "page": pinfo.page_num, "raw_header": htxt, "bbox": hdr["bbox"], "pins": pin_info,
                "norm_bbox": [round(min(b[0] for b in nbs), 4), round(max(0.0, min(b[1] for b in nbs) - 0.06), 4),
                              round(max(b[2] for b in nbs), 4), round(min(1.0, max(b[3] for b in nbs) + 0.06), 4)],
            }

    # ---- 3. Resolve label -> gauge via main-circuit label positions ----
    label_gauge_cache: Dict[str, Optional[str]] = {}

    def _gauge_for_label(label: Optional[str]) -> Optional[str]:
        if not label or label.upper() == "PE":
            return None
        if label in label_gauge_cache:
            return label_gauge_cache[label]
        best_g, best_score = None, 1e9
        for pinfo in pages_info:
            if pinfo.page_type != PageType.MAIN_CIRCUIT.value:
                continue
            for s in page_spans_map.get(pinfo.page_num, []):
                if s["text"].strip() != label:
                    continue
                for gm in gauge_marks:
                    if gm["page"] != pinfo.page_num:
                        continue
                    gx0, gy0, gx1, gy1 = gm["bbox"]
                    gcy = (gy0 + gy1) / 2.0
                    if gx0 - 6.0 <= s["cx"] <= gx1 + 6.0 and abs(gcy - s["cy"]) <= 90.0:
                        score = abs(gcy - s["cy"])  # same column
                    else:
                        d = math.hypot((gx0 + gx1) / 2.0 - s["cx"], gcy - s["cy"])
                        score = d + 1000.0 if d <= 120.0 else 1e9
                    if score < best_score:
                        best_score, best_g = score, gm["gauge"]
        label_gauge_cache[label] = best_g
        return best_g

    pin_gauge_counts: Dict[Optional[str], int] = {}
    conductors_by_gauge: Dict[str, set] = {}
    total_pins = 0
    control_pins = 0
    for sname, st in strips.items():
        prev_g: Optional[str] = None
        for pi in st["pins"]:
            total_pins += 1
            g = _gauge_for_label(pi["label"])
            if g is None and (pi["label"] or "").upper() == "PE":
                g = prev_g  # PE pin follows the power group it sits in
            if g is None:
                control_pins += 1
            else:
                if pi["label"] and pi["label"].upper() != "PE":
                    conductors_by_gauge.setdefault(g, set()).add(pi["label"])
            pin_gauge_counts[g] = pin_gauge_counts.get(g, 0) + 1
            prev_g = g

    # Wire gauges remain evidence for terminal sizing, never an estimated wire BOM.
    gauges_found = sorted({gm["gauge"] for gm in gauge_marks}, key=_gv, reverse=True)

    # ---- 6. Terminal items grouped by rating ----
    by_rating: Dict[str, Dict[str, Any]] = {}

    def _add_rating(rating: str, count: int, evs: List[EvidenceLocation], basis: str) -> None:
        slot = by_rating.setdefault(rating, {"count": 0, "evs": [], "basis": []})
        slot["count"] += count
        slot["evs"].extend(evs)
        slot["basis"].append(basis)

    strip_evs = [EvidenceLocation(page=st["page"], bbox=st["bbox"], norm_bbox=st["norm_bbox"],
                                  raw_text=f"{st['raw_header']} ({len(st['pins'])}位)", role="terminal_diagram_strip")
                 for st in strips.values()]
    for g, cnt in pin_gauge_counts.items():
        if g is None:
            _add_rating(control_rating, cnt, strip_evs, f"{cnt} 个控制/信号端子位")
        else:
            rating = rating_map.get(g.replace("mm²", ""), control_rating)
            gevs = [EvidenceLocation(page=gm["page"], bbox=gm["bbox"], norm_bbox=gm["norm_bbox"],
                                     raw_text=gm["raw"], role="wire_gauge") for gm in gauge_marks if gm["gauge"] == g][:2]
            _add_rating(rating, cnt, gevs + strip_evs, f"{cnt} 个接 {g} 的端子位")

    if incoming_poles > 0 and gauges_found:
        top_g = gauges_found[0]
        if top_g not in conductors_by_gauge:  # largest gauge never lands on a drawn terminal -> incoming feeder
            rating = rating_map.get(top_g.replace("mm²", ""), control_rating)
            gevs = [EvidenceLocation(page=gm["page"], bbox=gm["bbox"], norm_bbox=gm["norm_bbox"],
                                     raw_text=gm["raw"], role="wire_gauge") for gm in gauge_marks if gm["gauge"] == top_g]
            _add_rating(rating, incoming_poles, gevs, f"进线 {top_g} 未画端子，按配置配 {incoming_poles} 极进线端子")

    def _rating_val(r: str) -> float:
        m = re.match(r"(\d+(?:\.\d+)?)", r)
        return float(m.group(1)) if m else 0.0

    for rating in sorted(by_rating.keys(), key=_rating_val, reverse=True):
        slot = by_rating[rating]
        qty = math.ceil(slot["count"] * (1.0 + term_spare))
        evs = slot["evs"] or strip_evs
        if not evs:
            continue
        seq += 1
        items.append(BOMItem(
            id=f"term_{seq}", seq_no=seq, tag=f"X-{rating}", tags=[f"X-{rating}"], placement="",
            name="端子", model=rating, raw_model=rating, brand=terminal_brand, unit="个", quantity=float(qty),
            remark="；".join(slot["basis"]) + (f"；含 {term_spare:.0%} 备用" if term_spare else ""),
            pdf_page_label=", ".join(f"第{p}页" for p in sorted({e.page for e in evs})),
            primary_page=evs[0].page, primary_bbox=evs[0].norm_bbox, evidences=evs,
            source_type=SourceType.ESTIMATED_AUX.value, category="auxiliary", needs_review=True,
            review_note="端子位数取自端子图，额定电流按线径→端子规格配置映射，请工程师确认",
        ))

    if strips or gauge_marks:
        strip_summary = "、".join(f"{st['raw_header']}({len(st['pins'])}位)" for st in strips.values()) or "未检出端子图"
        first_ev = (strip_evs or [EvidenceLocation(page=gauge_marks[0]["page"], bbox=gauge_marks[0]["bbox"],
                                                   norm_bbox=gauge_marks[0]["norm_bbox"], raw_text="", role="")])[0]
        reviews.append(ReviewItem(
            id="rev_wires_terminals_summary", severity="INFO", issue_type="RULE_APPLIED",
            tag="端子",
            title=f"端子规格与数量核对：{len(gauges_found)} 种明文线径，端子图共 {total_pins} 位（控制位 {control_pins}）",
            description=(f"线径 {gauges_found}；端子排 {strip_summary}。端子规格为按 rules_config.json "
                         f"`wire_terminal_estimation` 的工程估算，非图纸明文。"),
            suggestion="请工程师核对端子规格映射与片数；不生成导线米数估算。",
            page=first_ev.page, norm_bbox=first_ev.norm_bbox, status="PENDING", action_type="CONFIRM",
        ))

    return items, reviews


def recover_cross_reference_evidence(doc, result, page_num=None):
    """Add missing sheet/column links to saved items without re-extracting the BOM."""
    pages = {info.page_num: info for info in result.pages}
    if page_num is not None and page_num not in pages:
        raise ValueError('图纸页码无效')
    sheets = {}
    for info in result.pages:
        if str(info.sheet_no).isdigit():
            sheets.setdefault(int(info.sheet_no), []).append(info.page_num)
    items = result.itemized_bom + result.filtered_items
    anchor_roles = {'main_circuit', 'coil', 'device', 'contact_or_symbol'}
    added = 0
    for source_page in ([page_num] if page_num is not None else sorted(pages)):
        anchors = [(item, ev) for item in items for ev in item.evidences
                   if ev.page == source_page and ev.role in anchor_roles and ev.bbox]
        if not anchors:
            continue
        page = doc[source_page - 1]
        native = extract_visual_spans(page)
        spans = DualChannelOCREngine.extract_dual_channel_spans(page, native, dpi=200)
        for marker in spans:
            reference = re.fullmatch(r'/([1-9]\d*)\.([0-9])', marker['text'].strip())
            if not reference:
                continue
            matches = []
            for item, anchor in anchors:
                gap = _reference_gap(anchor.bbox, marker['bbox'])
                if gap is not None:
                    ax = (anchor.bbox[0] + anchor.bbox[2]) / 2
                    mx = (marker['bbox'][0] + marker['bbox'][2]) / 2
                    matches.append((max(0, gap), abs(ax - mx), item))
            if not matches:
                continue
            matches.sort(key=lambda entry: entry[:2])
            item = matches[0][2]
            if any(ev.role == 'cross_reference' and ev.page == source_page and
                   all(abs(a - b) < 1 for a, b in zip(ev.bbox, marker['bbox']))
                   for ev in item.evidences):
                continue
            sheet, column = int(reference[1]), int(reference[2])
            target_pages = sheets.get(sheet, [])
            grid_x = {target: next((entry['x'] for entry in
                      pages[target].extraction_diagnostics.get('drawing_grid_columns', [])
                      if entry['number'] == column), None) for target in target_pages}
            resolved = any(ev.page == target and ev.role in anchor_roles and
                           (grid_x[target] is None or
                            abs((ev.norm_bbox[0] + ev.norm_bbox[2]) / 2 - grid_x[target]) <= .1)
                           for target in target_pages for ev in item.evidences)
            item.evidences.append(EvidenceLocation(
                source_page, marker['bbox'], marker['norm_bbox'], raw_text=marker['text'],
                role='cross_reference', device_tag=item.tag,
                cad_refs=[{'sheet': sheet, 'column': column, 'target_pages': target_pages,
                           'resolved': resolved, 'column_verified': any(x is not None for x in grid_x.values())}]))
            added += 1
    return added
