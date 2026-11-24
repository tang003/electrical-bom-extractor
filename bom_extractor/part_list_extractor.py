from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Optional
import pymupdf

from .classifier import extract_visual_spans
from .ocr_engine import DualChannelOCREngine
from .schematic_extractor import COMPONENT_TAG_RE, MODEL_TOKEN_RE, _is_model_compatible_with_tag, _infer_component_name_and_brand
from .models import BOMItem, ReviewItem, EvidenceLocation, SourceType, PageInfo, PageType


def _short_tag(tag: str) -> str:
    tag = tag.strip().upper()
    return tag[tag.index("-"):] if tag.startswith("=") and "-" in tag else tag


def _model_key(model: str) -> str:
    return re.sub(r"[\s,，]", "", model).upper()


def _nearby_models(anchor, spans, tag):
    candidates = []
    for span in spans:
        text = span["text"].strip()
        if MODEL_TOKEN_RE.fullmatch(text) and _is_model_compatible_with_tag(tag, text):
            distance = ((span["cx"]-anchor["cx"])**2 + (span["cy"]-anchor["cy"])**2)**.5
            if distance < 65:
                owners = [s for s in spans if COMPONENT_TAG_RE.fullmatch(s["text"]) and _is_model_compatible_with_tag(s["text"], text)]
                owner = min(owners, key=lambda s: (s["cx"]-span["cx"])**2+(s["cy"]-span["cy"])**2) if owners else anchor
                if owner["text"] == anchor["text"]:
                    candidates.append((distance, text))
    return list(dict.fromkeys(text for _, text in sorted(candidates)))


def _parse_placement_page_and_grid(placement: str, page_width: float, page_height: float) -> Tuple[Optional[int], List[float]]:
    """
    Parse EPLAN placement string like `=02/13.2:D` or `=02/5.1:E` into
    (schematic_page_num, approximate_norm_bbox).
    """
    m = re.search(r"/(\d+)\.(\d+):([A-F])", placement)
    if not m:
        return None, [0.1, 0.1, 0.3, 0.3]
    page_no = int(m.group(1))
    col_idx = max(1, min(8, int(m.group(2))))
    row_char = m.group(3)
    row_idx = ord(row_char) - ord("A")  # 0..5

    x0 = (col_idx - 1) / 8.0
    x1 = col_idx / 8.0
    y0 = 0.06 + (row_idx / 6.0) * 0.80
    y1 = 0.06 + ((row_idx + 1) / 6.0) * 0.80
    return page_no, [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def extract_from_part_list_pdf(
    doc: pymupdf.Document,
    pages_info: List[PageInfo],
    *, table_only: bool = False,
) -> Tuple[List[BOMItem], List[ReviewItem]]:
    """
    Extract itemized BOM rows from embedded EPLAN Part List pages (e.g., Pages 21-28)
    and cross-verify against schematic/layout pages (e.g., Pages 2-20).
    """
    raw_items: List[BOMItem] = []
    review_items: List[ReviewItem] = []

    # 1. Index schematic tags across non-part-list pages for exact bbox linking & cross-checking
    schematic_tag_locations: Dict[str, List[Dict[str, Any]]] = {}
    schematic_page_texts: Dict[int, List[Dict[str, Any]]] = {}

    for pinfo in pages_info:
        if table_only:
            break
        if pinfo.page_type == PageType.PART_LIST.value:
            continue
        page = doc[pinfo.page_num - 1]
        spans = DualChannelOCREngine.extract_dual_channel_spans(page, extract_visual_spans(page))
        schematic_page_texts[pinfo.page_num] = spans
        for s in spans:
            txt = s["text"]
            if COMPONENT_TAG_RE.fullmatch(txt) and not txt.startswith(("-X", "=+-X")):
                clean_tag = txt.split()[0]
                schematic_tag_locations.setdefault(clean_tag, []).append({
                    "page": pinfo.page_num,
                    "bbox": s["bbox"],
                    "norm_bbox": s["norm_bbox"],
                    "text": txt,
                })

    # 2. Extract all rows from PART_LIST pages
    part_list_pages = [p for p in pages_info if p.page_type == PageType.PART_LIST.value]
    seq_counter = 0

    for pl_idx, pinfo in enumerate(part_list_pages, start=1):
        page = doc[pinfo.page_num - 1]
        w = float(page.rect.width)
        h = float(page.rect.height)
        spans = DualChannelOCREngine.extract_dual_channel_spans(page, extract_visual_spans(page))

        # Filter spans inside the table body region (y in 140..755, x < 1100 to exclude stamp date)
        body_spans = [
            s for s in spans
            if 140.0 / 841.89 <= s["bbox"][1] / h <= 755.0 / 841.89 and s["bbox"][0] / w < 1100.0 / 1190.55
        ]

        # Find row anchors: 设备标识符 in column 1 (x0 < 100, text starts with '=' and has no '/')
        anchors = sorted(
            [
                s for s in body_spans
                if s["bbox"][0] / w < 100.0 / 1190.55 and (s["text"].startswith("=") or COMPONENT_TAG_RE.fullmatch(s["text"])) and "/" not in s["text"]
            ],
            key=lambda s: s["bbox"][1],
        )

        for idx, anchor in enumerate(anchors):
            seq_counter += 1
            y_top = anchor["bbox"][1] - 4.0 * h / 841.89
            if idx + 1 < len(anchors):
                y_bottom = anchors[idx + 1]["bbox"][1] - 4.0 * h / 841.89
            else:
                y_bottom = y_top + 34.0 * h / 841.89

            row_spans = [
                s for s in body_spans
                if y_top <= s["bbox"][1] < y_bottom
            ]

            device_tag = anchor["text"].strip()
            placement = ""
            qty_str = ""
            name_parts: List[str] = []
            model_parts: List[str] = []
            brand_parts: List[str] = []
            remark_parts: List[str] = []

            for s in sorted(row_spans, key=lambda x: (x["bbox"][1], x["bbox"][0])):
                x0 = s["bbox"][0] * 1190.55 / w
                txt = s["text"].strip()
                if x0 < 200.0:
                    if txt != device_tag and "/" in txt:
                        placement = txt
                elif 200.0 <= x0 < 270.0:
                    qty_str = txt
                elif 270.0 <= x0 < 560.0:
                    name_parts.append(txt)
                elif 560.0 <= x0 < 800.0:
                    model_parts.append(txt)
                elif 800.0 <= x0 < 915.0:
                    brand_parts.append(txt)
                elif 915.0 <= x0 < 1100.0:
                    remark_parts.append(txt)

            try:
                quantity = float(qty_str) if "." in qty_str else int(qty_str)
            except ValueError:
                quantity = None

            name_val = " ".join(name_parts).strip()
            model_val = " ".join(model_parts).strip()
            brand_val = " ".join(brand_parts).strip()
            remark_val = " ".join(remark_parts).strip()

            row_bbox = [45.0, round(y_top, 2), round(w - 45.0, 2), round(y_bottom, 2)]
            row_norm_bbox = [
                round(45.0 / w, 4),
                round(y_top / h, 4),
                round((w - 45.0) / w, 4),
                round(y_bottom / h, 4),
            ]

            evidences = [
                EvidenceLocation(
                    page=pinfo.page_num,
                    bbox=row_bbox,
                    norm_bbox=row_norm_bbox,
                    raw_text=f"{device_tag} | {name_val} | {model_val} | {brand_val} | {qty_str}",
                    role="table_row",
                )
            ]

            # Link schematic location from placement (e.g., =02/13.2:D -> Page 13)
            short_tag = _short_tag(device_tag)
            schem_page, schem_grid_bbox = _parse_placement_page_and_grid(placement, w, h)
            if short_tag in schematic_tag_locations:
                for loc in schematic_tag_locations[short_tag]:
                    evidences.append(
                        EvidenceLocation(
                            page=loc["page"],
                            bbox=loc["bbox"],
                            norm_bbox=loc["norm_bbox"],
                            raw_text=loc["text"],
                            role="schematic_symbol",
                        )
                    )
            elif schem_page and not table_only:
                evidences.append(
                    EvidenceLocation(
                        page=schem_page,
                        bbox=[
                            round(schem_grid_bbox[0] * w, 2),
                            round(schem_grid_bbox[1] * h, 2),
                            round(schem_grid_bbox[2] * w, 2),
                            round(schem_grid_bbox[3] * h, 2),
                        ],
                        norm_bbox=schem_grid_bbox,
                        raw_text=f"{device_tag} ({placement})",
                        role="schematic_grid",
                    )
                )

            raw_items.append(
                BOMItem(
                    id=f"item_{seq_counter}",
                    seq_no=seq_counter,
                    tag=device_tag,
                    tags=[device_tag],
                    placement=placement,
                    name=name_val,
                    model=model_val,
                    raw_model=model_val,
                    brand=brand_val,
                    unit='',
                    quantity=quantity,
                    remark=remark_val,
                    pdf_page_label=f"第{pinfo.page_num}页",
                    primary_page=pinfo.page_num,
                    primary_bbox=row_norm_bbox,
                    evidences=evidences,
                    source_type=SourceType.EXPLICIT.value,
                    category="primary",
                    needs_review=table_only,
                    quantity_status='unknown' if quantity is None else 'needs_confirmation' if table_only else 'by_tag',
                    quantity_basis='unknown' if quantity is None else 'explicit_total' if table_only else 'unique_tag',
                    observations=[{'source':'drawing_parts_list','cells':{'tag':device_tag,'name':name_val,'model':model_val,
                        'quantity':qty_str,'brand':brand_val,'unit':'','remark':remark_val,'placement':placement}}] if table_only else [],
                )
            )

    if table_only:
        return raw_items, []

    # Compare every observed device against every listed device, independent of tag/page numbers.
    listed = {}
    for item in raw_items:
        listed.setdefault(_short_tag(item.tag), []).append(item)
    for observed_tag, locations in schematic_tag_locations.items():
        short_tag = _short_tag(observed_tag)
        entries = listed.get(short_tag, [])
        location = locations[0]
        page_no = location["page"]
        if not entries:
            near = schematic_page_texts.get(page_no, [])
            tag_span = next((sp for sp in near if sp["text"] == observed_tag), None)
            models = _nearby_models(tag_span, near, short_tag) if tag_span else []
            model = models[0] if len(models) == 1 else "图纸未注(待定)"
            name, brand = _infer_component_name_and_brand(short_tag, model, [])
            review_items.append(ReviewItem(id=f"rev_missing_{observed_tag}", severity="WARNING",
                issue_type="MISSING_IN_PART_LIST", tag=observed_tag,
                title=f"疑似漏项：原理图位号 {observed_tag} 未列入部件列表",
                description=f"在第 {page_no} 页检出位号，未找到对应的部件列表记录。",
                suggestion="核实是否属于采购范围，并确认型号后补入 BOM。", page=page_no,
                norm_bbox=location["norm_bbox"], action_type="ADD_ITEM",
                proposed_data={"tag": observed_tag, "name": name, "model": model,
                               "brand": brand or "待定", "quantity": 1, "unit": "个"}))
            continue
        seen_conflicts = set()
        for loc in locations:
            near = schematic_page_texts.get(loc["page"], [])
            tag_span = next((sp for sp in near if sp["text"] == observed_tag), None)
            models = _nearby_models(tag_span, near, short_tag) if tag_span else []
            for model in models:
                model_key = _model_key(model)
                if any(model_key in _model_key(item.model) for item in entries):
                    continue
                if model_key in seen_conflicts:
                    continue
                seen_conflicts.add(model_key)
                target = entries[0]
                review_items.append(ReviewItem(id=f"rev_model_{observed_tag}_{len(seen_conflicts)}",
                    severity="CONFLICT", issue_type="PARAM_MISMATCH", tag=observed_tag,
                    title=f"图文型号待核实：{observed_tag}",
                    description=f"第 {loc['page']} 页附近型号 {model} 与部件列表 {' / '.join(it.model for it in entries)} 不一致。",
                    suggestion="核实是否为主件、附件或旧图标注，再确认最终型号。", page=loc["page"],
                    norm_bbox=loc["norm_bbox"], related_page=target.primary_page, related_norm_bbox=target.primary_bbox))
            if tag_span:
                local_text = " ".join(sp["text"] for sp in near if abs(sp["cx"]-tag_span["cx"]) < 90 and abs(sp["cy"]-tag_span["cy"]) < 90)
                if re.search(r"(?:漏电|剩余电流|RCD).*?30\s*mA", local_text, re.I) and not any(re.search(r"VIGI|5SU|A9D|RCBO|漏电", it.model, re.I) for it in entries):
                    review_items.append(ReviewItem(id=f"rev_rcd_{observed_tag}_{loc['page']}", severity="CONFLICT",
                        issue_type="PARAM_MISMATCH", tag=observed_tag, title=f"漏电保护要求待核实：{observed_tag}",
                        description="原理图局部标注 30mA 漏电保护要求，部件表型号未明确体现该功能。",
                        suggestion="请确认本体及漏电保护附件选型，不自动替换为猜测型号。", page=loc["page"], norm_bbox=loc["norm_bbox"]))

    # Check 3.4: Same model number with inconsistent manufacturer/brand (e.g., RXZE2M114 施耐德 vs 菲尼克斯)
    model_brand_counts: Dict[str, Dict[str, int]] = {}
    for it in raw_items:
        if it.model and it.brand:
            model_brand_counts.setdefault(it.model, {})
            model_brand_counts[it.model][it.brand] = model_brand_counts[it.model].get(it.brand, 0) + 1

    for it in raw_items:
        brands = model_brand_counts.get(it.model, {})
        if len(brands) > 1:
            dominant_brand = max(brands.items(), key=lambda kv: kv[1])[0]
            if it.brand != dominant_brand and brands[dominant_brand] >= 3:
                it.needs_review = True
                it.review_note = f"同型号 `{it.model}` 其余 {brands[dominant_brand]} 处品牌均为 `{dominant_brand}`，此处疑为笔误 `{it.brand}`"
                review_items.append(
                    ReviewItem(
                        id=f"rev_brand_typo_{it.seq_no}",
                        severity="WARNING",
                        issue_type="BRAND_TYPO",
                        tag=it.tag,
                        title=f"制造商疑似笔误：序号 {it.seq_no} [{it.tag}] 型号 `{it.model}` 制造商填为 `{it.brand}`",
                        description=(
                            f"在第 {it.primary_page} 页《部件列表》（序号 {it.seq_no}）中，`{it.tag}` 的继电器底座 `{it.model}` "
                            f"制造商被填写为 `{it.brand}`，但全图其余 {brands[dominant_brand]} 个相同型号 `{it.model}` 制造商均为 `{dominant_brand}`。"
                        ),
                        suggestion=f"若确系 EPLAN 部件库录入笔误，可在网页表格中将制造商改为 `{dominant_brand}`。",
                        page=it.primary_page,
                        norm_bbox=it.primary_bbox,
                        status="PENDING",
                        action_type="UPDATE_ITEM",
                        proposed_data={
                            "target_seq_no": it.seq_no,
                            "brand": dominant_brand,
                        },
                    )
                )

    return raw_items, review_items
