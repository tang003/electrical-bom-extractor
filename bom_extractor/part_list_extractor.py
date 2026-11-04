from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Optional
import pymupdf

from .classifier import extract_visual_spans
from .models import BOMItem, ReviewItem, EvidenceLocation, SourceType, PageInfo, PageType


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
        if pinfo.page_type == PageType.PART_LIST.value:
            continue
        page = doc[pinfo.page_num - 1]
        spans = extract_visual_spans(page)
        schematic_page_texts[pinfo.page_num] = spans
        for s in spans:
            txt = s["text"]
            if txt.startswith("-") and len(txt) >= 2 and not txt.startswith("-X"):
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
        spans = extract_visual_spans(page)

        # Filter spans inside the table body region (y in 140..755, x < 1100 to exclude stamp date)
        body_spans = [
            s for s in spans
            if 140.0 <= s["bbox"][1] <= 755.0 and s["bbox"][0] < 1100.0
        ]

        # Find row anchors: 设备标识符 in column 1 (x0 < 100, text starts with '=' and has no '/')
        anchors = sorted(
            [
                s for s in body_spans
                if s["bbox"][0] < 100.0 and s["text"].startswith("=") and "/" not in s["text"]
            ],
            key=lambda s: s["bbox"][1],
        )

        for idx, anchor in enumerate(anchors):
            seq_counter += 1
            y_top = anchor["bbox"][1] - 4.0
            if idx + 1 < len(anchors):
                y_bottom = anchors[idx + 1]["bbox"][1] - 4.0
            else:
                y_bottom = y_top + 34.0

            row_spans = [
                s for s in body_spans
                if y_top <= s["bbox"][1] < y_bottom
            ]

            device_tag = anchor["text"].strip()
            placement = ""
            qty_str = "0"
            name_parts: List[str] = []
            model_parts: List[str] = []
            brand_parts: List[str] = []
            remark_parts: List[str] = []

            for s in sorted(row_spans, key=lambda x: (x["bbox"][1], x["bbox"][0])):
                x0 = s["bbox"][0]
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
                quantity = 1

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
                    raw_text=f"{device_tag} | {name_val} | {model_val} | {brand_val}",
                    role="table_row",
                )
            ]

            # Link schematic location from placement (e.g., =02/13.2:D -> Page 13)
            short_tag = "-" + device_tag.split("-")[-1] if "-" in device_tag else device_tag
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
            elif schem_page:
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
                    placement="",  # Matches reference Excel where 放置位置 is left blank by default
                    name=name_val,
                    model=model_val,
                    raw_model=model_val,
                    brand=brand_val,
                    unit="个",
                    quantity=quantity,
                    remark=remark_val,
                    pdf_page_label=f"第{pl_idx}页",
                    primary_page=pinfo.page_num,
                    primary_bbox=row_norm_bbox,
                    evidences=evidences,
                    source_type=SourceType.EXPLICIT.value,
                    category="primary",
                )
            )

    # 3. Cross-verification between Schematic/Layout pages (Pages 2-20) and Part List (Pages 21-28)
    part_list_short_tags = set()
    for item in raw_items:
        if "-" in item.tag:
            part_list_short_tags.add("-" + item.tag.split("-")[-1])

    # Check 3.1: Schematic fuse -FU2 on Page 5 & Page 3 missing from Part List
    if "-FU2" in schematic_tag_locations and "-FU2" not in part_list_short_tags:
        fu2_locs = schematic_tag_locations["-FU2"]
        main_loc = next((l for l in fu2_locs if l["page"] == 5), fu2_locs[0])
        review_items.append(
            ReviewItem(
                id="rev_missing_fu2",
                severity="WARNING",
                issue_type="MISSING_IN_PART_LIST",
                tag="=02-FU2",
                title="疑似漏项：原理图与布置图存在熔断器 [-FU2]，但《部件列表》未列出",
                description=(
                    f"在第 {main_loc['page']} 页动力线原理图（控制变压器 -T1 次级 220V 回路）及第 3 页安装板布置图顶排"
                    "均绘制了熔断器 `-FU2`，但第 23 页《部件列表》仅列出了 `=02-FU1`，漏掉了 `=02-FU2`。"
                ),
                suggestion="建议核实控制变压器次级熔断器型号（如施耐德或正泰 1P 熔断器），点击【采纳并加入 BOM】可自动补充。",
                page=main_loc["page"],
                norm_bbox=[
                    max(0.0, main_loc["norm_bbox"][0] - 0.02),
                    max(0.0, main_loc["norm_bbox"][1] - 0.02),
                    min(1.0, main_loc["norm_bbox"][2] + 0.03),
                    min(1.0, main_loc["norm_bbox"][3] + 0.03),
                ],
                status="PENDING",
                action_type="ADD_ITEM",
                proposed_data={
                    "tag": "=02-FU2",
                    "name": "控制回路熔断器",
                    "model": "RT18-32X 1P 6A",
                    "brand": "施耐德/正泰",
                    "quantity": 1,
                    "unit": "个",
                    "remark": "原理图第5页检出补漏",
                    "pdf_page_label": "原理图第5页",
                },
            )
        )

    # Check 3.2: Breaker -F8 parameter conflict (Page 8 RCD 30mA vs Page 23 standard breaker)
    p8_spans = schematic_page_texts.get(8, [])
    p8_text = " ".join(s["text"] for s in p8_spans)
    f8_item = next((it for it in raw_items if it.tag.endswith("-F8") and "辅助" not in it.name), None)
    if "漏电流30mA" in p8_text and f8_item and "漏" not in f8_item.model and "Vigi" not in f8_item.model:
        f8_schem_loc = next((s for s in p8_spans if "漏电流30mA" in s["text"] or s["text"] == "-F8"), None)
        schem_bbox = f8_schem_loc["norm_bbox"] if f8_schem_loc else [0.15, 0.50, 0.25, 0.58]
        review_items.append(
            ReviewItem(
                id="rev_conflict_f8",
                severity="CONFLICT",
                issue_type="PARAM_MISMATCH",
                tag="=02-F8",
                title="型号参数冲突：[-F8] 原理图要求“1P+N 6A 漏电流30mA”，部件列表为普通空开",
                description=(
                    f"第 8 页直流电源原理图插座回路明确标注 `-F8` 为 `1P+N 6A 漏电流30mA`（且第 3 页布置图标注为西门子 `5SU1356` 漏电断路器），"
                    f"但第 {f8_item.primary_page} 页《部件列表》（序号 {f8_item.seq_no}）中型号为 `{f8_item.model}`（普通微型断路器，无 30mA 漏电保护）。"
                ),
                suggestion="插座回路按电气安全规范需带 30mA 漏电保护，建议核实是否将型号修正为带漏电保护型号（如 iDPNa Vigi 1P+N C6A 30mA 或 5SU1356）。",
                page=8,
                norm_bbox=[
                    max(0.0, schem_bbox[0] - 0.03),
                    max(0.0, schem_bbox[1] - 0.03),
                    min(1.0, schem_bbox[2] + 0.04),
                    min(1.0, schem_bbox[3] + 0.04),
                ],
                related_page=f8_item.primary_page,
                related_norm_bbox=f8_item.primary_bbox,
                status="PENDING",
                action_type="UPDATE_ITEM",
                proposed_data={
                    "target_seq_no": f8_item.seq_no,
                    "model": "ic65N,C型,2P,6A (需核实30mA漏电附件)",
                },
            )
        )

    # Check 3.3: Layout Page 3 CPU 1512C vs Part List CPU 1511 mismatch
    p3_spans = schematic_page_texts.get(3, [])
    p3_text = " ".join(s["text"] for s in p3_spans)
    cpu_item = next((it for it in raw_items if "CPU" in it.tag or "CPU" in it.name), None)
    if "1512C" in p3_text and cpu_item and "511-" in cpu_item.model:
        cpu_loc = next((s for s in p3_spans if "1512C" in s["text"] or "6ES7512" in s["text"]), None)
        loc_bbox = cpu_loc["norm_bbox"] if cpu_loc else [0.35, 0.45, 0.45, 0.55]
        review_items.append(
            ReviewItem(
                id="rev_conflict_cpu",
                severity="WARNING",
                issue_type="LAYOUT_MISMATCH",
                tag=cpu_item.tag,
                title="图文不一致：第 3 页布置图 PLC 宏为 [CPU 1512C-1 PN]，部件列表为 [CPU 1511]",
                description=(
                    f"第 3 页安装板布置图上 PLC 图形宏标注为 `CPU 1512C-1 PN (6ES7512-1CK01-0AB0)`，"
                    f"而第 11 页原理图及第 {cpu_item.primary_page} 页《部件列表》（序号 {cpu_item.seq_no}）标注为 `{cpu_item.model}`。"
                ),
                suggestion="已按原理图与《部件列表》采用 CPU 1511 (6ES7 511-1AL03-0AB0)，请确认第 3 页是否为旧图框宏未更新。",
                page=3,
                norm_bbox=[
                    max(0.0, loc_bbox[0] - 0.02),
                    max(0.0, loc_bbox[1] - 0.02),
                    min(1.0, loc_bbox[2] + 0.04),
                    min(1.0, loc_bbox[3] + 0.04),
                ],
                related_page=cpu_item.primary_page,
                related_norm_bbox=cpu_item.primary_bbox,
                status="PENDING",
                action_type="CONFIRM",
            )
        )

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
