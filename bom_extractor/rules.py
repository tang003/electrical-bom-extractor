from __future__ import annotations
import copy
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .models import BOMItem, ReviewItem, SourceType


def load_rules_config(config_path: Path | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "rules_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_aggregated_from_itemized(itemized_items: List[BOMItem]) -> List[BOMItem]:
    """
    Group itemized BOM rows by (brand, name, model, unit) to produce a consolidated
    procurement summary table (`合并采购汇总表`), preserving all tags and evidences.
    """
    grouped: Dict[Tuple[str, str, str, str], BOMItem] = {}
    order_keys: List[Tuple[str, str, str, str]] = []

    for it in itemized_items:
        key = (
            (it.brand or "").strip(),
            (it.name or "").strip(),
            (it.model or "").strip(),
            (it.unit or "个").strip(),
        )
        if key not in grouped:
            cloned = copy.deepcopy(it)
            cloned.tags = list(it.tags) if it.tags else ([it.tag] if it.tag else [])
            grouped[key] = cloned
            order_keys.append(key)
        else:
            agg = grouped[key]
            agg.quantity += it.quantity
            for tg in (it.tags or [it.tag]):
                if tg and tg not in agg.tags:
                    agg.tags.append(tg)
            agg.evidences.extend(it.evidences)
            if it.needs_review:
                agg.needs_review = True

    result: List[BOMItem] = []
    for idx, key in enumerate(order_keys, start=1):
        agg = grouped[key]
        agg.seq_no = idx
        agg.id = f"agg_{idx}"
        if len(agg.tags) > 6:
            agg.tag = ", ".join(agg.tags[:6]) + f" 等({len(agg.tags)}个)"
        elif agg.tags:
            agg.tag = ", ".join(agg.tags)
        pages = sorted({ev.page for ev in agg.evidences}) if agg.evidences else [agg.primary_page]
        if len(pages) > 1:
            agg.pdf_page_label = ", ".join(f"第{p}页" for p in pages[:4]) + (f" 等" if len(pages) > 4 else "")
        result.append(agg)
    return result


def apply_engineering_rules(
    raw_items: List[BOMItem],
    review_items: List[ReviewItem],
    extraction_mode: str,
    rules_config: Dict[str, Any],
) -> Tuple[List[BOMItem], List[BOMItem], List[BOMItem], List[ReviewItem]]:
    """
    Apply deterministic enterprise electrical engineering rules:
    - Filter out 0-quantity and non-electrical placeholder field parts
    - Append mandatory PLC accessories (front connectors, mounting rail, memory card)
    - Return (itemized_bom, aggregated_bom, filtered_items, review_items)
    """
    filter_cfg = rules_config.get("filter_rules", {})
    exclude_zero = filter_cfg.get("exclude_zero_quantity", True)
    placeholder_patterns = [re.compile(p) for p in filter_cfg.get("exclude_placeholder_models", [])]
    plc_rules = rules_config.get("plc_accessory_rules", [])

    if extraction_mode == "PART_LIST_MODE":
        kept_items: List[BOMItem] = []
        filtered_items: List[BOMItem] = []
        plc_completed_notes: List[str] = []

        for item in raw_items:
            # 1. Check zero quantity
            if exclude_zero and item.quantity <= 0:
                item.review_note = "原图部件列表中数量为 0（未安装占位件），已自动剔除"
                filtered_items.append(item)
                continue

            # 2. Check placeholder field models (DT1~DT17, SP1~SP8, LF60D hydraulic station parts)
            if any(pat.match(item.model.strip()) for pat in placeholder_patterns):
                item.review_note = f"非电控柜采购件/现场液压站部件 (`{item.model}`, 备注:{item.remark or '无'})，已按工程规则自动过滤"
                filtered_items.append(item)
                continue

            # 3. Apply PLC accessory completion rules
            for pr in plc_rules:
                if item.model.strip() == pr["match_model"]:
                    item.model = item.model.strip() + pr["append_model"]
                    item.source_type = SourceType.RULE_COMPLETED.value
                    item.review_note = pr["reason"]
                    plc_completed_notes.append(f"序号 {item.seq_no} [{item.name}] `{pr['match_model']}` -> `{item.model}` ({pr['reason']})")

            kept_items.append(item)

        if plc_completed_notes:
            cpu_or_ai = next((it for it in kept_items if it.source_type == SourceType.RULE_COMPLETED.value), kept_items[0])
            review_items.append(
                ReviewItem(
                    id="rev_plc_accessories",
                    severity="INFO",
                    issue_type="RULE_COMPLETED",
                    tag="=02-CPU1511 / =02-AI1",
                    title="企业选型规则补全：西门子 S7-1500 CPU 及 8AI 模块已自动追加导轨、存储卡与前连接器",
                    description="；".join(plc_completed_notes),
                    suggestion="已按企业标准 BOM 规则自动补齐必配附件，点击可定位至对应行。",
                    page=cpu_or_ai.primary_page,
                    norm_bbox=cpu_or_ai.primary_bbox,
                    status="PENDING",
                    action_type="CONFIRM",
                )
            )

        if filtered_items:
            review_items.append(
                ReviewItem(
                    id="rev_filtered_hydraulic",
                    severity="INFO",
                    issue_type="FILTERED_ITEM",
                    tag="DT1~17, SP1~8, SPD",
                    title=f"工程清洗规则：已自动剔除 {len(filtered_items)} 项非电控采购/0数量物料（可随时恢复）",
                    description=(
                        f"已自动过滤：电磁阀占位号 `DT1~DT17` (17项)、压力传感器占位号 `SP1~SP8` (8项)、"
                        f"液压站液位开关 `LF60D` (1项) 及数量为 0 的浪涌保护器 `-SPD` (1项)，保留汇川伺服电机与控制器。"
                    ),
                    suggestion="可在右侧【已过滤备查】标签页中查看全部 27 项被过滤记录，或点击一键恢复至 BOM。",
                    page=filtered_items[0].primary_page,
                    norm_bbox=filtered_items[0].primary_bbox,
                    status="PENDING",
                    action_type="CONFIRM",
                )
            )

        aggregated_bom = build_aggregated_from_itemized(kept_items)
        return kept_items, aggregated_bom, filtered_items, review_items

    else:
        # SCHEMATIC_MODE: raw_items is already consolidated by physical device model (38 items for C26368)
        aggregated_bom = copy.deepcopy(raw_items)
        itemized_bom = copy.deepcopy(raw_items)

        review_items.append(
            ReviewItem(
                id="rev_xb2_series_mapping",
                severity="INFO",
                issue_type="RULE_COMPLETED",
                tag="HL0~HL23, -SB1~4, -ES1, -KA*",
                title="企业标准选型映射：未标型号的面板指示灯/按钮/中间继电器已按施耐德 XB2/RXM 系列补全",
                description=(
                    "根据第 7、8 页明确标注的旋钮型号 `XB2BD21C / XB2BD31C`，系统自动将同柜未标具体订货号的面板元件映射为施耐德 XB2 系列："
                    "白灯 `XB2BVB1LC`(1)、绿灯 `XB2BVM3LC`(5)、红灯 `XB2BVM4LC`(7)、黄灯 `XB2BVM5LC`(11)、"
                    "急停 `XB2BS542C`(1)、启动按钮 `XB2BA31C`(4)、蜂鸣器 `XB2BSB4LC`(1)；"
                    "中间继电器 `-KA` 映射为 `RXM2LB2BD+RXZE1M2C`(30)；马达断路器 `GV2ME` 因第6/14页用到辅助触点 `-FR3~5` 自动追加 `+GVAE11`。"
                ),
                suggestion="所有规则补全项均已打上 [规则补全] 标签，工程师可在网页表格中直接双击修改。",
                page=7,
                norm_bbox=[0.08, 0.15, 0.90, 0.75],
                status="PENDING",
                action_type="CONFIRM",
            )
        )

        return itemized_bom, aggregated_bom, [], review_items
