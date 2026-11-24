from __future__ import annotations
from contextlib import closing
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import openpyxl

from .models import ExtractionResult, BOMItem


def _norm_str(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    # Normalize whitespace and fullwidth punctuation
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_model_key(val: Any) -> str:
    s = _norm_str(val)
    # Remove spaces around commas/plus and unify full-width commas for comparison
    s = s.replace("，", ",").replace(" ", "")
    return s.upper()


def load_ground_truth_excel(excel_path: Path | str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load reference standard BOM Excel (supports both `C26368` aggregated format
    and `双钱L9` EPLAN Part List format).
    Returns (format_type, rows).
    """
    with closing(openpyxl.load_workbook(excel_path, read_only=True, data_only=True)) as wb:
        rows = wb.worksheets[0].values
        for row in rows:
            headers = [_norm_str(value) for value in row]
            if "序号" in headers and ("型号" in headers or "订货编号" in headers):
                break
        else:
            raise ValueError(f"{Path(excel_path).name} 第一张工作表缺少标准 BOM 表头")
        if "数量" not in headers:
            raise ValueError(f"{Path(excel_path).name} 标准 BOM 表头缺少数量列")

        is_part_list_fmt = "订货编号" in headers
        gt_rows: List[Dict[str, Any]] = []

        for vals in rows:
            if not any(vals):
                continue
            row_dict = {headers[i]: vals[i] for i in range(len(headers)) if headers[i]}
            if row_dict.get("序号") is None:
                continue
            value = row_dict["数量"]
            quantity = float(value) if _norm_str(value) else None
            if is_part_list_fmt:
                gt_rows.append({
                    "seq_no": int(row_dict["序号"]),
                    "tag": _norm_str(row_dict.get("设备标识符(位号)") or row_dict.get("设备标识符") or row_dict.get("位号")),
                    "name": _norm_str(row_dict.get("名称/描述")),
                    "quantity": quantity,
                    "model": _norm_str(row_dict.get("订货编号")),
                    "brand": _norm_str(row_dict.get("制造商")),
                    "remark": _norm_str(row_dict.get("备注")),
                    "unit": _norm_str(row_dict.get("单位")),
                    "pdf_page_label": _norm_str(row_dict.get("PDF页码")),
                })
            else:
                gt_rows.append({
                    "seq_no": int(row_dict["序号"]),
                    "tag": _norm_str(row_dict.get("设备标识符(位号)") or row_dict.get("设备标识符") or row_dict.get("位号")),
                    "brand": _norm_str(row_dict.get("品牌")),
                    "name": _norm_str(row_dict.get("名称")),
                    "model": _norm_str(row_dict.get("型号")),
                    "unit": _norm_str(row_dict.get("单位")),
                    "quantity": quantity,
                })

    if not gt_rows:
        raise ValueError(f"{Path(excel_path).name} 标准 BOM 没有物料行")
    return ("PART_LIST_MODE" if is_part_list_fmt else "SCHEMATIC_MODE"), gt_rows


def evaluate_extraction(
    result: ExtractionResult,
    ground_truth_excel: Path | str,
) -> Dict[str, Any]:
    """
    Compare ExtractionResult against a ground-truth Excel BOM and compute:
    - 物料召回率 (Recall Rate)
    - 型号准确率 (Model Accuracy)
    - 数量准确率 (Quantity Accuracy)
    - 整单正确率 (Exact Row Match Rate)
    """
    gt_mode, gt_rows = load_ground_truth_excel(ground_truth_excel)
    pred_items: List[BOMItem] = (
        result.itemized_bom if gt_mode == "PART_LIST_MODE" else result.aggregated_bom
    )

    # Match business identities one-to-one; row numbers are presentation only.
    unused = set(range(len(pred_items)))
    matches = {}
    def compare(gt, pred):
        tag = gt["tag"]
        quantity = gt["quantity"]
        qty_ok = (pred.quantity is None if quantity is None else
                  pred.quantity is not None and abs(float(pred.quantity) - quantity) < 1e-5)
        return {
            "tag_ok": not tag or tag == pred.tag or tag in pred.tags,
            "model_ok": _norm_model_key(pred.model) == _norm_model_key(gt["model"]),
            "qty_ok": qty_ok,
            "name_ok": _norm_str(pred.name) == gt["name"],
            "brand_ok": _norm_str(pred.brand).upper() == gt["brand"].upper(),
            "unit_ok": not gt["unit"] or _norm_str(pred.unit) == gt["unit"],
        }

    # Reserve exact rows before pairing partial matches for the difference report.
    # A reference with an explicit tag has fewer valid matches than an untagged row.
    reference_order = sorted(range(len(gt_rows)), key=lambda gi: not gt_rows[gi]["tag"])
    for strict in (True, False):
        for gi in reference_order:
            if gi in matches:
                continue
            gt = gt_rows[gi]
            candidates = []
            for pi in unused:
                fields = compare(gt, pred_items[pi])
                tag_match = bool(gt["tag"]) and fields["tag_ok"]
                model_match = bool(gt["model"]) and fields["model_ok"]
                blank_identity = (not gt["model"] and fields["model_ok"]
                                  and fields["name_ok"] and fields["brand_ok"])
                if not (tag_match or model_match or blank_identity):
                    continue
                if strict and not all(fields.values()):
                    continue
                candidates.append((tag_match, sum(fields.values()), -pi, pi, fields))
            if candidates:
                _, _, _, pi, fields = max(candidates)
                matches[gi] = (pi, fields)
                unused.remove(pi)

    model_count = qty_count = exact_count = 0
    mismatches = []
    for gi, gt in enumerate(gt_rows):
        if gi not in matches:
            mismatches.append({"kind": "MISSING", "seq_no": gt["seq_no"], "ground_truth": gt,
                               "predicted": None, "reason": "未匹配到该物料"})
            continue
        pi, fields = matches[gi]
        model_count += fields["model_ok"]
        qty_count += fields["qty_ok"]
        exact = all(fields.values())
        exact_count += exact
        if not exact:
            mismatches.append({"kind": "MISMATCH", "seq_no": gt["seq_no"], "ground_truth": gt,
                               "predicted": pred_items[pi].to_dict(), **fields})
    for pi in sorted(unused):
        mismatches.append({"kind": "EXTRA", "ground_truth": None, "predicted": pred_items[pi].to_dict(), "reason": "多余预测物料"})
    total_gt, total_pred = len(gt_rows), len(pred_items)
    percent = lambda count, total: round(count / total * 100, 2) if total else 0.0
    recalled = len(matches)
    recall, precision = percent(recalled, total_gt), percent(recalled, total_pred)
    # Extra rows count against accuracy as well as precision.
    denominator = total_gt + len(unused)
    model_acc, qty_acc, exact_rate = (percent(n, denominator) for n in (model_count, qty_count, exact_count))
    whole_exact = not mismatches and bool(total_gt)
    return {
        "pdf_filename": result.pdf_filename, "ground_truth_file": Path(ground_truth_excel).name,
        "evaluation_mode": gt_mode, "total_ground_truth_items": total_gt, "total_predicted_items": total_pred,
        "recalled_items": recalled, "extra_items": len(unused), "missing_items": total_gt-recalled,
        "model_matched_items": model_count, "quantity_matched_items": qty_count,
        "exact_matched_items": exact_count, "whole_bom_exact": whole_exact,
        "metric_definition": "业务字段一对一匹配；准确率分母为标准行数+多余预测行数；整单完全正确须无任何差异",
        "metrics": {
            "物料召回率 (Recall)": f"{recall:.2f}%", "物料精确率 (Precision)": f"{precision:.2f}%",
            "型号准确率 (Model Accuracy)": f"{model_acc:.2f}%", "数量准确率 (Quantity Accuracy)": f"{qty_acc:.2f}%",
            "逐行完全匹配率 (Exact Row Match)": f"{exact_rate:.2f}%",
            "recall_value": recall, "precision_value": precision,
            "f1_value": round(2*recall*precision/(recall+precision), 2) if recall+precision else 0.0,
            "model_accuracy_value": model_acc, "quantity_accuracy_value": qty_acc, "exact_match_value": exact_rate,
        },
        "review_anomalies_detected": len(result.review_items), "mismatches": mismatches,
    }
