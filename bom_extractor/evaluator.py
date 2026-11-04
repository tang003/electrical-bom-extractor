from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, List
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
    wb = openpyxl.load_workbook(str(excel_path), data_only=True)
    ws = wb[wb.sheetnames[0]]

    # Find header row
    header_row_idx = 1
    headers: List[str] = []
    for r in range(1, min(10, ws.max_row + 1)):
        vals = [_norm_str(ws.cell(r, c).value) for c in range(1, ws.max_column + 1)]
        if "序号" in vals and ("型号" in vals or "订货编号" in vals):
            header_row_idx = r
            headers = vals
            break

    is_part_list_fmt = "订货编号" in headers
    gt_rows: List[Dict[str, Any]] = []

    for r in range(header_row_idx + 1, ws.max_row + 1):
        vals = [ws.cell(r, c).value for c in range(1, len(headers) + 1)]
        if not any(vals):
            continue
        row_dict = {headers[i]: vals[i] for i in range(len(headers)) if headers[i]}
        if row_dict.get("序号") is None:
            continue
        if is_part_list_fmt:
            gt_rows.append({
                "seq_no": int(row_dict["序号"]),
                "name": _norm_str(row_dict.get("名称/描述")),
                "quantity": float(row_dict.get("数量") or 0),
                "model": _norm_str(row_dict.get("订货编号")),
                "brand": _norm_str(row_dict.get("制造商")),
                "remark": _norm_str(row_dict.get("备注")),
                "pdf_page_label": _norm_str(row_dict.get("PDF页码")),
            })
        else:
            gt_rows.append({
                "seq_no": int(row_dict["序号"]),
                "brand": _norm_str(row_dict.get("品牌")),
                "name": _norm_str(row_dict.get("名称")),
                "model": _norm_str(row_dict.get("型号")),
                "unit": _norm_str(row_dict.get("单位")),
                "quantity": float(row_dict.get("数量") or 0),
            })

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

    pred_by_seq = {it.seq_no: it for it in pred_items}
    total_gt = len(gt_rows)
    recalled_count = 0
    model_match_count = 0
    qty_match_count = 0
    exact_match_count = 0
    mismatches: List[Dict[str, Any]] = []

    for gt in gt_rows:
        seq = gt["seq_no"]
        pred = pred_by_seq.get(seq)
        if pred is None:
            # Try fallback matching by normalized model + name
            pred = next(
                (
                    it for it in pred_items
                    if _norm_model_key(it.model) == _norm_model_key(gt["model"])
                    and _norm_str(it.name) == _norm_str(gt["name"])
                ),
                None,
            )

        if pred is not None:
            recalled_count += 1
            m_ok = _norm_model_key(pred.model) == _norm_model_key(gt["model"])
            q_ok = abs(float(pred.quantity) - float(gt["quantity"])) < 1e-5
            b_ok = _norm_str(pred.brand).upper() == _norm_str(gt["brand"]).upper()
            n_ok = _norm_str(pred.name) == _norm_str(gt["name"])

            if m_ok:
                model_match_count += 1
            if q_ok:
                qty_match_count += 1
            if m_ok and q_ok and b_ok and n_ok:
                exact_match_count += 1
            else:
                mismatches.append({
                    "seq_no": seq,
                    "ground_truth": gt,
                    "predicted": {
                        "seq_no": pred.seq_no,
                        "name": pred.name,
                        "model": pred.model,
                        "brand": pred.brand,
                        "quantity": pred.quantity,
                    },
                    "model_ok": m_ok,
                    "qty_ok": q_ok,
                    "brand_ok": b_ok,
                    "name_ok": n_ok,
                })
        else:
            mismatches.append({
                "seq_no": seq,
                "ground_truth": gt,
                "predicted": None,
                "reason": "未召回该序号物料",
            })

    recall_rate = round(recalled_count / total_gt * 100.0, 2) if total_gt else 0.0
    model_acc = round(model_match_count / total_gt * 100.0, 2) if total_gt else 0.0
    qty_acc = round(qty_match_count / total_gt * 100.0, 2) if total_gt else 0.0
    exact_rate = round(exact_match_count / total_gt * 100.0, 2) if total_gt else 0.0

    return {
        "pdf_filename": result.pdf_filename,
        "ground_truth_file": Path(ground_truth_excel).name,
        "evaluation_mode": gt_mode,
        "total_ground_truth_items": total_gt,
        "total_predicted_items": len(pred_items),
        "recalled_items": recalled_count,
        "model_matched_items": model_match_count,
        "quantity_matched_items": qty_match_count,
        "exact_matched_items": exact_match_count,
        "metrics": {
            "物料召回率 (Recall)": f"{recall_rate:.2f}%",
            "型号准确率 (Model Accuracy)": f"{model_acc:.2f}%",
            "数量准确率 (Quantity Accuracy)": f"{qty_acc:.2f}%",
            "整单正确率 (Exact Match Rate)": f"{exact_rate:.2f}%",
            "recall_value": recall_rate,
            "model_accuracy_value": model_acc,
            "quantity_accuracy_value": qty_acc,
            "exact_match_value": exact_rate,
        },
        "review_anomalies_detected": len(result.review_items),
        "mismatches": mismatches,
    }
