from __future__ import annotations
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.evaluator import evaluate_extraction


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ref_dir = ROOT_DIR / "参考文件"
    out_root = ROOT_DIR / "output"

    benchmarks = [
        {
            "name": "测试集 1：C26368（纯原理图模式，无自带部件列表，含星三角对称推导与CAD曲线文字）",
            "pdf": ref_dir / "C26368.pdf",
            "gt_excel": ref_dir / "C26368 bom20261102.xlsx",
            "proj_id": "C26368",
        },
        {
            "name": "测试集 2：双钱L9_20261020（EPLAN自带部件列表模式 + 原理图交叉校验 + 工程清洗）",
            "pdf": ref_dir / "双钱L9_20261020.pdf",
            "gt_excel": ref_dir / "双钱L9_部件清单_20261020.xlsx",
            "proj_id": "ShuangQian_L9",
        },
    ]

    all_reports = []
    for bm in benchmarks:
        print("=" * 80)
        print(f"正在运行：{bm['name']}")
        print(f"输入 PDF ：{bm['pdf']}")
        print(f"标准 BOM ：{bm['gt_excel']}")
        res = run_extraction_pipeline(
            pdf_path=bm["pdf"],
            output_dir=out_root / bm["proj_id"],
            project_id=bm["proj_id"],
        )
        eval_report = evaluate_extraction(res, bm["gt_excel"])
        all_reports.append(eval_report)

        print(f"识别模式 ：{res.summary['extraction_mode_label']}")
        print(f"导出 Excel：{res.summary['excel_path']}")
        print(f"导出 JSON ：{res.summary['json_path']}")
        print("-" * 40)
        for k, v in eval_report["metrics"].items():
            if not k.endswith("_value"):
                print(f"  {k}: {v}")
        print(f"  检出待人工审核/原图异常项: {len(res.review_items)} 项")
        for rev in res.review_items:
            print(f"    - [{rev.severity}] {rev.title} (P{rev.page})")
        if eval_report["mismatches"]:
            print(f"  差异明细 ({len(eval_report['mismatches'])} 项):")
            for m in eval_report["mismatches"][:5]:
                print("   ", json.dumps(m, ensure_ascii=False))
        else:
            print("  差异明细: 0 项（与人工标准 BOM 100% 完全一致！）")

    report_path = out_root / "benchmark_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print("=" * 80)
    print(f"全部基准评测完成！评测报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
