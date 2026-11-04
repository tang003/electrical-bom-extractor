from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Resolve workspace root by searching upwards for bom_extractor package
SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = next((p for p in SCRIPT_PATH.parents if (p / "bom_extractor").is_dir()), SCRIPT_PATH.parents[3])
sys.path.insert(0, str(ROOT_DIR))

from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.evaluator import evaluate_extraction


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Industrial Electrical Schematic BOM Extractor CLI")
    parser.add_argument("--pdf", required=True, help="Path to input electrical schematic PDF")
    parser.add_argument("--output-dir", default=None, help="Directory for intermediate and final outputs")
    parser.add_argument("--rules", default=None, help="Optional custom rules_config.json path")
    parser.add_argument("--ground-truth", default=None, help="Optional ground-truth Excel BOM for accuracy evaluation")
    args = parser.parse_args()

    result = run_extraction_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        rules_config_path=args.rules,
    )

    print(f"[OK] Project       : {result.project_name} ({result.drawing_no})")
    print(f"[OK] Mode          : {result.summary['extraction_mode_label']}")
    print(f"[OK] Pages         : {result.page_count}")
    print(f"[OK] Itemized Rows : {len(result.itemized_bom)}")
    print(f"[OK] Aggregated BOM: {len(result.aggregated_bom)}")
    print(f"[OK] Review Items  : {len(result.review_items)}")
    print(f"[OK] Excel Output  : {result.summary['excel_path']}")
    print(f"[OK] JSON Output   : {result.summary['json_path']}")

    if args.ground_truth:
        eval_res = evaluate_extraction(result, args.ground_truth)
        print("-" * 60)
        print("[EVALUATION AGAINST GROUND TRUTH]")
        for k, v in eval_res["metrics"].items():
            if not k.endswith("_value"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
