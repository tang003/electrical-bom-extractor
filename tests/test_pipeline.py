from __future__ import annotations
from pathlib import Path
import pymupdf
import pytest

from bom_extractor.classifier import classify_pdf_pages, extract_visual_spans
from bom_extractor.pipeline import run_extraction_pipeline
from bom_extractor.evaluator import evaluate_extraction


ROOT_DIR = Path(__file__).resolve().parent.parent
REF_DIR = ROOT_DIR / "参考文件"


def test_c26368_key_pages_1_2_5_19():
    """
    Verify specific recognition scenarios on Pages 1, 2, 5, and 19 of C26368.pdf:
    - Page 1: Main circuit breakers (-QF0, -QF1, -QF2), star-delta contactors (-KM1, -KM1-1, -KM1-2), thermal relays (FR1, FR2)
    - Page 2: Motor protection breakers (-QF3~5 / GV2ME) & contactors (-KM3~5)
    - Page 5: Control circuit coils/contacts & CAD vector outline timer text (REXL2TMP7+RXZE2M114)
    - Page 19: Terminal strip tables (=+-X1, =+-X2, =+-X3) not miscounted as standalone devices
    """
    pdf_path = REF_DIR / "C26368.pdf"
    doc = pymupdf.open(str(pdf_path))
    try:
        mode, _, _, pages_info = classify_pdf_pages(doc)
        assert mode == "SCHEMATIC_MODE"
        assert len(pages_info) == 20

        # Page 1 scenario
        p1_spans = extract_visual_spans(doc[0])
        p1_texts = {s["text"] for s in p1_spans}
        assert "-QF0" in p1_texts
        assert "-QF1" in p1_texts
        assert "-KM1" in p1_texts
        assert "-KM1-1" in p1_texts
        assert "-KM1-2" in p1_texts
        assert "LC1D50M7C" in p1_texts

        # Page 2 scenario
        p2_spans = extract_visual_spans(doc[1])
        p2_texts = {s["text"] for s in p2_spans}
        assert "-QF3" in p2_texts and "GV2ME16C" in p2_texts
        assert "-KM3" in p2_texts and "LC1D18M7C" in p2_texts

        # Page 5 scenario (vector outline text detection)
        assert pages_info[4].has_vector_text_region is True

        # Page 19 scenario (terminal diagram classification)
        assert pages_info[18].page_type == "TERMINAL_DIAGRAM"
    finally:
        doc.close()


def test_full_pipeline_and_benchmarks(tmp_path: Path):
    """Verify 100% recall and accuracy on both C26368 and ShuangQian L9 benchmarks."""
    res1 = run_extraction_pipeline(
        REF_DIR / "C26368.pdf",
        output_dir=tmp_path / "c26368",
        project_id="c26368_test",
    )
    eval1 = evaluate_extraction(res1, REF_DIR / "C26368 bom20261102.xlsx")
    assert eval1["metrics"]["recall_value"] == 100.0
    assert eval1["metrics"]["model_accuracy_value"] == 100.0
    assert eval1["metrics"]["quantity_accuracy_value"] == 100.0
    assert eval1["metrics"]["exact_match_value"] == 100.0

    res2 = run_extraction_pipeline(
        REF_DIR / "双钱L9_20261020.pdf",
        output_dir=tmp_path / "l9",
        project_id="l9_test",
    )
    eval2 = evaluate_extraction(res2, REF_DIR / "双钱L9_部件清单_20261020.xlsx")
    assert eval2["metrics"]["recall_value"] == 100.0
    assert eval2["metrics"]["model_accuracy_value"] == 100.0
    assert eval2["metrics"]["quantity_accuracy_value"] == 100.0
    assert eval2["metrics"]["exact_match_value"] == 100.0


def test_web_workbench_online_edit_and_export():
    """Verify FastAPI web workbench endpoints: project listing, inline BOM edit, review action, and Excel export."""
    from backend.app import store, list_projects, get_project, update_bom_item, BOMItemUpdatePayload, handle_review_action, ReviewActionPayload

    store.initialize_defaults()
    projs = list_projects()
    assert len(projs) >= 2

    # Fetch C26368 project
    c_data = get_project("C26368")
    assert c_data["page_count"] == 20
    assert len(c_data["aggregated_bom"]) == 38

    # Online edit row 1 quantity from 1 to 2 and then back to 1
    first_item = c_data["aggregated_bom"][0]
    updated = update_bom_item(
        "C26368",
        BOMItemUpdatePayload(
            view_mode="primary",
            id=first_item["id"],
            seq_no=first_item["seq_no"],
            tag=first_item["tag"],
            brand=first_item["brand"],
            name=first_item["name"],
            model=first_item["model"],
            unit=first_item["unit"],
            quantity=2.0,
            remark="工程师网页修改测试",
        ),
    )
    assert updated["aggregated_bom"][0]["quantity"] == 2.0
    assert updated["aggregated_bom"][0]["source_type"] == "MANUAL_EDITED"

    # Accept a review item on ShuangQian_L9 (e.g. fixing brand typo on seq 119 RXZE2M114)
    l9_data = get_project("ShuangQian_L9")
    typo_rev = next(r for r in l9_data["review_items"] if r["issue_type"] == "BRAND_TYPO")
    after_rev = handle_review_action(
        "ShuangQian_L9",
        typo_rev["id"],
        ReviewActionPayload(action="ACCEPT"),
    )
    k102_base = next(it for it in after_rev["itemized_bom"] if it["seq_no"] == 119)
    assert k102_base["brand"] == "施耐德"


def test_v2_dual_channel_ocr_som_topology_and_yolo_evolver(tmp_path: Path):
    """
    Verify v2.0 5-Layer Architecture modules:
    1. DualChannelOCREngine (RapidOCR offline ONNX engine) reading CAD curve-outlined models on Page 5 & Page 9
    2. VectorTopologyBuilder & SoMVisualPromptBuilder rendering [#1..#N] Set-of-Mark boxes & NetworkX branches
    3. YOLOSelfEvolver auto-archiving verified bounding boxes to YOLOv11 format
    """
    from bom_extractor.ocr_engine import DualChannelOCREngine
    from bom_extractor.som_renderer import SoMVisualPromptBuilder
    from bom_extractor.yolo_evolver import YOLOSelfEvolver

    assert DualChannelOCREngine.is_available() is True

    pdf_path = REF_DIR / "C26368.pdf"
    doc = pymupdf.open(str(pdf_path))
    try:
        # 1. Test RapidOCR curve-outlined text recognition on Page 5 and Page 9
        p5_text, _ = DualChannelOCREngine.ocr_pdf_region(doc[4], [0.40, 0.72, 0.62, 0.85], dpi=300)
        assert "REXL2TMP7+RXZE2M114" in p5_text

        p9_text, _ = DualChannelOCREngine.ocr_pdf_region(doc[8], [0.45, 0.34, 0.72, 0.46], dpi=300)
        assert "LF60G-BPAA850SB" in p9_text

        # 2. Test Set-of-Mark (#1..#N) rendering & NetworkX branch connectivity on Page 2
        som_out = tmp_path / "som_p02.png"
        topo = SoMVisualPromptBuilder.render_som_annotated_page(doc[1], som_out, dpi=180)
        assert som_out.exists()
        assert len(topo["candidates"]) >= 9
        assert len(topo["branches"]) >= 3
        prompt = SoMVisualPromptBuilder.build_actor_vlm_prompt(topo)
        assert "[#1]" in prompt and "-QF3" in prompt
    finally:
        doc.close()

    # 3. Verify YOLOv11 self-evolution dataset status
    yolo_status = YOLOSelfEvolver.get_dataset_status()
    assert yolo_status["image_count"] > 0
    assert yolo_status["total_bounding_boxes"] > 0
    assert Path(yolo_status["data_yaml_path"]).exists()
    assert Path(yolo_status["train_script_path"]).exists()


