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
    pdf_path = REF_DIR / "3-C26368.pdf"
    if not pdf_path.is_file():
        pytest.skip("本地参考图纸未提供")
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
    """Exercise both real drawings offline without treating reference row order as identity."""
    required = ["3-C26368.pdf", "3-C26368 bom20261102.xlsx",
                "4-双钱L9_20261020.pdf", "4-双钱L9_部件清单_20261020.xlsx"]
    if any(not (REF_DIR / name).is_file() for name in required):
        pytest.skip("本地参考图纸或核对清单未提供")
    for index, (pdf, gt, mode, page_count) in enumerate([
        ("3-C26368.pdf", "3-C26368 bom20261102.xlsx", "SCHEMATIC_MODE", 20),
        ("4-双钱L9_20261020.pdf", "4-双钱L9_部件清单_20261020.xlsx", "SCHEMATIC_MODE", 30),
    ]):
        result = run_extraction_pipeline(REF_DIR/pdf, output_dir=tmp_path/f"project{index}", project_id=f"test_{index}")
        evaluation = evaluate_extraction(result, REF_DIR/gt)
        assert result.extraction_mode == mode
        assert result.page_count == page_count
        assert len(result.itemized_bom) > 50
        assert Path(result.summary["excel_path"]).exists()
        assert Path(result.summary["json_path"]).exists()
        assert evaluation["total_predicted_items"] > 0
        assert evaluation["recalled_items"] + evaluation["missing_items"] == evaluation["total_ground_truth_items"]
        assert evaluation["extra_items"] == sum(m["kind"] == "EXTRA" for m in evaluation["mismatches"])
        assert evaluation["whole_bom_exact"] == (not evaluation["mismatches"])
        if index == 0:
            meters = [i for i in result.aggregated_bom if i.model == 'NHR-1100C-27-X/2/P-A']
            assert len(meters) == 1 and meters[0].quantity == 5
            assert set(meters[0].tags) == {'-TI1', '-LI1', '-PI1', '-PI2', '-FI1'}
            assert any(it.tag == "-QF0" and "EZD250" in it.model for it in result.itemized_bom)
            assert any(it.tag == "-KT1" for it in result.itemized_bom)
            assert not any(it.source_type == "TOPOLOGY_INFERRED" for it in result.itemized_bom)
        else:
            assert any(it.tag.endswith("-CPU1511") for it in result.declared_parts)
            assert result.summary['independent_analysis']['excluded_table_pages']==list(range(21,29))
            assert result.reconciliation
            assert all(e.role!='table_row' for it in result.independent_bom for e in it.evidences)


def test_web_workbench_starts_without_preseeded_projects():
    """Fresh installation must not depend on the developer's cached sample projects."""
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as client:
        assert client.get("/api/projects").json() == []
        assert client.get("/").status_code == 200
        assert client.get("/api/projects/missing").status_code == 404


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

    pdf_path = REF_DIR / "3-C26368.pdf"
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
        assert {c["tag"] for c in topo["candidates"]} >= {"-QF3", "-QF4", "-QF5", "-KM3", "-KM4", "-KM5"}
        assert len(topo["branches"]) >= 3
        prompt = SoMVisualPromptBuilder.build_actor_vlm_prompt(topo)
        assert "[#1]" in prompt and "-QF3" in prompt
    finally:
        doc.close()

    # 3. Verify YOLOv11 self-evolution dataset status
    yolo_status = YOLOSelfEvolver.get_dataset_status()
    assert yolo_status["image_count"] == 0
    assert yolo_status["total_bounding_boxes"] == 0
    assert yolo_status["training_ready"] is False


