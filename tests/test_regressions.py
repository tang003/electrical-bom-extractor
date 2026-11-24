from __future__ import annotations
import asyncio
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import openpyxl
import pymupdf
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

import backend.app as backend
from bom_extractor.models import BOMItem, EvidenceLocation, ExtractionResult, PageInfo, ReviewItem
from bom_extractor.rules import apply_engineering_rules, build_aggregated_from_itemized, load_rules_config
from bom_extractor.exporter import export_to_excel, export_to_json
from bom_extractor.evaluator import evaluate_extraction
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.vision import VisionRecognizer
from bom_extractor.yolo_evolver import YOLOSelfEvolver
from bom_extractor.schematic_extractor import extract_from_schematic_pdf
from bom_extractor.part_list_extractor import extract_from_part_list_pdf


def result(items=None, mode="SCHEMATIC_MODE", project="proj_test"):
    items = items if items is not None else [BOMItem("detail_1", 1, "-KM1", tags=["-KM1"], name="Relay", model="M1", brand="B", quantity=1)]
    return ExtractionResult(project, "drawing.pdf", "test", "test", mode, 1,
        pages=[PageInfo(1,"CONTROL_CIRCUIT","control",width=600,height=400)], itemized_bom=items,
        aggregated_bom=build_aggregated_from_itemized(items))


def register(res):
    backend.store.results[res.project_id] = res
    return res.project_id


def create_pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        doc.save(path)
    return path


def test_restart_restores_all_manual_state(tmp_path, monkeypatch):
    res = result()
    res.itemized_bom[0].quantity = 5
    res.itemized_bom[0].model = "MANUAL-MODEL"
    res.itemized_bom[0].evidences = [EvidenceLocation(1,[60,40,120,80],[.1,.1,.2,.2],role="manual_box",verified=True,symbol_class="contactor_contact")]
    res.review_items = [ReviewItem("review", "WARNING", "PARAM_MISMATCH", "-KM1", "title", "desc", "suggestion", 1, status="ACCEPTED")]
    pid = register(res)
    backend.store.pdf_paths[pid] = create_pdf(backend.OUTPUT_DIR / pid / "source.pdf")
    backend.store.sync_and_persist(pid)
    extract = Mock(side_effect=AssertionError("must not re-extract saved state"))
    monkeypatch.setattr(backend, "run_extraction_pipeline", extract)
    restored = backend.ProjectStore()
    restored.initialize_defaults()
    loaded = restored.results[pid]
    assert loaded.itemized_bom[0].quantity == 5
    assert loaded.itemized_bom[0].model == "MANUAL-MODEL"
    assert loaded.itemized_bom[0].evidences[0].symbol_class == "contactor_contact"
    assert loaded.review_items[0].status == "ACCEPTED"
    assert not extract.called


def test_corrupt_saved_state_is_not_overwritten(monkeypatch):
    folder = backend.OUTPUT_DIR / "proj_broken"
    create_pdf(folder / "source.pdf")
    state = folder / "project_state.json"
    state.write_text("broken json",encoding="utf-8")
    extract = Mock()
    monkeypatch.setattr(backend,"run_extraction_pipeline",extract)
    backend.store.initialize_defaults()
    assert state.read_text() == "broken json"
    assert "proj_broken" in backend.store.load_errors
    assert not extract.called


def test_legacy_summary_quantity_is_preserved_on_migration():
    saved = result()
    saved.aggregated_bom[0].quantity = 5
    folder = backend.OUTPUT_DIR / saved.project_id
    export_to_json(saved, folder / "drawing_提取结果.json")
    backend.store.initialize_defaults()
    loaded = backend.store.results[saved.project_id]
    assert sum(it.quantity for it in loaded.itemized_bom) == 5
    assert loaded.aggregated_bom[0].quantity == 5
    backend.store.sync_and_persist(saved.project_id)
    assert loaded.aggregated_bom[0].quantity == 5


def test_failed_import_with_partial_json_is_not_restored():
    draft = result()
    folder = backend.OUTPUT_DIR / draft.project_id
    export_to_json(draft, folder / "drawing_提取结果.json")
    (folder / "import_failure.json").write_text('{"message":"unfinished import"}', encoding="utf8")
    backend.store.initialize_defaults()
    assert draft.project_id not in backend.store.results
    assert backend.store.load_errors[draft.project_id] == "unfinished import"
    export_to_json(draft, folder / "project_state.json")
    backend.store.initialize_defaults()
    assert draft.project_id in backend.store.results
    assert not backend.store.load_errors


def test_restoring_zero_quantity_and_editing_accessory_retains_source_fields():
    item = BOMItem("base", 1, "-K1", model="BASE-A", quantity=0, entity_role="accessory", parent_tag="-K1")
    res = result([])
    res.filtered_items.append(item)
    pid = register(res)
    backend.restore_filtered_item(pid, item.id)
    assert res.itemized_bom[0].quantity == 0
    with TestClient(backend.app) as client:
        edit = {"view_mode":"itemized", "id":item.id, "entity_role":"accessory", "quantity":3}
        response = client.put(f"/api/projects/{pid}/bom/item", json=edit)
        assert response.status_code == 200
        row = response.json()["itemized_bom"][0]
        assert (row["model"], row["entity_role"], row["parent_tag"], row["quantity"]) == ("BASE-A", "accessory", "-K1", 3)


def test_download_keeps_snapshot_until_stream_finishes():
    res = result()
    pid = register(res)
    backend.store.sync_and_persist(pid)
    download = backend.download_json(pid)
    try:
        backend.update_bom_item(pid, backend.BOMItemUpdatePayload(id="detail_1", view_mode="itemized", quantity=7))
        def quantity(path):
            return json.loads(Path(path).read_text(encoding="utf8"))["itemized_bom"][0]["quantity"]
        assert quantity(download.path) == 1
        assert quantity(res.summary["json_path"]) == 7
    finally:
        asyncio.run(download.background())
    assert not Path(download.path).exists()


def test_excel_lock_does_not_lose_saved_changes(monkeypatch):
    res = result()
    pid = register(res)
    res.itemized_bom[0].quantity = 8
    monkeypatch.setattr(backend,"export_to_excel",Mock(side_effect=PermissionError("Excel open")))
    backend.store.sync_and_persist(pid)
    saved = json.loads((backend.OUTPUT_DIR/pid/"project_state.json").read_text(encoding="utf-8"))
    assert saved["itemized_bom"][0]["quantity"] == 8
    assert saved["summary"]["export_warning"]


@pytest.mark.parametrize("mode", ["SCHEMATIC_MODE","PART_LIST_MODE"])
def test_summary_quantity_and_fields_survive_export_and_restart(mode):
    a = BOMItem("one",1,"-K1",tags=["-K1"],brand="B",name="Relay",model="M1")
    b = BOMItem("two",2,"-K2",tags=["-K2"],brand="B",name="Relay",model="M1")
    pid = register(result([a,b],mode))
    row = backend.store.results[pid].aggregated_bom[0]
    response = backend.update_bom_item(pid,backend.BOMItemUpdatePayload(view_mode="aggregated",id=row.id,
        tag=row.tag,brand="B2",name="Relay",model="M2",unit="个",quantity=5,remark="checked"))
    assert response["aggregated_bom"][0]["quantity"] == 5
    assert sum(it.quantity for it in backend.store.results[pid].itemized_bom) == 5
    before = copy.deepcopy(response)
    backend.download_excel(pid)
    backend.download_json(pid)
    assert backend.get_project(pid) == before
    restored = backend.ProjectStore()
    restored.initialize_defaults()
    assert restored.results[pid].aggregated_bom[0].quantity == 5
    assert all(it.model == "M2" for it in restored.results[pid].itemized_bom)


def test_decreased_summary_quantity_and_http_download():
    items=[BOMItem(f"item{i}",i,f"-K{i}",tags=[f"-K{i}"],brand="B",name="Relay",model="M1") for i in range(1,5)]
    res=result(items)
    pid=register(res)
    row=res.aggregated_bom[0]
    with TestClient(backend.app) as client:
        response=client.put(f"/api/projects/{pid}/bom/item",json={"id":row.id,"view_mode":"aggregated",
            "tag":row.tag,"brand":"B","name":"Relay","model":"M1","unit":"个","quantity":1})
        assert response.status_code == 200
        assert response.json()["aggregated_bom"][0]["quantity"] == 1
        response=client.get(f"/api/projects/{pid}/export/excel")
        assert response.status_code == 200
        wb=openpyxl.load_workbook(io.BytesIO(response.content),data_only=True)
        assert wb['物料清单']["E7"].value == 1
        wb.close()
        assert client.get(f"/api/projects/{pid}/export/json").json()["aggregated_bom"][0]["quantity"] == 1
    assert [it.quantity for it in items if not it.is_quantity_adjustment] == [1,1,1,1]
    assert sum(it.quantity for it in res.itemized_bom) == 1


def test_ai_and_review_additions_are_retained(monkeypatch):
    res = result()
    pid = register(res)
    backend.store.pdf_paths[pid] = create_pdf(backend.OUTPUT_DIR / pid / "source.pdf")
    monkeypatch.setattr(VisionRecognizer,"extract_page_components_via_vlm",lambda *args,**kwargs:[{"tag":"-K99","name":"Relay","model":"NEW","quantity":1,"norm_bbox":[.1,.1,.2,.2]}])
    out = backend.ai_recognize_page(pid,1)
    assert out["ai_added_count"] == 1
    assert any(it["tag"] == "-K99" for it in out["itemized_bom"])
    rev = ReviewItem("missing","WARNING","MISSING_IN_PART_LIST","-K98","missing","desc","suggestion",1,
        action_type="ADD_ITEM",proposed_data={"tag":"-K98","model":"OTHER"})
    res.review_items.append(rev)
    out = backend.handle_review_action(pid,"missing",backend.ReviewActionPayload(action="ACCEPT"))
    assert any(it["tag"] == "-K98" for it in out["itemized_bom"])
    assert next(r for r in out["review_items"] if r["id"] == "missing")["status"] == "ACCEPTED"


def test_ai_conflicting_same_tag_does_not_duplicate_physical_device(monkeypatch):
    res = result()
    pid = register(res)
    backend.store.pdf_paths[pid] = create_pdf(backend.OUTPUT_DIR/pid/"source.pdf")
    monkeypatch.setattr(VisionRecognizer,"extract_page_components_via_vlm",lambda *a,**k:[{"tag":"-KM1","model":"CONFLICT","norm_bbox":[.1,.1,.2,.2]}])
    response = backend.ai_recognize_page(pid,1)
    assert response["ai_added_count"] == 0
    assert len(res.itemized_bom) == 1
    assert any(r.issue_type == "PARAM_MISMATCH" for r in res.review_items)


def test_invalid_ai_result_does_not_partially_write(monkeypatch):
    res = result()
    pid = register(res)
    backend.store.pdf_paths[pid] = create_pdf(backend.OUTPUT_DIR/pid/"source.pdf")
    monkeypatch.setattr(VisionRecognizer,"extract_page_components_via_vlm",lambda *a,**k:[
        {"tag":"-K20","model":"VALID","norm_bbox":[.1,.1,.2,.2]},
        {"tag":"-K21","model":"BAD","quantity":float("nan"),"norm_bbox":[.1,.1,.2,.2]}])
    with pytest.raises(backend.HTTPException) as exc:
        backend.ai_recognize_page(pid,1)
    assert exc.value.status_code == 502
    assert len(res.itemized_bom) == 1


def test_partial_edit_and_stale_ids_do_not_replace_material_fields():
    res = result()
    pid = register(res)
    with TestClient(backend.app) as client:
        url = f"/api/projects/{pid}/bom/item"
        args = {"id": "detail_1", "view_mode": "itemized", "quantity": 4}
        assert client.put(url, json=args).status_code == 200
        item = res.itemized_bom[0]
        assert (item.model, item.name, item.brand, item.tag, item.tags) == ("M1", "Relay", "B", "-KM1", ["-KM1"])
        assert client.put(url, json={**args, "id": "old_id", "seq_no": 1}).status_code == 404
        stale = {**args, "expected_values": {"quantity": 1}, "quantity": 9}
        assert client.put(url, json=stale).status_code == 409
        assert item.quantity == 4


def test_clearing_adjusted_quantity_persists_unknown_without_adjustment():
    res = result()
    pid = register(res)
    def update(quantity):
        row = res.aggregated_bom[0]
        return backend.update_bom_item(pid, backend.BOMItemUpdatePayload(id=row.id, tag=row.tag, quantity=quantity))
    update(5)
    update(None)
    restored = backend.ProjectStore()
    restored.initialize_defaults()
    saved = restored.results[pid]
    assert [(i.id, i.quantity) for i in saved.itemized_bom] == [("detail_1", None)]
    assert saved.aggregated_bom[0].quantity is None


def test_reaccept_after_reset_and_restart_keeps_manually_edited_addition():
    res = result()
    pid = register(res)
    res.review_items.append(ReviewItem("missing", "WARNING", "MISSING_IN_PART_LIST", "-K2",
                                       "Missing", "", "", 1, action_type="ADD_ITEM", proposed_data={"model": "A"}))
    def act(action):
        return backend.handle_review_action(pid, "missing", backend.ReviewActionPayload(action=action))
    act("ACCEPT")
    added = res.itemized_bom[-1]
    backend.update_bom_item(pid, backend.BOMItemUpdatePayload(id=added.id, view_mode="itemized", model="MANUAL"))
    act("RESET")
    backend.store.initialize_defaults()
    out = act("ACCEPT")
    assert [i["model"] for i in out["itemized_bom"]] == ["M1", "MANUAL"]


def test_manual_binding_targets_only_one_detail():
    a = BOMItem("a",1,"-K1",tags=["-K1"],model="SAME")
    b = BOMItem("b",2,"-K2",tags=["-K2"],model="SAME")
    res = result([a,b])
    pid = register(res)
    payload = backend.BindBoxPayload(page=1,norm_bbox=[.1,.1,.2,.2],target_tag="-K2",symbol_class="contactor_contact")
    backend.bind_box_to_bom_item(pid,res.aggregated_bom[0].id,payload)
    assert not a.evidences
    assert len(b.evidences) == 1
    assert b.evidences[0].verified and b.evidences[0].symbol_class == "contactor_contact"


def test_summary_delete_does_not_delete_same_tag_accessory():
    main = BOMItem("main",1,"=02-K1",tags=["=02-K1"],model="COIL")
    base = BOMItem("base",2,"=02-K1",tags=["=02-K1"],model="BASE")
    res = result([main,base],"PART_LIST_MODE")
    pid = register(res)
    backend.delete_bom_item(pid,res.aggregated_bom[0].id)
    assert [it.id for it in res.itemized_bom] == ["base"]


def test_review_update_only_changes_selected_same_tag_part():
    main=BOMItem("main",1,"=02-K1",tags=["=02-K1"],model="COIL",brand="B")
    base=BOMItem("base",2,"=02-K1",tags=["=02-K1"],model="BASE",brand="B")
    res=result([main,base],"PART_LIST_MODE")
    res.review_items=[ReviewItem("fix","WARNING","PARAM_MISMATCH","=02-K1","fix","desc","suggestion",1,
        action_type="UPDATE_ITEM",proposed_data={"target_seq_no":2,"model":"CORRECT-BASE"})]
    pid=register(res)
    backend.handle_review_action(pid,"fix",backend.ReviewActionPayload(action="ACCEPT"))
    assert main.model == "COIL" and base.model == "CORRECT-BASE"


def test_explicit_reextract_keeps_manual_snapshot(monkeypatch):
    res=result()
    res.itemized_bom[0].model="MANUAL"
    pid=register(res)
    backend.store.pdf_paths[pid]=create_pdf(backend.OUTPUT_DIR/pid/"source.pdf")
    monkeypatch.setattr(backend,"run_extraction_pipeline",lambda **kw:result(project=pid))
    response=backend.reextract_project(pid)
    saved=Path(response["summary"]["previous_state_path"])
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["itemized_bom"][0]["model"] == "MANUAL"


def test_key_is_private_and_blank_update_preserves_it():
    VisionRecognizer.save_config({"api_key":"test-secret-value","model":"first"})
    with TestClient(backend.app) as client:
        response = client.get("/api/vlm-config")
        assert "test-secret-value" not in response.text
        assert response.json()["has_api_key"] is True
        response = client.put("/api/vlm-config",json={"api_key":"","model":"changed"})
        assert "test-secret-value" not in response.text
        assert VisionRecognizer().api_key == "test-secret-value"
        assert "test-secret-value" not in VisionRecognizer.CONFIG_FILE.read_text(encoding="utf-8")
        assert client.put("/api/vlm-config",json={},headers={"Origin":"https://example.com"}).status_code == 403
        assert client.get("/api/vlm-config",headers={"Host":"evil.example"}).status_code == 400


def test_legacy_key_is_migrated_and_disabled_until_replaced():
    VisionRecognizer.CONFIG_FILE.write_text(json.dumps({"api_key":"old-exposed-key"}),encoding="utf-8")
    cfg=VisionRecognizer.load_saved_config()
    assert cfg["credential_rotation_required"]
    assert "old-exposed-key" not in VisionRecognizer.CONFIG_FILE.read_text(encoding="utf-8")
    assert VisionRecognizer().api_key == ""
    VisionRecognizer.save_config({"api_key":"replacement-key"})
    assert not VisionRecognizer.load_saved_config()["credential_rotation_required"]
    assert VisionRecognizer().api_key == "replacement-key"


def test_upload_names_cannot_escape_or_overwrite(tmp_path,monkeypatch):
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    monkeypatch.setattr(backend,"run_extraction_pipeline",lambda *a,**k:result(project=k["project_id"]))
    outside = tmp_path / "outside.pdf"
    ids=[]
    for _ in range(2):
        uploaded = UploadFile(io.BytesIO(data),filename=str(outside))
        out = asyncio.run(backend.upload_pdf(uploaded))
        ids.append(out["project_id"])
        saved = backend.store.pdf_paths[ids[-1]]
        assert saved.resolve().is_relative_to(backend.OUTPUT_DIR.resolve())
    assert not outside.exists()
    assert ids[0] != ids[1]


def test_eplan_identifiers_stay_text_in_excel(tmp_path):
    res=result([BOMItem("x",1,"=02-K1",tags=["=02-K1"],name="=literal",model="MODEL")],"PART_LIST_MODE")
    output=tmp_path/"bom.xlsx"
    export_to_excel(res,output)
    with open(output,"rb") as stream:
        workbook=openpyxl.load_workbook(stream,data_only=False)
        assert workbook['物料清单']["G7"].data_type == "s"
        assert workbook['物料清单']["G7"].value == "=02-K1"
        workbook.close()
    workbook=openpyxl.load_workbook(output,data_only=True)
    assert workbook['物料清单']["G7"].value == "=02-K1"
    workbook.close()


@pytest.mark.parametrize("rotation",[0,90,180,270])
def test_ocr_coordinates_use_visual_dimensions(rotation,monkeypatch):
    def fake_ocr(img,**kwargs):
        h,w=img.shape[:2]
        return [[[[.75*w,.2*h],[.85*w,.2*h],[.85*w,.3*h],[.75*w,.3*h]],"-QF1",.99]],None
    monkeypatch.setattr(DualChannelOCREngine,"get_ocr",lambda:fake_ocr)
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        page.set_rotation(rotation)
        spans=DualChannelOCREngine.extract_dual_channel_spans(page,[],dpi=72)
        assert spans[0]["norm_bbox"] == [.75,.2,.85,.3]
        assert spans[0]["cx"] == pytest.approx(page.rect.width*.8,abs=.01)


def test_blank_vector_page_never_fabricates_cabinet(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,"ocr_pdf_region",lambda *a,**k:("",[]))
    fake_page=SimpleNamespace(number=0,get_drawings=lambda:[{}]*501)
    findings=VisionRecognizer().analyze_vector_regions_on_page(fake_page,[])
    assert findings and all(f["type"] != "CABINET_DIMENSION" for f in findings)
    assert "1550" not in json.dumps(findings)


def schematic_texts(monkeypatch, texts):
    monkeypatch.setattr(DualChannelOCREngine,"extract_dual_channel_spans",lambda page,spans,**k:spans)
    monkeypatch.setattr(VisionRecognizer,"analyze_vector_regions_on_page",lambda *args:[])
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=400)
        for point,text in texts:
            page.insert_text(point,text,fontsize=10)
        pages=[PageInfo(1,"MAIN_CIRCUIT","main",width=600,height=400)]
        return extract_from_schematic_pdf(doc,pages,load_rules_config())


def test_unconnected_peers_do_not_inherit_ratings(monkeypatch):
    items,reviews=schematic_texts(monkeypatch,[((100,100),"-KM1"),((100,120),"LC1D50M7C"),((450,100),"-KM2")])
    peer=next(it for it in items if it.tag == "-KM2")
    assert peer.model == "未提取到型号(待核实)" and peer.needs_review
    assert peer.source_type != "TOPOLOGY_INFERRED"


def test_unresolved_timers_are_not_lost(monkeypatch):
    items,reviews=schematic_texts(monkeypatch,[((100,100),"-KT1"),((300,100),"-KT2"),((450,100),"-KT3")])
    assert {it.tag for it in items} >= {"-KT1","-KT2","-KT3"}
    assert {r.tag for r in reviews} >= {"-KT1","-KT2","-KT3"}


def test_generic_part_list_cross_check_detects_fu3(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,"extract_dual_channel_spans",lambda page,spans,**k:spans)
    with pymupdf.open() as doc:
        doc.new_page(width=1190.55,height=841.89).insert_text((200,200),"-FU3")
        page=doc.new_page(width=1190.55,height=841.89)
        for x,text in [(60,"=02-FU1"),(230,"1"),(300,"Fuse"),(600,"RT18-32"),(820,"CHNT")]:
            page.insert_text((x,200),text)
        pages=[PageInfo(1,"CONTROL_CIRCUIT","control"),PageInfo(2,"PART_LIST","list")]
        items,reviews=extract_from_part_list_pdf(doc,pages)
        assert any(it.tag == "=02-FU1" for it in items)
        assert any(r.tag == "-FU3" and r.issue_type == "MISSING_IN_PART_LIST" for r in reviews)


def test_scanned_part_list_can_be_classified(monkeypatch):
    from bom_extractor.classifier import classify_pdf_pages
    monkeypatch.setattr(DualChannelOCREngine,"extract_dual_channel_spans",lambda *a,**kw:[{"text":t} for t in ("部件列表","设备标识符","订货编号")])
    with pymupdf.open() as doc:
        doc.new_page()
        mode,_,_,pages=classify_pdf_pages(doc)
        assert mode == "PART_LIST_MODE" and pages[0].page_type == "PART_LIST"


@pytest.mark.parametrize("mode",["SCHEMATIC_MODE","PART_LIST_MODE"])
def test_plc_accessories_are_idempotent_suggestions_without_changing_drawing_model(mode):
    item=BOMItem("cpu",1,"-CPU1",model="6ES7 511-1AL03-0AB0")
    first=apply_engineering_rules([item],[],mode,load_rules_config())[0]
    second=apply_engineering_rules(first,[],mode,load_rules_config())[0]
    assert first[0].model==item.model
    assert "6ES7590-1AB60-0AA0" in first[0].selection_suggestions[0]['model']
    assert first[0].model == second[0].model
    assert len(second[0].selection_suggestions)==1


def ground_truth(path,rows):
    wb=openpyxl.Workbook()
    wb.active.append(["序号","品牌","名称","型号","单位","数量"])
    for row in rows: wb.active.append(row)
    wb.save(path)
    wb.close()
    return path


def test_evaluation_ignores_order_and_reports_extra_items(tmp_path):
    gt=ground_truth(tmp_path/"gt.xlsx",[[1,"B","Relay","M1","个",1],[2,"B","Other","M2","个",2]])
    res=result([BOMItem("a",99,"",brand="B",name="Other",model="M2",quantity=2),BOMItem("b",98,"",brand="B",name="Relay",model="M1")])
    assert evaluate_extraction(res,gt)["whole_bom_exact"]
    res.aggregated_bom.append(BOMItem("extra",3,"-K9",model="BAD",quantity=999))
    evaluation=evaluate_extraction(res,gt)
    assert evaluation["extra_items"] == 1
    assert evaluation["metrics"]["precision_value"] < 100
    assert evaluation["metrics"]["exact_match_value"] < 100
    assert not evaluation["whole_bom_exact"]


def test_evaluation_does_not_reuse_one_prediction(tmp_path):
    gt=ground_truth(tmp_path/"gt.xlsx",[[1,"B","Relay","M1","个",1],[2,"B","Relay","M1","个",1]])
    evaluation=evaluate_extraction(result(),gt)
    assert evaluation["recalled_items"] == 1
    assert evaluation["missing_items"] == 1


def test_evaluation_checks_scoped_tags_units_and_unknown_quantities(tmp_path):
    path = tmp_path / "reference.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append(["序号", "设备标识符", "名称/描述", "订货编号", "单位", "数量"])
    wb.active.append([1, "+A-K1", "Relay", "M1", "个", 0])
    wb.active.append([2, "+B-K1", "Relay", "M1", "套", None])
    wb.save(path)
    wb.close()
    first = BOMItem("a", 1, "+A-K1", name="Relay", model="M1", unit="个", quantity=0)
    second = BOMItem("b", 2, "+B-K1", name="Relay", model="M1", unit="套", quantity=None)
    res = result([second, first])
    assert evaluate_extraction(res, path)["whole_bom_exact"]
    second.tag = "+C-K1"
    second.tags = []
    second.quantity = 0
    second.unit = "个"
    difference, = evaluate_extraction(res, path)["mismatches"]
    assert difference["seq_no"] == 2
    assert not any(difference[key] for key in ("tag_ok", "unit_ok", "qty_ok"))


def test_evaluation_rejects_empty_reference(tmp_path):
    path = ground_truth(tmp_path / "empty.xlsx", [])
    with pytest.raises(ValueError, match="没有物料行"):
        evaluate_extraction(result(), path)


def test_yolo_verified_types_project_split_and_stale_cleanup(tmp_path):
    pdf=create_pdf(tmp_path/"source.pdf")
    ev=EvidenceLocation(1,[60,40,120,80],[.1,.1,.2,.2],role="contact",verified=True,symbol_class="contactor_contact")
    item=BOMItem("relay",1,"-KM1",tags=["-KM1"],evidences=[ev],needs_review=True)
    res=result([item])
    assert YOLOSelfEvolver.archive_project_to_yolo_dataset(res,pdf)["newly_archived_boxes"] == 0
    item.needs_review=False
    status=YOLOSelfEvolver.archive_project_to_yolo_dataset(res,pdf)
    assert status["class_distribution"]["contactor_contact"] == 1
    assert status["class_distribution"]["contactor_coil"] == 0
    label=next((YOLOSelfEvolver.DATASET_DIR/"labels/train").glob("*.txt"))
    other=copy.deepcopy(res)
    other.project_id="proj_other"
    YOLOSelfEvolver.archive_project_to_yolo_dataset(other,pdf)
    train=set((YOLOSelfEvolver.DATASET_DIR/"train.txt").read_text().splitlines())
    val=set((YOLOSelfEvolver.DATASET_DIR/"val.txt").read_text().splitlines())
    assert train and val and not train.intersection(val)
    res.itemized_bom=[]
    YOLOSelfEvolver.archive_project_to_yolo_dataset(res,pdf)
    assert not label.exists()
