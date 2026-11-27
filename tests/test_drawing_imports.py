import threading
import time

import pymupdf
from fastapi.testclient import TestClient

import backend.app as backend


def wait_for_drawings(client, predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get("/api/drawings")
        assert response.status_code == 200, response.text
        data = response.json()
        if predicate(data):
            return data
        time.sleep(0.025)
    raise AssertionError(f"Expected drawing state, received {data}")


def test_queued_upload_keeps_running_after_uploader_closes(monkeypatch):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((48, 80), "Electrical schematic - QF1")
        drawing = pdf.tobytes()

    started = threading.Event()
    proceed = threading.Event()
    calls = []
    extract = backend._extract_uploaded_drawing

    def delayed_extract(*args):
        calls.append(args[2])
        started.set()
        if not proceed.wait(timeout=10):
            raise TimeoutError("Import test did not release the worker")
        return extract(*args)

    monkeypatch.setattr(backend, "_extract_uploaded_drawing", delayed_extract)
    with TestClient(backend.app) as later_visit:
        try:
            home = later_visit.get("/", follow_redirects=False)
            assert home.headers["location"] == "/projects"
            uploader = TestClient(backend.app)
            response = uploader.post(
                "/api/drawing-imports",
                files={"file": ("drawing.pdf", drawing, "application/pdf")},
                data={"workflow_mode": "TABLE_EXTRACT"},
            )
            uploader.close()
            assert response.status_code == 202, response.text
            project_id = response.json()["project_id"]
            assert started.wait(timeout=5)
            data = later_visit.get("/api/drawings").json()
            assert data["projects"] == []
            assert [(job["project_id"], job["status"]) for job in data["imports"]] == [(project_id, "running")]
            url = f"/api/projects/{project_id}"
            assert later_visit.get(url).status_code == 404
        finally:
            proceed.set()
        data = wait_for_drawings(later_visit, lambda data: data["projects"])
        assert data["projects"][0]["project_id"] == project_id
        assert data["imports"] == []
        first_open = later_visit.get(url).json()
        assert first_open["summary"]["workflow_mode"] == "TABLE_EXTRACT"
        assert later_visit.get(url + "/page-references").json()["links"] == []
        assert later_visit.get(url).json() == first_open
        assert (backend.OUTPUT_DIR / project_id / "project_state.json").is_file()
        assert calls == [project_id]


def test_invalid_drawing_remains_a_failed_import_after_restart(monkeypatch):
    with TestClient(backend.app) as client:
        response = client.post("/api/drawing-imports", files={"file": ("broken.pdf", b"broken")})
        assert response.status_code == 202, response.text
        failed = wait_for_drawings(client, lambda data: data["imports"][0]["status"] == "failed")
        assert failed["projects"] == []
        assert "解析失败" in failed["imports"][0]["message"]
    monkeypatch.setattr(backend, "store", backend.ProjectStore())
    with TestClient(backend.app) as client:
        assert client.get("/api/drawings").json() == failed
        assert client.get("/api/projects").json() == []
