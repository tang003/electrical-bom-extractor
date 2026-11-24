from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

import backend.app as backend
from backend.page_images import page_image


def test_page_image_revalidates_and_changes_when_source_changes(tmp_path):
    source = tmp_path / "source.pdf"
    def write(text):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((40, 40), text)
            doc.save(source)
    write("First revision")
    backend.store.pdf_paths["sample"] = source
    with TestClient(backend.app) as client:
        url = "/api/projects/sample/pages/1/image?dpi=72"
        first = client.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        assert first.headers["cache-control"] == "private, no-cache"
        cached = client.get(url, headers={"If-None-Match": etag})
        assert cached.status_code == 304 and not cached.content
        write("Second revision with changed content")
        updated = client.get(url, headers={"If-None-Match": etag})
        assert updated.status_code == 200
        assert updated.headers["etag"] != etag and updated.content != first.content
        for suffix in ("?dpi=0", "?dpi=9999", "?dpi=-1"):
            assert client.get(url.split("?")[0] + suffix).status_code == 400
        assert client.get("/api/projects/sample/pages/0/image").status_code == 400
        assert client.get("/api/projects/sample/pages/2/image").status_code == 400


def test_concurrent_requests_share_one_complete_cached_image(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(source)
    render = pymupdf.Page.get_pixmap
    calls = []
    def counted(page, **kwargs):
        calls.append(page.number)
        return render(page, **kwargs)
    monkeypatch.setattr(pymupdf.Page, "get_pixmap", counted)
    def request(_):
        response = page_image(source, tmp_path / "cache", 1, 72)
        return Path(response.path).read_bytes()
    with ThreadPoolExecutor(max_workers=4) as executor:
        images = list(executor.map(request, range(4)))
    assert len(calls) == 1
    assert all(data == images[0] and data.startswith(b"\x89PNG") for data in images)
    assert len(list((tmp_path / "cache").glob("*.png"))) == 1
