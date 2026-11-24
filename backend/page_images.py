"""Render a page once per source revision and serve browser-cacheable images."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import threading

import pymupdf
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response


_RENDER_LOCKS = [threading.Lock() for _ in range(16)]


def page_image(source: Path, cache: Path, page_num: int, dpi: int, request=None):
    if page_num < 1 or not 36 <= dpi <= 300:
        raise HTTPException(400, "页码或分辨率无效")
    stat = source.stat()
    revision = hashlib.sha256(f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    target = cache / f"{revision}_{page_num}_{dpi}.png"
    lock = _RENDER_LOCKS[int(revision, 16) % len(_RENDER_LOCKS)]
    with lock:
        if not target.is_file():
            with pymupdf.open(source) as doc:
                if page_num > len(doc):
                    raise HTTPException(400, "页码或分辨率无效")
                pixmap = doc[page_num - 1].get_pixmap(dpi=dpi, alpha=False)
                cache.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(dir=cache, suffix=".png")
                os.close(descriptor)
                try:
                    pixmap.save(temporary)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
    etag = f'"{revision}-{page_num}-{dpi}"'
    # Revalidate across project re-extraction while avoiding another image transfer.
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request is not None:
        candidates = request.headers.get("if-none-match", "").split(",")
        if any(value.strip().removeprefix("W/") in (etag, "*") for value in candidates):
            return Response(status_code=304, headers=headers)
    return FileResponse(target, media_type="image/png", headers=headers)
