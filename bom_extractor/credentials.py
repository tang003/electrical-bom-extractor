"""Per-user credentials kept outside the source tree (DPAPI protected on Windows)."""
from __future__ import annotations

import base64
import ctypes
import json
import os
import tempfile
from pathlib import Path

SECRET_FILE = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config")) / "ElectricalBOM" / "credentials.json"


def _dpapi(value: bytes, decrypt: bool = False) -> bytes:
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]

    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    output = Blob()
    if decrypt:
        ok = ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output))
    else:
        ok = ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output))
    if not ok:
        raise OSError("无法访问当前用户的加密凭据")
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        ctypes.windll.kernel32.LocalFree(output.data)


def read_api_key() -> str:
    if not SECRET_FILE.exists():
        return ""
    saved = json.loads(SECRET_FILE.read_text(encoding="utf-8"))
    value = base64.b64decode(saved["value"])
    if saved["storage"] == "windows-dpapi":
        value = _dpapi(value, decrypt=True)
    return value.decode("utf-8")


def write_api_key(key: str) -> None:
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    value = key.encode("utf-8")
    storage = "windows-dpapi" if os.name == "nt" else "user-file"
    if os.name == "nt":
        value = _dpapi(value)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=SECRET_FILE.parent, delete=False) as f:
        temp_path = Path(f.name)
        os.chmod(temp_path, 0o600)
        json.dump({"storage": storage, "value": base64.b64encode(value).decode("ascii")}, f)
    try:
        os.replace(temp_path, SECRET_FILE)
    finally:
        temp_path.unlink(missing_ok=True)
