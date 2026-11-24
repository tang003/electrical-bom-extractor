from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_user_data(tmp_path, monkeypatch):
    """Tests never write production state/training data or send a model request."""
    import backend.app as backend
    from bom_extractor import credentials
    from bom_extractor.vision import VisionRecognizer
    from bom_extractor.yolo_evolver import YOLOSelfEvolver
    import requests

    monkeypatch.setattr(backend, "OUTPUT_DIR", tmp_path / "projects")
    monkeypatch.setattr(backend, "store", backend.ProjectStore())
    monkeypatch.setattr(credentials, "SECRET_FILE", tmp_path / "credentials.json")
    monkeypatch.setattr(VisionRecognizer, "CONFIG_FILE", tmp_path / "vlm_config.json")
    monkeypatch.setattr(YOLOSelfEvolver, "DATASET_DIR", tmp_path / "dataset")
    monkeypatch.delenv("VLM_API_KEY", raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("External network calls are prohibited during tests")
    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
