from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Any
import pymupdf
from .models import ExtractionResult, PageType
from .topology_graph import VectorTopologyBuilder


class YOLOSelfEvolver:
    """Only explicitly confirmed symbol boxes enter managed training manifests."""
    DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "yolo_electrical"

    @classmethod
    def _manifest(cls):
        path = cls.DATASET_DIR / "verified_manifest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @classmethod
    def has_archived_project(cls, project_id):
        return project_id in cls._manifest()

    @classmethod
    def archive_project_to_yolo_dataset(cls, result: ExtractionResult, pdf_path: Path, dpi: int = 160) -> Dict[str, Any]:
        root = cls.DATASET_DIR.resolve()
        images, labels = root / "images" / "train", root / "labels" / "train"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        token = re.sub(r"[^\w-]", "_", result.project_id)[:60] + "_" + hashlib.sha256(result.project_id.encode()).hexdigest()[:12]
        blocked_tags = {tag.strip() for rev in result.review_items if rev.status == "PENDING" for tag in re.split(r"[,，]", rev.tag)}
        page_types = {p.page_num: p.page_type for p in result.pages}
        by_page = {}
        for item in result.itemized_bom:
            if item.needs_review or item.category == "auxiliary" or item.is_quantity_adjustment or item.quantity is None or item.quantity <= 0:
                continue
            if any(tag in blocked_tags for tag in (item.tags or [item.tag])):
                continue
            for ev in item.evidences:
                if not ev.verified or ev.symbol_class not in VectorTopologyBuilder.ELECTRICAL_CLASSES:
                    continue
                if page_types.get(ev.page) in (PageType.COVER.value, PageType.PART_LIST.value, PageType.TERMINAL_DIAGRAM.value):
                    continue
                nb = ev.norm_bbox
                if len(nb) != 4 or not all(0 <= v <= 1 for v in nb):
                    continue
                x0,y0,x1,y1 = nb
                if not (0.005 < x1-x0 < .85 and .005 < y1-y0 < .85):
                    continue
                class_id = VectorTopologyBuilder.ELECTRICAL_CLASSES.index(ev.symbol_class)
                line = f"{class_id} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}"
                by_page.setdefault(ev.page, set()).add(line)
        manifest = cls._manifest()
        previous = manifest.get(result.project_id, [])
        stems = []
        if by_page and Path(pdf_path).exists():
            with pymupdf.open(pdf_path) as doc:
                for pno, lines in sorted(by_page.items()):
                    if not 1 <= pno <= len(doc):
                        continue
                    stem = f"{token}_p{pno:03d}"
                    # Always refresh an image with its labels; a replacement PDF must not reuse stale pixels.
                    doc[pno-1].get_pixmap(dpi=dpi).save(images / f"{stem}.png")
                    (labels / f"{stem}.txt").write_text("\n".join(sorted(lines))+"\n", encoding="utf-8")
                    stems.append(stem)
        # Delete only this manifest's managed, obsolete label files. Legacy data remains untouched.
        for stem in set(previous)-set(stems):
            obsolete = (labels / f"{stem}.txt").resolve()
            if obsolete.parent == labels.resolve():
                obsolete.unlink(missing_ok=True)
        manifest[result.project_id] = stems
        (root / "verified_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        cls._write_data_yaml_and_train_script()
        status = cls.get_dataset_status()
        status.update(last_archived_project=result.project_id, newly_archived_pages=len(stems),
                      newly_archived_boxes=sum(len(by_page.get(int(stem.rsplit('_p',1)[1]), [])) for stem in stems))
        return status

    @classmethod
    def _write_data_yaml_and_train_script(cls) -> None:
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        projects = sorted(pid for pid, stems in cls._manifest().items() if stems)
        # Hold out complete projects, never random pages of the same drawing.
        val_projects = set(projects[-max(1, len(projects)//5):]) if len(projects) > 1 else set()
        for split in ("train", "val"):
            files = []
            for pid in projects:
                if (pid in val_projects) != (split == "val"):
                    continue
                files.extend((cls.DATASET_DIR / "images" / "train" / f"{stem}.png").resolve().as_posix() for stem in cls._manifest()[pid])
            (cls.DATASET_DIR / f"{split}.txt").write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
        yaml = [f"path: {json.dumps(cls.DATASET_DIR.resolve().as_posix(), ensure_ascii=False)}", "train: train.txt", "val: val.txt", "", "names:"]
        yaml.extend(f"  {idx}: {name}" for idx, name in enumerate(VectorTopologyBuilder.ELECTRICAL_CLASSES))
        (cls.DATASET_DIR / "data.yaml").write_text("\n".join(yaml)+"\n", encoding="utf-8")
        train_script = '''"""
Auto-generated Ultralytics YOLOv11 Training & ONNX Export Script
Run:
    pip install ultralytics
    python dataset/yolo_electrical/train_yolo11.py
"""
from pathlib import Path
import shutil

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装 ultralytics: pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return

    ds_dir = Path(__file__).resolve().parent
    data_yaml = ds_dir / "data.yaml"
    train_list, val_list = ds_dir / "train.txt", ds_dir / "val.txt"
    if not train_list.is_file() or not val_list.is_file() or not train_list.read_text(encoding="utf-8").strip() or not val_list.read_text(encoding="utf-8").strip():
        raise ValueError("训练前至少需要两个独立项目的已确认符号样本，分别用于训练和验证")
    model = YOLO("yolo11n.pt")
    model.train(data=str(data_yaml), epochs=50, imgsz=640, batch=8, project=str(ds_dir / "runs"), name="electrical_yolo11")
    onnx_path = model.export(format="onnx", imgsz=640)
    target_onnx = ds_dir.parent.parent / "bom_extractor" / "models" / "yolo_electrical.onnx"
    target_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, target_onnx)
    print(f"训练完成并已自动部署 ONNX 权重至: {target_onnx}")

if __name__ == "__main__":
    main()
'''
        (cls.DATASET_DIR / "train_yolo11.py").write_text(train_script, encoding="utf-8")


    @classmethod
    def get_dataset_status(cls) -> Dict[str, Any]:
        manifest = cls._manifest()
        counts = {name: 0 for name in VectorTopologyBuilder.ELECTRICAL_CLASSES}
        stems = {stem for files in manifest.values() for stem in files}
        for stem in stems:
            label = cls.DATASET_DIR / "labels" / "train" / f"{stem}.txt"
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 5 and parts[0].isdigit() and int(parts[0]) < len(counts):
                        counts[VectorTopologyBuilder.ELECTRICAL_CLASSES[int(parts[0])]] += 1
        return {"dataset_dir": str(cls.DATASET_DIR), "data_yaml_path": str(cls.DATASET_DIR / "data.yaml"),
            "train_script_path": str(cls.DATASET_DIR / "train_yolo11.py"),
            "yolo_onnx_deployed": VectorTopologyBuilder.YOLO_ONNX_PATH.exists(),
            "image_count": len(stems), "label_file_count": len(stems), "total_bounding_boxes": sum(counts.values()),
            "class_distribution": counts, "verified_project_count": sum(bool(files) for files in manifest.values()),
            "training_ready": sum(bool(files) for files in manifest.values()) >= 2,
            "legacy_labels_excluded": True}
