from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pymupdf

from .models import ExtractionResult, PageType
from .topology_graph import VectorTopologyBuilder


class YOLOSelfEvolver:
    """
    Layer 5: Human-in-the-Loop YOLOv11 Auto-Distillation & Self-Evolution Engine
    Automatically converts engineer-verified BOM items & bounding boxes ( plus
    vector-clustered symbol boxes ) into a standard Ultralytics YOLOv8/YOLOv11
    training dataset (`dataset/yolo_electrical/data.yaml`, `images/train`, `labels/train`).
    """

    DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "yolo_electrical"

    @classmethod
    def archive_project_to_yolo_dataset(
        cls,
        result: ExtractionResult,
        pdf_path: Path,
        dpi: int = 160,
    ) -> Dict[str, Any]:
        """
        Export validated component bounding boxes from `result` and `pdf_path`
        into YOLOv11 format (`<class_id> <xc> <yc> <w> <h>`).
        """
        images_dir = cls.DATASET_DIR / "images" / "train"
        labels_dir = cls.DATASET_DIR / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Group bounding boxes by 1-based page number
        boxes_by_page: Dict[int, List[str]] = {}
        items = result.itemized_bom if result.extraction_mode == "PART_LIST_MODE" else result.aggregated_bom

        # Map page types to avoid exporting pure cover/part-list text tables as symbol training pages
        page_type_map = {p.page_num: p.page_type for p in result.pages}

        total_boxes = 0
        for item in items:
            if item.category == "auxiliary":
                continue
            cls_id, _ = VectorTopologyBuilder.classify_tag_to_yolo_class(item.tag, item.name)

            # Collect from primary_bbox + all evidence locations
            ev_list = [(item.primary_page, item.primary_bbox)]
            for ev in item.evidences:
                if ev.norm_bbox and len(ev.norm_bbox) == 4:
                    ev_list.append((ev.page, ev.norm_bbox))

            seen_on_page = set()
            for pno, nb in ev_list:
                if not nb or len(nb) != 4:
                    continue
                ptype = page_type_map.get(pno, "")
                if ptype in (PageType.COVER.value, PageType.PART_LIST.value):
                    continue
                nx0, ny0, nx1, ny1 = [max(0.0, min(1.0, float(v))) for v in nb]
                bw = nx1 - nx0
                bh = ny1 - ny0
                if bw <= 0.005 or bh <= 0.005 or bw > 0.85 or bh > 0.85:
                    continue
                xc = (nx0 + nx1) / 2.0
                yc = (ny0 + ny1) / 2.0
                key = (pno, round(xc, 2), round(yc, 2))
                if key in seen_on_page:
                    continue
                seen_on_page.add(key)
                line = f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                boxes_by_page.setdefault(pno, []).append(line)
                total_boxes += 1

        saved_pages = 0
        if pdf_path.exists() and boxes_by_page:
            doc = pymupdf.open(str(pdf_path))
            try:
                for pno, lines in boxes_by_page.items():
                    if 1 <= pno <= len(doc):
                        stem = f"{result.project_id}_p{pno:03d}"
                        img_file = images_dir / f"{stem}.png"
                        lbl_file = labels_dir / f"{stem}.txt"
                        if not img_file.exists():
                            pix = doc[pno - 1].get_pixmap(dpi=dpi)
                            pix.save(str(img_file))
                        # Deduplicate lines
                        unique_lines = list(dict.fromkeys(lines))
                        lbl_file.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
                        saved_pages += 1
            finally:
                doc.close()

        cls._write_data_yaml_and_train_script()
        status = cls.get_dataset_status()
        status["last_archived_project"] = result.project_id
        status["last_archived_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status["newly_archived_pages"] = saved_pages
        status["newly_archived_boxes"] = total_boxes

        meta_path = cls.DATASET_DIR / "dataset_meta.json"
        meta_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    @classmethod
    def _write_data_yaml_and_train_script(cls) -> None:
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        root_posix = cls.DATASET_DIR.as_posix()
        yaml_lines = [
            f"path: {root_posix}",
            "train: images/train",
            "val: images/train",
            "",
            "names:",
        ]
        for idx, name in enumerate(VectorTopologyBuilder.ELECTRICAL_CLASSES):
            yaml_lines.append(f"  {idx}: {name}")

        (cls.DATASET_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

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
        images_dir = cls.DATASET_DIR / "images" / "train"
        labels_dir = cls.DATASET_DIR / "labels" / "train"
        img_files = list(images_dir.glob("*.png")) if images_dir.exists() else []
        lbl_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        total_annotations = 0
        class_counts: Dict[str, int] = {name: 0 for name in VectorTopologyBuilder.ELECTRICAL_CLASSES}

        for lf in lbl_files:
            try:
                for line in lf.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        total_annotations += 1
                        cid = int(parts[0])
                        if 0 <= cid < len(VectorTopologyBuilder.ELECTRICAL_CLASSES):
                            cname = VectorTopologyBuilder.ELECTRICAL_CLASSES[cid]
                            class_counts[cname] = class_counts.get(cname, 0) + 1
            except Exception:
                pass

        return {
            "dataset_dir": str(cls.DATASET_DIR),
            "data_yaml_path": str(cls.DATASET_DIR / "data.yaml"),
            "train_script_path": str(cls.DATASET_DIR / "train_yolo11.py"),
            "yolo_onnx_deployed": VectorTopologyBuilder.YOLO_ONNX_PATH.exists(),
            "image_count": len(img_files),
            "label_file_count": len(lbl_files),
            "total_bounding_boxes": total_annotations,
            "class_distribution": class_counts,
        }
