"""
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
