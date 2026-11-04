from __future__ import annotations
import sys
import types
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pymupdf
from PIL import Image


def _ensure_onnxruntime_available() -> bool:
    """
    Ensure `onnxruntime` can be imported by `rapidocr_onnxruntime`.
    If Windows SmartAppControl/Defender blocks `onnxruntime_pybind11_state.pyd`,
    transparently provide an OpenCV DNN (`cv2.dnn.readNetFromONNX`) drop-in session
    so PP-OCRv3 ONNX models still execute locally on CPU without error.
    """
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        pass

    class _DummyNodeArg:
        def __init__(self, name: str):
            self.name = name

    class _DummyModelMeta:
        def __init__(self, meta_map: Dict[str, str]):
            self.custom_metadata_map = meta_map

    class _Cv2DnnInferenceSession:
        def __init__(self, model_path: str, sess_options: Any = None, providers: Any = None):
            self.model_path = str(model_path)
            self._nets_by_shape: Dict[Tuple[int, ...], Any] = {}
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self._meta: Dict[str, str] = {}
            raw = Path(self.model_path).read_bytes()
            marker = b"\n\tcharacter\x12"
            idx = raw.find(marker)
            if idx != -1:
                pos = idx + len(marker)
                while pos < len(raw) and (raw[pos] & 0x80):
                    pos += 1
                pos += 1
                try:
                    self._meta["character"] = raw[pos:].decode("utf-8", errors="ignore")
                except Exception:
                    pass

        def get_providers(self) -> List[str]:
            return ["CPUExecutionProvider"]

        def get_inputs(self) -> List[_DummyNodeArg]:
            return [_DummyNodeArg("x")]

        def get_outputs(self) -> List[_DummyNodeArg]:
            out_names = self.net.getUnconnectedOutLayersNames()
            return [_DummyNodeArg(n) for n in (out_names or ["output"])]

        def get_modelmeta(self) -> _DummyModelMeta:
            return _DummyModelMeta(self._meta)

        def _forward_single(self, sample: np.ndarray) -> np.ndarray:
            # Ensure minimum spatial dimensions and bucketed width to avoid OpenCV DNN dynamic shape reallocation
            _, c, h, w = sample.shape
            if h == 48:
                target_w = max(320, int(np.ceil(w / 64.0) * 64))
                if target_w > w:
                    padded = np.zeros((1, c, h, target_w), dtype=sample.dtype)
                    padded[:, :, :, :w] = sample
                    sample = padded
            else:
                target_h = max(64, int(np.ceil(h / 32.0) * 32))
                target_w = max(64, int(np.ceil(w / 32.0) * 32))
                if target_h != h or target_w != w:
                    padded = np.zeros((1, c, target_h, target_w), dtype=sample.dtype)
                    padded[:, :, :h, :w] = sample
                    sample = padded

            shape_key = tuple(sample.shape)
            net = self._nets_by_shape.get(shape_key)
            if net is None:
                net = cv2.dnn.readNetFromONNX(self.model_path)
                self._nets_by_shape[shape_key] = net
            net.setInput(sample)
            out = net.forward()
            if h != sample.shape[2] or (h != 48 and w != sample.shape[3]):
                out = out[:, :, :h, :w]
            return out

        def run(self, output_names: Any, input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
            inp = next(iter(input_feed.values()))
            if inp.ndim == 4 and inp.shape[0] > 1:
                outs = [self._forward_single(inp[i : i + 1]) for i in range(inp.shape[0])]
                return [np.concatenate(outs, axis=0)]
            return [self._forward_single(inp)]

    ort_mod = types.ModuleType("onnxruntime")
    ort_mod.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)  # type: ignore[attr-defined]
    ort_mod.SessionOptions = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        log_severity_level=4, enable_cpu_mem_arena=False, graph_optimization_level=99
    )
    ort_mod.InferenceSession = _Cv2DnnInferenceSession  # type: ignore[attr-defined]
    ort_mod.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    ort_mod.get_device = lambda: "CPU"  # type: ignore[attr-defined]
    sys.modules["onnxruntime"] = ort_mod
    return True


_ensure_onnxruntime_available()

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR = True
except Exception:
    RapidOCR = None
    HAS_RAPIDOCR = False


class DualChannelOCREngine:
    """
    Layer 1: Dual-Channel PDF Text & Coordinate Perception Base
    Combines PyMuPDF native vector text spans (0 character error) with
    offline RapidOCR (PaddleOCR ONNX engine) to recover CAD exploded/curve-outlined
    fonts (SHX/TTF outlines) and scanned/rasterized text with exact PDF coordinates.
    """

    _ocr_instance: Optional[Any] = None

    @classmethod
    def get_ocr(cls) -> Optional[Any]:
        if not HAS_RAPIDOCR:
            return None
        if cls._ocr_instance is None:
            try:
                cls._ocr_instance = RapidOCR()
            except Exception:
                cls._ocr_instance = None
        return cls._ocr_instance

    @classmethod
    def is_available(cls) -> bool:
        return cls.get_ocr() is not None

    @classmethod
    def crop_visual_region_from_page(
        cls,
        page: pymupdf.Page,
        norm_bbox: List[float],
        dpi: int = 300,
        save_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Render a PDF page in its upright visual orientation (handling rotation=90/180/270
        automatically) and crop the normalized bounding box [nx0, ny0, nx1, ny1].
        """
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        nx0 = max(0.0, min(1.0, float(norm_bbox[0])))
        ny0 = max(0.0, min(1.0, float(norm_bbox[1])))
        nx1 = max(nx0 + 0.01, min(1.0, float(norm_bbox[2])))
        ny1 = max(ny0 + 0.01, min(1.0, float(norm_bbox[3])))
        cropped = img.crop((
            int(nx0 * pix.width),
            int(ny0 * pix.height),
            int(nx1 * pix.width),
            int(ny1 * pix.height),
        ))
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(str(save_path))
        return cropped

    @classmethod
    def ocr_image_file(cls, image_path: str | Path) -> List[Dict[str, Any]]:
        """
        Run local RapidOCR (PP-OCRv3) on an image file and return list of recognized text boxes:
        [{"text": str, "confidence": float, "box": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]}]
        """
        engine = cls.get_ocr()
        if engine is None or not Path(image_path).exists():
            return []
        try:
            result, _ = engine(str(image_path), use_det=True, use_cls=False, use_rec=True)
            if not result:
                # Try direct single-line recognition if text fills the entire cropped patch
                rec_res, _ = engine(str(image_path), use_det=False, use_cls=False, use_rec=True)
                if rec_res:
                    items = []
                    for row in rec_res:
                        text, score = str(row[0]).strip(), float(row[1])
                        if text and score >= 0.35:
                            items.append({"text": text, "confidence": round(score, 4), "box": [[0, 0], [100, 0], [100, 30], [0, 30]]})
                    return items
                return []
            items = []
            for row in result:
                box, text, score = row[0], str(row[1]).strip(), float(row[2])
                if text:
                    items.append({
                        "text": text,
                        "confidence": round(score, 4),
                        "box": box,
                    })
            return items
        except Exception:
            return []

    @classmethod
    def ocr_pdf_region(
        cls,
        page: pymupdf.Page,
        norm_bbox: List[float],
        dpi: int = 300,
        save_crop_path: Optional[Path] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Render a specific normalized bounding box [x0, y0, x1, y1] from the PDF page
        at 300 DPI in visual upright coordinates and run local RapidOCR.
        """
        cropped = cls.crop_visual_region_from_page(page, norm_bbox, dpi=dpi, save_path=save_crop_path)
        engine = cls.get_ocr()
        if engine is None:
            return "", []

        arr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        try:
            result, _ = engine(arr, use_det=True, use_cls=False, use_rec=True)
            items: List[Dict[str, Any]] = []
            texts: List[str] = []
            if result:
                for row in result:
                    box, text, score = row[0], str(row[1]).strip(), float(row[2])
                    if text and score >= 0.40:
                        texts.append(text)
                        items.append({"text": text, "confidence": round(score, 4), "box": box})
            if not texts:
                rec_res, _ = engine(arr, use_det=False, use_cls=False, use_rec=True)
                if rec_res:
                    for row in rec_res:
                        text, score = str(row[0]).strip(), float(row[1])
                        if text and score >= 0.35:
                            texts.append(text)
                            items.append({"text": text, "confidence": round(score, 4), "box": []})
            return " ".join(texts), items
        except Exception:
            return "", []

    @classmethod
    def extract_dual_channel_spans(
        cls,
        page: pymupdf.Page,
        native_spans: List[Dict[str, Any]],
        dpi: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Merge PyMuPDF native vector text spans with RapidOCR spans on pages that are
        scanned/rasterized or have sparse native text.
        """
        merged = list(native_spans)
        engine = cls.get_ocr()
        if engine is None:
            return merged

        rot_rect = page.rect * page.rotation_matrix
        w = abs(float(rot_rect.width)) or 1.0
        h = abs(float(rot_rect.height)) or 1.0

        if len(native_spans) < 12:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            scale_x = w / float(pix.width or 1)
            scale_y = h / float(pix.height or 1)
            try:
                result, _ = engine(arr, use_det=True, use_cls=False, use_rec=True)
                if result:
                    for row in result:
                        box, text, score = row[0], str(row[1]).strip(), float(row[2])
                        if not text or score < 0.45:
                            continue
                        xs = [pt[0] * scale_x for pt in box]
                        ys = [pt[1] * scale_y for pt in box]
                        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                        if not cls._overlaps_existing([x0, y0, x1, y1], merged):
                            merged.append({
                                "text": text,
                                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                                "norm_bbox": [
                                    round(x0 / w, 4),
                                    round(y0 / h, 4),
                                    round(x1 / w, 4),
                                    round(y1 / h, 4),
                                ],
                                "cx": round((x0 + x1) / 2.0, 2),
                                "cy": round((y0 + y1) / 2.0, 2),
                                "size": round(max(6.0, y1 - y0), 1),
                                "channel": "RAPID_OCR",
                                "confidence": round(score, 4),
                            })
            except Exception:
                pass

        return merged

    @staticmethod
    def _overlaps_existing(bbox: List[float], existing_spans: List[Dict[str, Any]], iou_thresh: float = 0.35) -> bool:
        x0, y0, x1, y1 = bbox
        area_a = max(1.0, (x1 - x0) * (y1 - y0))
        for sp in existing_spans:
            bx0, by0, bx1, by1 = sp["bbox"]
            ix0, iy0 = max(x0, bx0), max(y0, by0)
            ix1, iy1 = min(x1, bx1), min(y1, by1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / area_a >= iou_thresh:
                    return True
        return False
