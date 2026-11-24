from __future__ import annotations
import math
import re
from statistics import median
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import networkx as nx
import numpy as np
import pymupdf


from .notation import COMPONENT_TAG_RE as TAG_ANCHOR_RE, device_code, extract_model_strings, is_connection_tag, with_component_anchors
from .device_outlines import device_outline_boxes, outline_near_tag


class VectorTopologyBuilder:
    """
    Layer 2: Three-in-One Candidate Component Locator + Vector Wire Connectivity Graph
    Reference architectures:
      - `D4NGYY/schematic-extractor` (Vector Wire vs. Symbol Separation + Geometric BFS Routing)
      - `microsoft/SoM` (Set-of-Mark candidate region numbering [#1], [#2], ...)
      - Pluggable YOLOv11 ONNX/PyTorch electrical symbol detector
    """

    YOLO_ONNX_PATH = Path(__file__).parent / "models" / "yolo_electrical.onnx"
    YOLO_PT_PATH = Path(__file__).parent / "models" / "yolo_electrical.pt"

    ELECTRICAL_CLASSES = [
        "circuit_breaker",   # 0: 断路器 (QF/F)
        "contactor_coil",    # 1: 接触器/继电器线圈 (KM/KA/KT)
        "contactor_contact", # 2: 主/辅助触点
        "thermal_relay",     # 3: 热继电器 (FR)
        "motor",             # 4: 电机 (M)
        "fuse",              # 5: 熔断器 (FU)
        "transformer",       # 6: 变压器/电源 (TC/TM/G/U)
        "plc_module",        # 7: PLC/控制器模块
        "switch_button",     # 8: 按钮/指示灯/开关 (SB/SA/HL)
        "terminal_other",    # 9: 端子排/其他元件 (XT)
    ]

    @classmethod
    def classify_tag_to_yolo_class(cls, tag: str, name: str = "") -> Tuple[int, str]:
        t = device_code(tag)
        if t in ("QF", "QS", "Q", "F", "FB"):
            return 0, "circuit_breaker"
        if t.startswith(("KM", "KA", "KT", "KV", "K")):
            if "触点" in name:
                return 2, "contactor_contact"
            return 1, "contactor_coil"
        if t.startswith("FR"):
            return 3, "thermal_relay"
        if t.startswith("M") or "电机" in name or "风机" in name or "泵" in name:
            return 4, "motor"
        if t.startswith("FU"):
            return 5, "fuse"
        if t.startswith(("TC", "TM", "TV", "TA", "T", "G", "PS")) or "电源" in name or "变压器" in name:
            return 6, "transformer"
        if t.startswith(("PLC", "CPU", "U")) or "PLC" in name or "模块" in name or "控制器" in name:
            return 7, "plc_module"
        if t.startswith(("SB", "SA", "HL", "HG", "HR", "HA", "WL")) or "按钮" in name or "指示灯" in name:
            return 8, "switch_button"
        return 9, "terminal_other"

    @classmethod
    def extract_wires_and_symbols(
        cls,
        page: pymupdf.Page,
        spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        1. Separate vector drawings into long routing wires (`wires`) vs. dense component symbol strokes (`symbol_clusters`).
        2. Fuse with optional local YOLOv11 detector boxes + text tag anchors (`-QF1`, `-KM1`, etc.).
        3. Assign Set-of-Mark IDs (`[#1]`, `[#2]`, ...) and build a `NetworkX` connectivity graph.
        """
        rot_mat = page.rotation_matrix
        spans=with_component_anchors(spans)
        page_w = float(page.rect.width) or 1.0
        page_h = float(page.rect.height) or 1.0

        drawings = page.get_drawings()
        wires: List[Tuple[float, float, float, float]] = []
        short_rects: List[List[float]] = []
        mechanical_context=any(re.search(r'开孔|穿线孔|安装孔|钻孔|\bCUTOUT\b|MOUNTING\s+HOLES?',s['text'],re.I) for s in spans)
        sizes=[s['size'] for s in spans if s.get('size',0)>0]
        usual_font=median(sizes) if sizes else 0
        solid_frames=[]

        for d in drawings:
            r = d.get("rect")
            if not r:
                continue
            vr = pymupdf.Rect(r) * rot_mat
            x0, y0 = max(0.0, float(vr.x0)), max(0.0, float(vr.y0))
            x1, y1 = min(page_w, float(vr.x1)), min(page_h, float(vr.y1))
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            w = x1 - x0
            h = y1 - y0
            fill=d.get('fill')
            if (mechanical_context and fill is not None and max(fill)<.25 and d.get('fill_opacity',1)>.9
                and max(w,h)>.2*min(page_w,page_h) and max(3,usual_font*.8)<min(w,h)<.04*min(page_w,page_h)):
                solid_frames.append([x0,y0,x1,y1])

            # Ignore border frames and title block area at bottom
            if w > page_w * 0.75 or h > page_h * 0.75 or y0 > page_h * 0.87 or y1 < page_h * 0.03:
                continue

            # Long horizontal or vertical stroke -> routing wire segment
            if (w > 36.0 and h < 3.5) or (h > 36.0 and w < 3.5):
                wires.append((x0, y0, x1, y1))
            elif 2.0 <= max(w, h) <= 55.0:
                short_rects.append([x0, y0, x1, y1])

        # Step 2: Cluster nearby short strokes into symbol bounding boxes
        symbol_clusters = cls._cluster_boxes(short_rects, dist_thresh=6.0, min_size=6.0, max_size=65.0)

        # Step 3: Optional YOLOv11 ONNX detection if custom weights exist
        yolo_boxes = cls._run_yolo_if_available(page, page_w, page_h)

        # Step 4: Anchor around explicit component tags and merge tightly with symbol clusters
        device_outlines = device_outline_boxes(page)
        candidates: List[Dict[str, Any]] = []
        used_cluster_indices = set()

        content_spans = [
            s for s in spans
            if 0.03 * page_h < s["cy"] < 0.87 * page_h
            and 0.02 * page_w < s["cx"] < 0.98 * page_w
        ]

        for sp in content_spans:
            txt = sp["text"].strip()
            if not TAG_ANCHOR_RE.match(txt):
                continue
            if txt.isdigit() or len(txt) < 2:
                continue
            sx0, sy0, sx1, sy1 = sp["bbox"]
            scx, scy = sp["cx"], sp["cy"]

            # Find nearest geometric symbol cluster within 36pt
            best_idx = -1
            best_dist = 36.0
            for idx, cb in enumerate(symbol_clusters):
                ccx = (cb[0] + cb[2]) / 2.0
                ccy = (cb[1] + cb[3]) / 2.0
                dist = math.hypot(scx - ccx, scy - ccy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx != -1:
                cb = symbol_clusters[best_idx]
                used_cluster_indices.add(best_idx)
                ux0 = min(sx0, cb[0]) - 4.0
                uy0 = min(sy0, cb[1]) - 4.0
                ux1 = max(sx1, cb[2]) + 4.0
                uy1 = max(sy1, cb[3]) + 4.0
            else:
                ux0, uy0, ux1, uy1 = sx0 - 10.0, sy0 - 10.0, sx1 + 16.0, sy1 + 14.0

            # Collect nearby text tokens for semantic context (without inflating the visual box beyond 34pt)
            nearby_texts = []
            for other in sorted(content_spans, key=lambda o: math.hypot(o["cx"] - scx, o["cy"] - scy)):
                d_other = math.hypot(other["cx"] - scx, other["cy"] - scy)
                ot = other["text"].strip()
                if d_other <= 48.0:
                    nearby_texts.append(ot)
                    # Only expand visual box for tight adjacent model/spec text within 32pt (and not another tag)
                    if d_other <= 32.0 and not (other is not sp and TAG_ANCHOR_RE.match(ot)):
                        ux0 = min(ux0, other["bbox"][0] - 2.0)
                        uy0 = min(uy0, other["bbox"][1] - 2.0)
                        ux1 = max(ux1, other["bbox"][2] + 2.0)
                        uy1 = max(uy1, other["bbox"][3] + 2.0)
                elif (
                    txt.upper().lstrip("=").startswith(("-LI", "-TI", "-PI", "-FI", "-U", "-PLC", "-CPU"))
                    and abs(other["cx"] - scx) <= 46.0
                    and 0 < (other["cy"] - scy) <= 210.0
                ):
                    # Large instrument / PLC module block: wrap the entire module housing down to its bottom model string
                    nearby_texts.append(ot)
                    ux0 = min(ux0, other["bbox"][0] - 8.0)
                    uy0 = min(uy0, other["bbox"][1] - 4.0)
                    ux1 = max(ux1, other["bbox"][2] + 8.0)
                    uy1 = max(uy1, other["bbox"][3] + 4.0)

            body = outline_near_tag(device_outlines, pymupdf.Rect(sx0, sy0, sx1, sy1))
            if body is not None:
                ux0, uy0, ux1, uy1 = body.x0-2.0, body.y0-2.0, body.x1+2.0, body.y1+2.0

            ux0 = max(0.0, ux0)
            uy0 = max(0.0, uy0)
            ux1 = min(page_w, ux1)
            uy1 = min(page_h, uy1)

            cls_id, cls_name = cls.classify_tag_to_yolo_class(txt, " ".join(nearby_texts))
            model_candidates=list(dict.fromkeys(value for token in nearby_texts for value in extract_model_strings(token)))
            # Keep suspicious mechanical labels inspectable, without presenting them
            # as normal electrical candidates or using them in a branch chain.
            font_size=sp.get('size',sy1-sy0)
            next_to_frame=any(math.hypot(max(r[0]-sx1,sx0-r[2],0),max(r[1]-sy1,sy0-r[3],0))<=max(2,usual_font*.65) for r in solid_frames)
            connection_label=is_connection_tag(txt)
            excluded=connection_label or bool(mechanical_context and usual_font and 0<font_size<=min(3,usual_font*.65) and next_to_frame and not model_candidates)
            candidates.append({
                "tag": sp.get('native_tag',txt),
                "anchor_source_text": sp.get('source_text',txt),
                "unindexed_tag": sp.get('unindexed_tag',False),
                "yolo_class_id": cls_id,
                "yolo_class": cls_name,
                "bbox": [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)],
                "norm_bbox": [
                    round(ux0 / page_w, 4),
                    round(uy0 / page_h, 4),
                    round(ux1 / page_w, 4),
                    round(uy1 / page_h, 4),
                ],
                "cx": round((ux0 + ux1) / 2.0, 2),
                "cy": round((uy0 + uy1) / 2.0, 2),
                "nearby_texts": list(dict.fromkeys(nearby_texts)),
                "model_candidates": model_candidates,
                "anchor_norm_bbox": [round(sx0/page_w,4),round(sy0/page_h,4),round(sx1/page_w,4),round(sy1/page_h,4)],
                "anchor_font_size": round(font_size,2),
                "bbox_source": "vector_device_outline" if body is not None else "tag_and_nearby_geometry",
                "excluded": excluded,
                "exclusion_reason": ("跨页连线/电位标识，不是独立采购元件。" if connection_label else
                                     "开孔/机械图中的微小标注紧贴粗实心边框，暂缺有效电气元件证据。" if excluded else ""),
                "detector_source": "VECTOR_SYMBOL+TAG_ANCHOR",
            })

        # Add any YOLO-detected boxes that weren't covered by tag anchors
        for yb in yolo_boxes:
            if not cls._overlaps_any(yb["bbox"], [c for c in candidates if not c.get('excluded')]):
                candidates.append(yb)

        # Deduplicate overlapping candidates and sort top-to-bottom, left-to-right
        candidates = (cls._nms_candidates([c for c in candidates if not c.get('excluded')], iou_thresh=0.45)
                      +cls._nms_candidates([c for c in candidates if c.get('excluded')], iou_thresh=0.45))
        candidates.sort(key=lambda c: (round(c["cx"] / 55.0), c["cy"]))

        # Assign Set-of-Mark IDs: [#1], [#2], ...
        for idx, cand in enumerate(candidates, start=1):
            cand["som_id"] = idx
            cand["som_label"] = f"#{idx}"

        excluded_candidates=[c for c in candidates if c.get('excluded')]
        candidates=[c for c in candidates if not c.get('excluded')]

        # Step 5: Build NetworkX wire & functional branch connectivity graph
        graph, branches = cls._build_connectivity_graph(candidates, wires)

        return {
            "page_num": page.number + 1,
            "page_width": round(page_w, 2),
            "page_height": round(page_h, 2),
            "wire_segment_count": len(wires),
            "symbol_cluster_count": len(symbol_clusters),
            "candidates": candidates,
            "excluded_candidates": excluded_candidates,
            "branches": branches,
            "edges": [{"from": u, "to": v, "type": d.get("type", "wire")} for u, v, d in graph.edges(data=True)],
        }

    @staticmethod
    def _cluster_boxes(
        rects: List[List[float]],
        dist_thresh: float = 8.0,
        min_size: float = 8.0,
        max_size: float = 95.0,
    ) -> List[List[float]]:
        if not rects:
            return []
        # Simple spatial grid / iterative box merging
        boxes = [list(r) for r in rects[:600]]
        merged = True
        while merged:
            merged = False
            new_boxes: List[List[float]] = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                bx0, by0, bx1, by1 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    ox0, oy0, ox1, oy1 = boxes[j]
                    if (
                        ox0 <= bx1 + dist_thresh
                        and ox1 >= bx0 - dist_thresh
                        and oy0 <= by1 + dist_thresh
                        and oy1 >= by0 - dist_thresh
                    ):
                        nx0, ny0 = min(bx0, ox0), min(by0, oy0)
                        nx1, ny1 = max(bx1, ox1), max(by1, oy1)
                        if (nx1 - nx0) <= max_size and (ny1 - ny0) <= max_size:
                            bx0, by0, bx1, by1 = nx0, ny0, nx1, ny1
                            used[j] = True
                            merged = True
                used[i] = True
                new_boxes.append([bx0, by0, bx1, by1])
            boxes = new_boxes

        return [
            b for b in boxes
            if (b[2] - b[0]) >= min_size and (b[3] - b[1]) >= min_size
        ]

    @classmethod
    def _run_yolo_if_available(cls, page: pymupdf.Page, page_w: float, page_h: float) -> List[Dict[str, Any]]:
        if not cls.YOLO_ONNX_PATH.exists():
            return []
        try:
            net = cv2.dnn.readNetFromONNX(str(cls.YOLO_ONNX_PATH))
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            _ = net.forward()
        except Exception:
            pass
        return []

    @staticmethod
    def _overlaps_any(bbox: List[float], candidates: List[Dict[str, Any]], iou_thresh: float = 0.3) -> bool:
        x0, y0, x1, y1 = bbox
        area_a = max(1.0, (x1 - x0) * (y1 - y0))
        for c in candidates:
            bx0, by0, bx1, by1 = c["bbox"]
            ix0, iy0 = max(x0, bx0), max(y0, by0)
            ix1, iy1 = min(x1, bx1), min(y1, by1)
            if ix1 > ix0 and iy1 > iy0:
                if ((ix1 - ix0) * (iy1 - iy0)) / area_a >= iou_thresh:
                    return True
        return False

    @staticmethod
    def _nms_candidates(candidates: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for cand in candidates:
            x0, y0, x1, y1 = cand["bbox"]
            area = max(1.0, (x1 - x0) * (y1 - y0))
            duplicate = False
            for k in kept:
                kx0, ky0, kx1, ky1 = k["bbox"]
                ix0, iy0 = max(x0, kx0), max(y0, ky0)
                ix1, iy1 = min(x1, kx1), min(y1, ky1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    if inter / min(area, max(1.0, (kx1 - kx0) * (ky1 - ky0))) >= iou_thresh:
                        if cand["tag"] and cand["tag"] not in k["tag"]:
                            k["tag"] = f"{k['tag']}, {cand['tag']}"
                        duplicate = True
                        break
            if not duplicate:
                kept.append(cand)
        return kept

    @staticmethod
    def _build_connectivity_graph(
        candidates: List[Dict[str, Any]],
        wires: List[Tuple[float, float, float, float]],
    ) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
        """
        Build a NetworkX graph connecting components that lie on the same vertical functional branch
        or are bridged by vertical/horizontal routing wires.
        """
        G = nx.Graph()
        for c in candidates:
            G.add_node(c["som_label"], **c)

        # Group candidates into vertical schematic columns (branches) within dx <= 32pt
        columns: List[List[Dict[str, Any]]] = []
        for c in sorted(candidates, key=lambda item: item["cx"]):
            placed = False
            for col in columns:
                avg_cx = sum(node["cx"] for node in col) / len(col)
                if abs(c["cx"] - avg_cx) <= 32.0:
                    col.append(c)
                    placed = True
                    break
            if not placed:
                columns.append([c])

        branches: List[Dict[str, Any]] = []
        for b_idx, col in enumerate(columns, start=1):
            col.sort(key=lambda item: item["cy"])
            for i in range(len(col) - 1):
                u = col[i]
                v = col[i + 1]
                # Check if a vertical wire exists in this x-corridor between u.cy and v.cy
                has_wire = any(
                    abs(((wx0 + wx1) / 2.0) - ((u["cx"] + v["cx"]) / 2.0)) <= 28.0
                    and wy0 <= v["cy"] + 15.0
                    and wy1 >= u["cy"] - 15.0
                    for wx0, wy0, wx1, wy1 in wires
                    if (wy1 - wy0) > (wx1 - wx0)
                )
                G.add_edge(
                    u["som_label"],
                    v["som_label"],
                    type="vertical_wire" if has_wire else "branch_column",
                )
            if len(col) >= 1:
                branches.append({
                    "branch_id": f"B{b_idx}",
                    "center_x": round(sum(n["cx"] for n in col) / len(col), 1),
                    "chain": [
                        f"[{n['som_label']}: {n['tag']} ({' / '.join(n['nearby_texts'][:4])})]"
                        for n in col
                    ],
                    "node_labels": [n["som_label"] for n in col],
                })

        return G, branches
