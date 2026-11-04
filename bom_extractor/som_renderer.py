from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image, ImageDraw, ImageFont
import pymupdf

from .topology_graph import VectorTopologyBuilder
from .ocr_engine import DualChannelOCREngine
from .classifier import extract_visual_spans


class SoMVisualPromptBuilder:
    """
    Layer 2.5 & Layer 3: Microsoft Set-of-Mark (SoM) Visual Prompting Renderer
    & Grounded Actor-Critic Prompt Generator.
    Draws numbered visual markers [#1], [#2], ... on high-DPI schematic images
    and pairs them with Dual-Channel OCR text anchors + NetworkX wire branch chains.
    """

    @classmethod
    def render_som_annotated_page(
        cls,
        page: pymupdf.Page,
        out_path: Path,
        dpi: int = 200,
    ) -> Dict[str, Any]:
        """
        1. Extract dual-channel text spans (PyMuPDF + RapidOCR).
        2. Run VectorTopologyBuilder to detect candidate components and build NetworkX branch chains.
        3. Render high-DPI page image with Set-of-Mark [#1], [#2] bounding boxes and badges.
        """
        native_spans = extract_visual_spans(page)
        spans = DualChannelOCREngine.extract_dual_channel_spans(page, native_spans, dpi=dpi)
        topo = VectorTopologyBuilder.extract_wires_and_symbols(page, spans)

        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arialbd.ttf", 15)
        except Exception:
            font = ImageFont.load_default()

        w, h = float(pix.width), float(pix.height)
        for cand in topo["candidates"]:
            nx0, ny0, nx1, ny1 = cand["norm_bbox"]
            x0, y0 = int(nx0 * w), int(ny0 * h)
            x1, y1 = int(nx1 * w), int(ny1 * h)

            # Draw red candidate bounding box
            draw.rectangle([x0, y0, x1, y1], outline=(225, 29, 72), width=2)

            # Draw solid [#N] Set-of-Mark badge at top-left
            label = f"[{cand['som_label']}]"
            badge_w = max(32, len(label) * 8 + 6)
            badge_h = 18
            by0 = max(0, y0 - badge_h)
            draw.rectangle([x0, by0, x0 + badge_w, by0 + badge_h], fill=(225, 29, 72))
            draw.text((x0 + 3, by0 + 1), label, fill=(255, 255, 255), font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        topo["som_image_path"] = str(out_path)
        topo["dual_channel_span_count"] = len(spans)
        return topo

    @classmethod
    def build_actor_vlm_prompt(cls, topo: Dict[str, Any]) -> str:
        """
        Build the grounded Actor VLM prompt combining Set-of-Mark visual IDs [#1..#N],
        exact Dual-Channel OCR text tokens, and NetworkX wire connectivity branches.
        """
        lines = [
            "你是一个资深工业电气控制柜识图 Agent（Actor）。",
            "我已经使用几何矢量分离与 OCR 在这张高清图纸上为你框出了所有候选电气元件，并标上了醒目的红色编号 `[#1]`, `[#2]`...",
            "",
            "【底层双通道提取的候选框文本锚点】：",
        ]
        for cand in topo.get("candidates", [])[:60]:
            nearby_str = ", ".join(cand.get("nearby_texts", [])[:8])
            lines.append(
                f"- [{cand['som_label']}] 位号锚点: `{cand['tag']}` | 初始分类: `{cand['yolo_class']}` | 框内及周边精确文本: [{nearby_str}] | 坐标: {cand['norm_bbox']}"
            )

        if topo.get("branches"):
            lines.append("")
            lines.append("【底层 NetworkX 几何追线算出的纵向电气支路连通链（从上往下）】：")
            for b in topo["branches"][:20]:
                lines.append(f"- 支路 {b['branch_id']}: " + " --> ".join(b["chain"]))

        lines.extend([
            "",
            "请结合图片上的 `[#编号]` 红框、上述精确文本锚点以及纵向电气支路连通链，完成以下识别：",
            "1. 排除纯导线编号、普通端子排数字、液压电磁铁占位符（如 DT1~DT17）；",
            "2. 对同一位号的主触点、辅助触点与线圈不要重复计算采购数量；",
            "3. 如果某支路元件漏标型号，请根据上述纵向支路连通链及相邻对称支路推导型号，并在 remark 中注明 `AI拓扑推导`；",
            "4. 根据工业电气型号前缀（如 NSX/GV2/LC1D/LRD/RXM/REXL/XB2=施耐德, 6ES7/3RT=西门子, NDR/EDR=明纬, UK=菲尼克斯）自动补全品牌。",
            "",
            "请严格输出 JSON 数组格式（不要输出任何额外文字），每个元素格式如下：",
            '[{"som_label": "#1", "tag": "-QF1", "brand": "施耐德", "name": "塑壳断路器", "model": "NSX100F", "unit": "个", "quantity": 1, "role": "main", "confidence": 0.98, "remark": "SoM视觉锚定识别", "norm_bbox": [0.15, 0.20, 0.28, 0.32]}]',
        ])
        return "\n".join(lines)

    @classmethod
    def build_critic_audit_prompt(cls, page_num: int, actor_items: List[Dict[str, Any]], topo: Dict[str, Any]) -> str:
        """
        Build the Critic LLM self-reflection & audit prompt (`CoRe-VLM` Actor-Critic paradigm).
        """
        return (
            f"你是资深电气成套总工（Critic 审计 Agent）。请对第 {page_num} 页 Actor 提取出的电气元件清单进行严格复核：\n"
            f"【Actor 初步提取结果】：\n{actor_items}\n\n"
            f"【本页导线连通支路】：\n{topo.get('branches', [])}\n\n"
            "请检查并修正：\n"
            "1. 是否有把接触器/继电器的辅助触点当成独立元件重复计数的？若有请合并；\n"
            "2. 是否有型号字母拼写错误（对照双通道文本锚点校正 0/O、1/I）；\n"
            "3. 品牌是否与型号前缀 100% 匹配（如 RXZE/REXL/LC1D/GV2 必为施耐德，6ES7 必为西门子）；\n"
            "请输出修正后的最终 JSON 数组（保持原字段结构不变）。"
        )
