# 工业电控柜电气原理图 BOM 智能提取与图文核对系统 (`electrical-bom-extractor` v2.0)

一套面向工业电控柜 PDF 电气原理图（同时支持 **纯 CAD 原理图**、**含部件列表的 EPLAN 标准图纸** 与 **扫描/转曲 PDF**）的高精度 BOM 自动提取、跨页去重、工程规则清洗与“左图右表”在线图文核对系统。

---

## 一、 v2.0 五层终极架构 (`bom_extractor/`)

1. **第 1 层：双通道文字与转曲字还原底座 (`bom_extractor/classifier.py` + `bom_extractor/ocr_engine.py`)**
   - **`PyMuPDF` 矢量解析**：提取原生矢量文字、精确坐标与 `270°/90°` 旋转矩阵归一化（0 字符误差）。
   - **`RapidOCR` (PaddleOCR PP-OCRv3 ONNX 离线引擎)**：无需联网和外部 API Key，直接在本地 CPU/GPU 上识别 CAD 导出时炸成几何轮廓曲线的文字（如第 5 页 `REXL2TMP7+RXZE2M114`、第 9 页 `LF60G-BPAA850SB`、第 20 页纯矢量柜体尺寸 `1550*900*300mm`）。

2. **第 2 层：三合一候选元件定位 + 几何矢量追线 (`bom_extractor/topology_graph.py`)**
   - 借鉴开源项目 `D4NGYY/schematic-extractor`：自动分离 PDF 底层的长导线（Routing Wires）与短笔画元件符号簇（Symbol Clusters），结合位号文本锚点（`-QF1`, `-KM1` 等）与可插拔 `YOLOv11` ONNX 检测器，确保接近 100% 的元件召回率。
   - 使用 **`NetworkX`** 构建纵向电气支路连通图（自动串联同一竖向支路上的断路器 $\rightarrow$ 接触器 $\rightarrow$ 热继 $\rightarrow$ 电机负载）。

3. **第 3 层：微软 `Set-of-Mark (SoM)` 视觉打标 + Actor 视觉大模型 (`bom_extractor/som_renderer.py` + `bom_extractor/vision.py`)**
   - 借鉴微软开源 `microsoft/SoM`：在高清图纸上自动用红框圈出所有候选元件并打上醒目的 `[#1]`, `[#2]`, `[#3]` 编号标签，连同框内精确的双通道 OCR 文字锚点与 `NetworkX` 纵向支路链一起喂给 Actor 视觉大模型（支持 `Gemini 2.5 Pro` / `Claude 3.7 Sonnet` / `GPT-4o` / `Qwen-VL-Max`）。

4. **第 4 层：Critic 推理审计大模型 + 品牌基因库与成套规则引擎 (`bom_extractor/rules.py` + `bom_extractor/rules_config.json`)**
   - 借鉴 `CoRe-VLM` Actor-Critic 架构：对单页提取结果进行复核，跨页合并同一位号的线圈、主触点与辅助触点，顺着纵向支路推导漏标型号。
   - 内置**工业电气型号前缀基因库**（如 `NSX/GV2/LC1D/LRD/RXM/REXL/XB2` $\rightarrow$ 施耐德，`6ES7/3RT` $\rightarrow$ 西门子，`NDR/EDR` $\rightarrow$ 明纬，`UK` $\rightarrow$ 菲尼克斯），自动反推品牌并纠错（如自动抓出 `双钱L9` 中 `RXZE2M114` 品牌误填为菲尼克斯的笔误）。
   - 自动过滤数量为 `0` 的占位件及非电控采购的液压电磁铁（`DT1~17`, `SP1~8`），并按企业规则自动补齐西门子 `S7-1500` 前连接器/导轨/存储卡及施耐德 `GVAE11` 辅助触点。

5. **第 5 层：工程师“左图右表” Web 工作台 + YOLOv11 自进化飞轮 (`backend/app.py` + `frontend/` + `bom_extractor/yolo_evolver.py`)**
   - **左图右表高亮核对**：点击右侧任意物料或待审异常，左侧自动翻页并红框高亮定位；支持一键开启 **【🎯 SoM 候选框/追线层】** 直接查看图纸上的 `[#1]`, `[#2]` 视觉打标框与底部 `NetworkX` 纵向支路连通链。
   - **YOLOv11 自动蒸馏训练集沉淀 (`dataset/yolo_electrical/`)**：工程师每次在网页上确认或导出 BOM，系统自动将验证后的元件坐标按 10 类 IEC 电气符号标准导出为 `YOLOv11` 训练集（`data.yaml` + `images/train` + `labels/train` + 一键训练脚本 `train_yolo11.py`），全程无需人工画框标注。

---

## 二、 快速启动方法

### 方式 1：启动 Web 图文审核工作台（交付给电气工程师日常使用）
直接双击运行根目录下的 **`start_workbench.bat`**，或在终端执行：
```powershell
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
浏览器打开 **http://127.0.0.1:8000** 即可使用。

### 方式 2：使用 Skill / CLI 命令行提取与基准评测
```powershell
# 提取单份 PDF 图纸并生成 Excel + JSON + 局部切片 + YOLO 标注样本
python skills/electrical-bom-extractor/scripts/extract_bom.py --pdf "参考文件/C26368.pdf"

# 运行全量基准评测与单元测试
python scripts/evaluate_benchmark.py
python -m pytest tests/test_pipeline.py -v
```

### 方式 3：一键训练本厂专属 YOLOv11 电气符号模型（可选自进化）
```powershell
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
python dataset/yolo_electrical/train_yolo11.py
```
