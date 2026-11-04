# 工业电控柜电气原理图 BOM 智能提取与图文核对系统 (`electrical-bom-extractor`)

一套针对工业电控柜 PDF 电气原理图（同时支持 **纯 CAD 原理图** 与 **含部件列表的 EPLAN 标准图纸**）的高精度 BOM 自动提取、跨页去重、工程规则清洗与“左图右表”在线图文核对系统。

---

## 一、 核心能力与架构

1. **底层双模解析引擎 (`bom_extractor/`)** — 与上层 Skill / Web 完全解耦：
   - **模式 A（含 EPLAN 部件列表图纸，如 `双钱L9_20261020.pdf`）**：
     - 基于 `PyMuPDF` 矢量坐标网格精准提取《部件列表》全部明细行；
     - 自动扫描前面原理图与安装板布置图，执行**双向交叉校验**（自动抓出原理图有但列表漏列的 `-FU2`、参数不符的 `-F8` 漏电断路器、同型号制造商笔误 `RXZE2M114 菲尼克斯 -> 施耐德`、布置图与列表 PLC 型号不一致等）；
     - 自动剔除数量为 `0` 的占位件及无订货号的现场液压件（`DT1~17`, `SP1~8`），并按企业规则为西门子 `S7-1500` 模块自动补齐前连接器、导轨与存储卡。
   - **模式 B（无自带列表的纯原理图，如 `C26368.pdf`）**：
     - 自动归一化图纸旋转角度（如 CAD 270° 旋转图框），基于视觉空间距离绑定位号与型号；
     - 对 CAD 导出变成纯几何曲线的文字区域（如第 5 页 `REXL2TMP7+RXZE2M114`、第 9 页 `LF60G-BPAA850SB`、第 20 页柜体尺寸图 `1550*900*300mm`）自动高清切片并调用视觉识别；
     - 跨页合并同一接触器/继电器的主触点、线圈与辅助触点，按星三角（Y-Δ）对称电机拓扑自动推导省略标注的支路型号。

2. **可复用 Agent Skill (`.agents/skills/electrical-bom-extractor/` & `skills/electrical-bom-extractor/`)**：
   - 提供标准 `SKILL.md`、命令行脚本 `scripts/extract_bom.py` 与电气规则参考文档 `references/electrical_rules.md`。

3. **工程师“左图右表”交互式 Web 审核工作台 (`backend/` + `frontend/`)**：
   - **左侧 PDF 图纸高亮预览**：点击右侧任意一行物料或待审异常，左侧自动翻到对应页码并用**动态红框圈出该元器件坐标位置**；支持跨页关联位置一键跳转。
   - **右侧在线编辑与实时汇总**：支持直接在网页上双击或点击【修改】编辑品牌、名称、型号、数量、单位、备注；支持【新增物料行】、【删除行】、【采纳待审补漏项】、【恢复已过滤物料】。修改后自动同步《明细表》与《合并采购汇总表》。
   - **一键导出标准 Excel**：随时点击右上角【导出标准 BOM (Excel)】，下载包含多 Sheet（标准采购清单、合并汇总表、待审核与校验报告、图纸页面结构分析）的标准 Excel 文件。

---

## 二、 快速启动方法

### 方式 1：启动 Web 图文审核工作台（推荐给工程师日常使用）
直接双击运行根目录下的 **`start_workbench.bat`**，或在终端执行：
```powershell
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
浏览器打开 **http://127.0.0.1:8000** 即可直接使用（已预载 `C26368.pdf` 与 `双钱L9_20261020.pdf` 两套完整工程，也可直接点击右上角【上传新图纸 PDF】）。

### 方式 2：使用 Skill / CLI 命令行提取与评测
```powershell
# 提取单份 PDF 图纸并生成 Excel + JSON + 局部切片
python skills/electrical-bom-extractor/scripts/extract_bom.py --pdf "参考文件/C26368.pdf"

# 对比人工标准 BOM 运行全量基准评测
python scripts/evaluate_benchmark.py
```

### 方式 3：配置多模态视觉大模型 API Key（可选）
系统默认内置矢量几何分析与离线识别兜底；如需对复杂扫描件启用外部视觉大模型，可设置环境变量（兼容 OpenAI / 阿里通义千问 Qwen-VL / DeepSeek 等接口）：
```powershell
$env:VLM_API_KEY="your-api-key"
$env:VLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VLM_MODEL="qwen-vl-max"
```
