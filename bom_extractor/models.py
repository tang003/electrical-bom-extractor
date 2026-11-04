from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


class PageType(str, Enum):
    COVER = "COVER"                         # 封面
    CABINET_LAYOUT = "CABINET_LAYOUT"       # 柜体外形图/开孔图
    INSTALL_LAYOUT = "INSTALL_LAYOUT"       # 安装板布置图
    MAIN_CIRCUIT = "MAIN_CIRCUIT"           # 动力线/主电路原理图
    CONTROL_CIRCUIT = "CONTROL_CIRCUIT"     # 控制回路原理图
    PLC_WIRING = "PLC_WIRING"               # PLC模块/接线图
    INSTRUMENT_WIRING = "INSTRUMENT_WIRING" # 仪表/传感器/阀门接线图
    TOPOLOGY = "TOPOLOGY"                   # 网络拓扑图
    TERMINAL_DIAGRAM = "TERMINAL_DIAGRAM"   # 端子排/场内接线图
    PART_LIST = "PART_LIST"                 # 部件列表/元器件清单
    OTHER = "OTHER"


class SourceType(str, Enum):
    EXPLICIT = "EXPLICIT"                   # 图纸明文标注
    TOPOLOGY_INFERRED = "TOPOLOGY_INFERRED" # 对称回路/星三角拓扑推导
    RULE_COMPLETED = "RULE_COMPLETED"       # 企业标准选型/附件规则补全
    ESTIMATED_AUX = "ESTIMATED_AUX"         # 辅材/端子/导线工程估算
    MANUAL_EDITED = "MANUAL_EDITED"         # 工程师网页端人工修改


@dataclass
class EvidenceLocation:
    page: int                               # 1-based PDF page number
    bbox: List[float]                       # [x0, y0, x1, y1] in PDF points
    norm_bbox: List[float]                  # [x0/w, y0/h, x1/w, y1/h] in 0..1
    raw_text: str = ""
    role: str = "main"                      # main | coil | contact | table_row | layout | dimension

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageInfo:
    page_num: int                           # 1-based PDF page number
    page_type: str                          # PageType value
    page_type_label: str                    # 中文描述
    title: str = ""
    sheet_no: str = ""
    width: float = 0.0
    height: float = 0.0
    text_block_count: int = 0
    drawing_count: int = 0
    has_vector_text_region: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BOMItem:
    id: str
    seq_no: int                             # 序号
    tag: str                                # 位号汇总字符串，如 "-QF1, -QF2" 或 "=02-K1"
    tags: List[str] = field(default_factory=list)
    placement: str = ""                     # 放置位置，如 "=02/13.2:D"
    name: str = ""                          # 名称/描述
    model: str = ""                         # 最终型号/订货编号
    raw_model: str = ""                     # 图纸原始提取型号
    brand: str = ""                         # 品牌/制造商
    unit: str = "个"                        # 单位
    quantity: float = 1                     # 数量
    remark: str = ""                        # 备注
    pdf_page_label: str = ""                # 页码标识（如 "第1页" 或 "P1, P5"）
    primary_page: int = 1                   # 点击联动跳转的主页码 (1-based)
    primary_bbox: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.2, 0.2])  # 归一化坐标 [x0, y0, x1, y1]
    evidences: List[EvidenceLocation] = field(default_factory=list)
    source_type: str = SourceType.EXPLICIT.value
    category: str = "primary"               # primary (主元器件) | auxiliary (导线/端子/柜体/人工)
    needs_review: bool = False
    review_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewItem:
    id: str
    severity: str                           # CONFLICT (冲突) | WARNING (疑似漏项/异常) | INFO (推导/补全确认)
    issue_type: str                         # MISSING_IN_PART_LIST | PARAM_MISMATCH | BRAND_TYPO | LAYOUT_MISMATCH | SYMMETRIC_INFERRED | RULE_COMPLETED | FILTERED_ITEM | VECTOR_VISION
    tag: str
    title: str
    description: str
    suggestion: str
    page: int
    norm_bbox: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.3, 0.3])
    related_page: Optional[int] = None
    related_norm_bbox: Optional[List[float]] = None
    status: str = "PENDING"                 # PENDING | ACCEPTED | IGNORED
    action_type: str = "CONFIRM"            # ADD_ITEM | UPDATE_ITEM | CONFIRM
    proposed_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    project_id: str
    pdf_filename: str
    project_name: str
    drawing_no: str
    extraction_mode: str                    # PART_LIST_MODE | SCHEMATIC_MODE
    page_count: int
    pages: List[PageInfo] = field(default_factory=list)
    itemized_bom: List[BOMItem] = field(default_factory=list)    # 按位号展开明细表
    aggregated_bom: List[BOMItem] = field(default_factory=list)  # 按型号合并采购汇总表
    filtered_items: List[BOMItem] = field(default_factory=list)  # 被规则过滤的0数量/非电控采购件（可恢复）
    review_items: List[ReviewItem] = field(default_factory=list) # 待人工审核与交叉校验清单
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
