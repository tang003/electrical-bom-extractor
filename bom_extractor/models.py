from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import List, Optional, Dict, Any


class PageType(str, Enum):
    COVER = "COVER"                         # 封面
    CONTENTS = "CONTENTS"                   # 目录/图纸索引
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
    UNRESOLVED = "UNRESOLVED"               # 位号已识别，型号未可靠提取
    VISION_RECOGNIZED = "VISION_RECOGNIZED" # OCR/AI 视觉结果，需人工核实
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
    verified: bool = False                  # Human-confirmed symbol box, eligible for training
    symbol_class: Optional[str] = None      # Explicit symbol type; never inferred from a BOM tag
    device_tag: str = ""
    cad_refs: List[Dict[str, Any]] = field(default_factory=list)
    model_binding: str = ""                 # inline, enclosure (unique owner), enclosure_candidate or spatial

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
    location_code: str = ""
    function_code: str = ""
    extraction_diagnostics: Dict[str, Any] = field(default_factory=dict)

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
    quantity: Optional[float] = 1           # None means the purchase count/length is not established
    remark: str = ""                        # 备注
    pdf_page_label: str = ""                # 页码标识（如 "第1页" 或 "P1, P5"）
    primary_page: int = 1                   # 点击联动跳转的主页码 (1-based)
    primary_bbox: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.2, 0.2])  # 归一化坐标 [x0, y0, x1, y1]
    evidences: List[EvidenceLocation] = field(default_factory=list)
    source_type: str = SourceType.EXPLICIT.value
    category: str = "primary"               # primary (主元器件) | auxiliary (导线/端子/柜体/人工)
    needs_review: bool = False
    review_note: str = ""
    member_ids: List[str] = field(default_factory=list)  # Stable link from a summary row to its details
    is_quantity_adjustment: bool = False    # Explicit procurement correction, not a physical device
    local_tag: str = ""
    location_code: str = ""
    specification: str = ""
    group_identity: str = ""                # Keeps unknown-model procurement corrections with their device
    entity_role: str = "device"             # device | coil | contact | terminal | cable | assembly
    parent_tag: str = ""                    # Evidence-backed assembly/component relationship
    procurement_scope: str = "unknown"      # cabinet | field | unknown | excluded
    procurement_scope_basis: str = ""        # project_cabinet_default | drawing_field_load | manual
    quantity_status: str = "by_tag"         # by_tag | confirmed | unknown | needs_confirmation
    quantity_basis: str = "unique_tag"
    quantity_candidates: List[float] = field(default_factory=list)
    selection_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    model_origin: str = ""
    parts_reference_id: str = ""
    procurement_name: str = ""              # Device class before AI adds a drawing-specific functional description
    component_parts: List[Dict[str, Any]] = field(default_factory=list)  # Evidence-backed body/socket/accessory candidates
    terminal_strip_tag: str = ""             # Parent strip identity when one strip has several terminal models
    terminal_numbers: List[int] = field(default_factory=list)  # Drawn numbers for this material row

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
    norm_bbox: Optional[List[float]] = None
    related_page: Optional[int] = None
    related_norm_bbox: Optional[List[float]] = None
    status: str = "PENDING"                 # PENDING | ACCEPTED | IGNORED
    action_type: str = "CONFIRM"            # ADD_ITEM | UPDATE_ITEM | CONFIRM
    proposed_data: Optional[Dict[str, Any]] = None
    scope: str = "page"                     # page | project; project warnings have no component box

    def __post_init__(self):
        # Older snapshots attached a generic rectangle to import-wide warnings.
        # Restore their true scope without changing an engineer's decision status.
        if self.issue_type=='CAD_IMPORT_REVIEW':
            self.scope='project';self.norm_bbox=None

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
    source_format: str = "pdf"
    source_filename: str = ""
    independent_bom: List[BOMItem] = field(default_factory=list)
    declared_parts: List[BOMItem] = field(default_factory=list)
    reconciliation: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        from .reconciliation import tag_members
        data=asdict(self)
        for key in ('itemized_bom','aggregated_bom','filtered_items','independent_bom','declared_parts'):
            for item in data[key]:
                identities = [item['terminal_strip_tag']] if item.get('terminal_strip_tag') else (item['tags'] or [item['tag']])
                item['tag_members']=list(dict.fromkeys(member for value in identities for member in tag_members(value) if member))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """Restore persisted edits, evidence and review state without re-extraction."""
        def construct(kind, values):
            allowed = {f.name for f in fields(kind)}
            return kind(**{k: v for k, v in values.items() if k in allowed})

        values = dict(data)
        values["pages"] = [construct(PageInfo, p) for p in data.get("pages", [])]
        for key in ("itemized_bom", "aggregated_bom", "filtered_items", "independent_bom", "declared_parts"):
            values[key] = []
            for raw in data.get(key, []):
                item = dict(raw)
                item["evidences"] = [construct(EvidenceLocation, ev) for ev in raw.get("evidences", [])]
                values[key].append(construct(BOMItem, item))
        values["review_items"] = [construct(ReviewItem, r) for r in data.get("review_items", [])]
        return construct(cls, values)
