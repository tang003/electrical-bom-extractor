"""Headless DWG/DXF ingestion with native text evidence and a paginated review PDF.

The PDF is a view, not the source of CAD facts. Its versioned attachment retains
native text/attributes, entity handles, block paths and the CAD-to-page transform.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ATTACHMENT = "electrical_bom_cad_native_v1.json"
NATIVE_MARKER = "electrical-bom-cad-native-v1"
_IMPORT_LOCK = threading.Lock()
MAX_CAD_ENTITIES = 300_000
MAX_CAD_SHEETS = 200


class CadImportError(ValueError):
    def __init__(self, message: str, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _converter_command():
    configured = os.environ.get("CAD_DWG_COMMAND_JSON")
    if configured:
        try:
            command = json.loads(configured)
            if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
                raise ValueError()
            if not any("{input}" in s for s in command) or not any("{output}" in s for s in command):
                raise ValueError()
            return command, "configured"
        except (ValueError, TypeError):
            raise CadImportError("服务器 CAD_DWG_COMMAND_JSON 配置无效，需要含 {input} 与 {output} 的 JSON 参数数组。", 503)
    dll = Path(os.environ.get("CAD_CONVERTER_DLL", ROOT / "tools/cad_converter/publish/ElectricalBom.CadConverter.dll"))
    dotnet = shutil.which("dotnet")
    if dll.is_file() and dotnet:
        return [dotnet, str(dll.resolve()), "{input}", "{output}", "{report}"], "ACadSharp"
    raise CadImportError("服务器尚未配置 DWG 解析器。请部署 .NET 8 CAD 转换服务并构建 tools/cad_converter；也可先上传 DXF 或 PDF。", 503)


def drawing_capabilities():
    try:
        import ezdxf  # noqa: F401
        dxf = {"available": True}
    except ImportError:
        dxf = {"available": False, "reason": "服务器缺少 ezdxf，请安装项目依赖。"}
    try:
        _, provider = _converter_command()
        dwg = {"available": dxf["available"], "provider": provider, "reason": dxf.get("reason", "")}
    except CadImportError as exc:
        dwg = {"available": False, "reason": str(exc)}
    return {"pdf": {"available": True}, "dxf": dxf, "dwg": dwg}


def _dwg_to_dxf(source: Path, folder: Path):
    with source.open("rb") as stream:
        version = stream.read(6).decode("ascii", errors="replace")
    if version not in {"AC1014", "AC1015", "AC1018", "AC1021", "AC1024", "AC1027", "AC1032"}:
        raise CadImportError(f"无法识别或暂不支持此 DWG 版本（{version}），请核实源文件或另存为 DWG 2018 / DXF。")
    command, provider = _converter_command()
    target, report_path = folder / "drawing.dxf", folder / "converter.json"
    replacements = {"{input}": str(source.resolve()), "{output}": str(target.resolve()), "{report}": str(report_path.resolve())}
    argv = [arg for arg in command]
    for key, value in replacements.items():
        argv = [arg.replace(key, value) for arg in argv]
    try:
        timeout = max(5, min(300, int(os.environ.get("CAD_CONVERT_TIMEOUT", "90"))))
    except ValueError:
        raise CadImportError("服务器 CAD_CONVERT_TIMEOUT 配置无效。", 503)
    try:
        run = subprocess.run(argv, cwd=folder, capture_output=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise CadImportError("DWG 转换超时，尚未创建可审核项目。请检查图纸复杂度或服务器转换配置。", 422) from exc
    except OSError as exc:
        raise CadImportError("服务器 DWG 转换器无法启动，请检查 CAD 服务部署。", 503) from exc
    (folder / "converter.log").write_bytes(run.stdout + run.stderr)
    if run.returncode or not target.is_file() or target.stat().st_size == 0:
        raise CadImportError("DWG 读取失败，转换日志已保留。请核实文件是否损坏、加密或包含不支持的对象。", 422)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.is_file() else {}
    except (ValueError, UnicodeError):
        raise CadImportError("DWG 转换器返回的导入报告无效。", 422)
    return target, {"provider": provider, "dwg_version": version, **report}


@dataclass
class Sheet:
    layout: str
    bbox: object
    title: str
    handle: str = ""
    attributes: dict = field(default_factory=dict)
    overlapping_handles: list = field(default_factory=list)


def _rectangles(entities):
    from ezdxf.math import BoundingBox2d
    for e in entities:
        if e.dxftype() != "LWPOLYLINE" or not e.closed or len(e) not in (4, 5):
            continue
        points = list(e.vertices())
        box = BoundingBox2d(points)
        if box.size.x <= 0 or box.size.y <= 0:
            continue
        corners = {(round(p.x, 5), round(p.y, 5)) for p in box.rect_vertices()}
        if all((round(p[0], 5), round(p[1], 5)) in corners for p in points):
            yield box, e


def detect_model_sheets(doc):
    from ezdxf import bbox as ezbbox
    from ezdxf.math import BoundingBox2d, Vec3
    sheets = []
    for insert in doc.modelspace().query("INSERT"):
        if not _visible(doc,insert):
            continue
        block = insert.block()
        if block is None or block.block.is_xref:
            continue
        attrs = {a.dxf.tag: a.plain_text() for a in insert.attribs}
        is_frame = re.search(r"(?:^A[0-5]$|图框|图幅|FRAME|TITLE|SHEET|BORDER)", insert.dxf.name, re.I)
        if not is_frame and not any(k.upper() in {"图名", "图号", "SHEET", "SHEET_NO", "DRAWING_NO"} for k in attrs):
            continue
        boxes = list(_rectangles(block))
        if boxes:
            raw = max(boxes, key=lambda pair: pair[0].size.x * pair[0].size.y)[0]
        else:
            geometry = [e for e in block if e.dxftype() not in {"TEXT", "MTEXT", "ATTDEF", "ATTRIB"}]
            extents = ezbbox.extents(geometry, fast=True)
            if not extents.has_data:
                continue
            raw = BoundingBox2d([extents.extmin, extents.extmax])
        box = BoundingBox2d(insert.matrix44().transform(Vec3(v)) for v in raw.rect_vertices())
        if box.size.x <= 0 or box.size.y <= 0 or not .45 <= box.size.x / box.size.y <= 2.5:
            continue
        if any(abs(s.bbox.extmin.x-box.extmin.x) < .001 and abs(s.bbox.extmin.y-box.extmin.y) < .001
               and abs(s.bbox.size.x-box.size.x) < .001 and abs(s.bbox.size.y-box.size.y) < .001 for s in sheets):
            continue
        title = next((attrs[k] for k in ("图名", "图纸名称", "TITLE", "SHEET_NAME") if attrs.get(k)), insert.dxf.name)
        sheets.append(Sheet("Model", box, title, insert.dxf.handle, attrs))
    if not sheets:
        rectangles = list(_rectangles(doc.modelspace()))
        if rectangles:
            largest = max(b.size.x*b.size.y for b, _ in rectangles)
            candidates = [(b,e) for b,e in rectangles if b.size.x*b.size.y >= largest*.9 and .65 < b.size.x/b.size.y < 1.6]
            sheets = [Sheet("Model", b, "模型空间图框", e.dxf.handle) for b,e in candidates]
    # Coincident inner/outer borders describe one physical sheet, not versions.
    unique=[]
    for sheet in sorted(sheets,key=lambda s:-(s.bbox.size.x*s.bbox.size.y)):
        match=next((s for s in unique if s.layout==sheet.layout and s.title==sheet.title and s.attributes==sheet.attributes and
            s.bbox.extmin.x-.001<=sheet.bbox.extmin.x and s.bbox.extmin.y-.001<=sheet.bbox.extmin.y and
            s.bbox.extmax.x+.001>=sheet.bbox.extmax.x and s.bbox.extmax.y+.001>=sheet.bbox.extmax.y and
            abs(s.bbox.center.x-sheet.bbox.center.x)<s.bbox.size.x*.05 and
            abs(s.bbox.center.y-sheet.bbox.center.y)<s.bbox.size.y*.05 and
            .87<(sheet.bbox.size.x*sheet.bbox.size.y)/(s.bbox.size.x*s.bbox.size.y)<=1.001),None)
        if match:match.overlapping_handles.append(sheet.handle)
        else:unique.append(sheet)
    sheets=unique
    if len(sheets) > MAX_CAD_SHEETS:
        raise CadImportError(f"CAD 图框超过 {MAX_CAD_SHEETS} 个，请按项目拆分后导入。", 422)
    # Spatial rows; duplicate titles are deliberately retained as separate sheets.
    ordered = []
    remaining = sorted(sheets, key=lambda s: -s.bbox.center.y)
    while remaining:
        first = remaining[0]
        row = [s for s in remaining if abs(s.bbox.center.y-first.bbox.center.y) < min(s.bbox.size.y,first.bbox.size.y)*.25]
        ordered.extend(sorted(row, key=lambda s: s.bbox.center.x))
        remaining = [s for s in remaining if all(s is not r for r in row)]
    return ordered


def _visible(doc, e, inherited="0"):
    if e.dxf.get("invisible", 0):
        return False
    name = e.dxf.get("layer", "0")
    name = inherited if name == "0" else name
    try:
        layer = doc.layers.get(name)
        return not layer.is_off() and not layer.is_frozen()
    except Exception:
        return True


def native_texts(doc, entities, warnings):
    from ezdxf import disassemble
    from ezdxf.math import BoundingBox2d
    records = []
    count = 0
    def visit(entities, chain=(), inherited="0", depth=0):
        nonlocal count
        if depth > 16:
            raise CadImportError("CAD 嵌套块超过支持深度，请整理图纸块后重试。", 422)
        for e in entities:
            count += 1
            if count > MAX_CAD_ENTITIES:
                raise CadImportError("CAD 展开实体过多，请按子项目拆分图纸。", 422)
            if not _visible(doc,e,inherited):
                continue
            original = getattr(e, "source_of_copy", None) or e
            handle = original.dxf.get("handle", "") or ""
            kind = e.dxftype()
            layer = e.dxf.get("layer", inherited)
            layer = inherited if layer == "0" else layer
            if kind == "INSERT":
                block = e.block()
                if block is None or block.block.is_xref:
                    warnings.add("存在外部参照或缺失块；外部文件未随单文件上传，需核对原图。")
                    continue
                path = (*chain, f"{e.dxf.name}:{handle}")
                if e.mcount > 1000:
                    raise CadImportError("CAD 阵列规模过大，请拆分后导入。", 422)
                if e.mcount > 1:
                    visit(e.multi_insert(),chain,inherited,depth+1)
                else:
                    visit(e.attribs,path,layer,depth+1)
                    visit(e.virtual_entities(),path,layer,depth+1)
                continue
            if kind in {"DIMENSION", "LEADER", "MLEADER", "MLINE"} and hasattr(e,"virtual_entities"):
                visit(e.virtual_entities(),(*chain,f"{kind}:{handle}"),layer,depth+1)
                continue
            if kind not in {"TEXT", "MTEXT", "ATTRIB"}:
                continue
            text = e.plain_text().strip()
            if not text:
                continue
            try:
                extent = disassemble.make_primitive(e).bbox()
                if not extent.has_data:
                    raise ValueError("empty text geometry")
                box = BoundingBox2d([extent.extmin,extent.extmax])
                if not all(math.isfinite(v) for v in (*box.extmin,*box.extmax)):
                    raise ValueError("invalid text geometry")
            except Exception:
                warnings.add("部分 CAD 文字无法建立几何包围框，需在原图核实。")
                continue
            records.append({"text":text,"handle":handle,"block_path":list(chain),"layer":layer,
                            "entity_type":kind,"attribute_tag":e.dxf.tag if kind=="ATTRIB" else "",
                            "cad_bbox":[box.extmin.x,box.extmin.y,box.extmax.x,box.extmax.y]})
    visit(entities)
    return records


def _spans_on_sheet(records, sheet, matrix, width, height):
    from ezdxf.math import BoundingBox2d, Vec2
    spans = []
    for raw in records:
        x0,y0,x1,y1=raw.get("sheet_bbox",raw["cad_bbox"])
        center=Vec2((x0+x1)/2,(y0+y1)/2)
        if not sheet.bbox.inside(center):
            continue
        box=BoundingBox2d(matrix.fast_2d_transform(((x0,y0),(x1,y0),(x1,y1),(x0,y1))))
        bx=[max(0,min(width,box.extmin.x)),max(0,min(height,box.extmin.y)),max(0,min(width,box.extmax.x)),max(0,min(height,box.extmax.y))]
        if bx[2] <= bx[0] or bx[3] <= bx[1]:
            continue
        lines=raw["text"].splitlines() or [raw["text"]]
        for i,text in enumerate(lines):
            if not text.strip():continue
            line=[bx[0],bx[1]+(bx[3]-bx[1])*i/len(lines),bx[2],bx[1]+(bx[3]-bx[1])*(i+1)/len(lines)]
            spans.append({**raw,"text":text.strip(),"bbox":[round(v,3) for v in line],
                          "norm_bbox":[round(line[0]/width,6),round(line[1]/height,6),round(line[2]/width,6),round(line[3]/height,6)],
                          "cx":(line[0]+line[2])/2,"cy":(line[1]+line[3])/2,"size":max(.1,min(line[2]-line[0],line[3]-line[1])),
                          "layout":sheet.layout,"cad_sheet_handle":sheet.handle,"source":"cad_native"})
    return spans


def _label_cabinet_sheet_sets(pages, outside):
    """Associate an explicit cabinet heading above a numbered set of model-space sheets."""
    # The same drawing number/title may occur in revisions. A page-number reset
    # alone is insufficient evidence of a separate cabinet.
    series = []
    for page in pages:
        title = re.fullmatch(r"(.+?)\s*[（(]\s*(\d+)\s*[)）]\s*", page["title"])
        if page["layout"] != "Model" or not title:
            return
        stem, number = title[1].strip(), int(title[2])
        if number == 1:
            series.append([])
        if not series or number != len(series[-1]) + 1:
            return
        if series[-1] and stem != series[-1][0][0]:
            return
        series[-1].append((stem, page))
    if len(series) < 2 or any(len(group) < 2 for group in series):
        return
    headings = []
    for source in outside:
        caption = re.fullmatch(r"\s*([\w-]{1,20}(?:柜|盘|箱))\s*(?:[xX×*]\s*(\d{1,3})\s*(?:套|台)?)?\s*", source["text"])
        if caption:
            headings.append((caption[1], source))
    resolved = []
    for group in series:
        first = group[0][1]["cad_bbox"]
        height = first[3] - first[1]
        first_row = [page["cad_bbox"] for _, page in group if abs(page["cad_bbox"][3] - first[3]) < height / 4]
        left, right = min(b[0] for b in first_row), max(b[2] for b in first_row)
        matches = []
        for name, source in headings:
            x0, y0, x1, y1 = source["cad_bbox"]
            x, y = (x0+x1)/2, (y0+y1)/2
            if left <= x <= right and first[3] < y <= first[3] + height * 1.5:
                matches.append((name, source))
        if len(matches) != 1:
            return
        resolved.append(matches[0])
    if len({name for name, _ in resolved}) != len(series):
        return
    for group, (name, source) in zip(series, resolved):
        for _, page in group:
            page["cabinet_heading"] = {"name": name, "text": source["text"], "cad_bbox": source["cad_bbox"],
                                       "handle": source.get("handle", "")}


def prepare_cad(source: Path, output_dir: Path, display_name: str | None = None) -> Path:
    """Return the view PDF; keep source, conversion logs and native evidence alongside it."""
    try:
        import ezdxf
        from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, pymupdf as pm
        from ezdxf.addons.drawing.recorder import Player
        from ezdxf.fonts import fonts
        from ezdxf.math import BoundingBox2d
    except ImportError as exc:
        raise CadImportError("服务器 CAD 依赖未安装，请安装 requirements.txt 中的 ezdxf。",503) from exc
    source, output_dir = Path(source).resolve(),Path(output_dir).resolve()
    if source.suffix.lower() not in {".dwg",".dxf"}:
        raise CadImportError("CAD 导入仅接受 DWG 或 DXF 文件。")
    folder=output_dir/"cad";folder.mkdir(parents=True,exist_ok=True)
    if (output_dir/"source.pdf").exists():
        raise CadImportError("此项目已有审核视图，请使用独立目录导入 CAD。")
    with _IMPORT_LOCK:
        dxf, conversion = _dwg_to_dxf(source,folder) if source.suffix.lower()==".dwg" else (source,{"provider":"ezdxf","version":ezdxf.__version__})
        try:
            doc=ezdxf.readfile(dxf)
        except Exception as exc:
            raise CadImportError("DXF 内容无法读取，请核实文件格式或重新导出。",422) from exc
        if len(doc.modelspace()) > MAX_CAD_ENTITIES:
            raise CadImportError("CAD 实体过多，请按子项目拆分图纸。",422)
        warnings=set()
        entity_types=Counter(e.dxftype() for space in doc.layouts for e in space)
        unsupported=set(entity_types)&{"IMAGE","PDFUNDERLAY","DGNUNDERLAY","DWFUNDERLAY","ACAD_PROXY_ENTITY","3DSOLID","BODY","REGION","OLE2FRAME","ACAD_TABLE"}
        if unsupported:
            warnings.add("包含当前二维预览未完整支持的对象："+"、".join(sorted(unsupported))+"；请核对原始 CAD。")
        notices=sum(conversion.get("notification_counts",{}).values())
        if notices:
            warnings.add(f"DWG 读取器报告 {notices} 条对象或引用兼容性提醒，需核对预览与原文件；详见转换报告。")
        # Use a portable CJK face bundled with the already installed PyMuPDF runtime.
        # Never require fonts from the developer's AutoCAD installation.
        font_file=folder/"fonts"/"CadFallbackCJK.ttf";font_file.parent.mkdir(exist_ok=True)
        font_file.write_bytes(pymupdf.Font("china-s").buffer)
        fonts.font_manager.build([str(font_file.parent)],support_dirs=False)
        original_fonts={s.dxf.name:{"font":s.dxf.font,"bigfont":s.dxf.bigfont} for s in doc.styles}
        for style in doc.styles:
            style.dxf.font=font_file.name;style.dxf.bigfont=""
        warnings.add("CAD 审核视图使用替代字体，文字位置和排版需对照源图核实。")
        for block in doc.blocks:
            for entity in block.query("MTEXT"):
                entity.text=entity.plain_text()
        frames=detect_model_sheets(doc)
        papers=[s for s in doc.layouts if s.name!="Model" and any(e.dxftype()!="VIEWPORT" for e in s)]
        # Prefer explicit model-space drawing frames; a viewport containing many
        # such frames is not itself a single engineering sheet.
        target_layouts=[doc.modelspace()] if frames or not papers else papers
        all_native={}
        output=pymupdf.open()
        page_records=[]
        cfg=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,color_policy=config.ColorPolicy.BLACK,
                                 image_policy=config.ImagePolicy.IGNORE,hatching_timeout=5)
        for space in target_layouts:
            texts=native_texts(doc,space,warnings)
            if space.name!="Model" and any(v.dxf.get("id",1)!=1 for v in space.query("VIEWPORT")):
                for viewport in space.query("VIEWPORT"):
                    if viewport.dxf.get("id",1)==1:continue
                    try:
                        matrix=viewport.get_transformation_matrix()
                        model_texts=native_texts(doc,doc.modelspace(),warnings)
                        viewbox=BoundingBox2d([(viewport.dxf.center.x-viewport.dxf.width/2,viewport.dxf.center.y-viewport.dxf.height/2),
                                              (viewport.dxf.center.x+viewport.dxf.width/2,viewport.dxf.center.y+viewport.dxf.height/2)])
                        for item in model_texts:
                            a,b,c,d=item['cad_bbox'];b2=BoundingBox2d(matrix.fast_2d_transform(((a,b),(c,b),(c,d),(a,d))))
                            if viewbox.inside(b2.center):
                                texts.append({**item,'sheet_bbox':[b2.extmin.x,b2.extmin.y,b2.extmax.x,b2.extmax.y],
                                              'block_path':[f'VIEWPORT:{viewport.dxf.handle}',*item['block_path']]})
                    except Exception:
                        warnings.add("某个布局视口的原生文字映射未完成，需核对该布局。")
            all_native[space.name]=texts
            backend=pm.PyMuPdfBackend()
            Frontend(RenderContext(doc),backend,config=cfg).draw_layout(space,finalize=True)
            base=backend.player()
            if not base.bbox().has_data:
                continue
            selected=frames if space.name=="Model" and frames else [Sheet(space.name,base.bbox(),space.name)]
            if not frames and space.name=="Model":
                warnings.add("未检测到可分离图框，模型空间保留为一张总览；请核对分页与文字可读性。")
            if len(page_records)+len(selected)>MAX_CAD_SHEETS:
                raise CadImportError("CAD 布局或图框过多，请拆分后导入。",422)
            record_boxes=[(r,r.bbox()) for r in base.records]
            for sheet in selected:
                box=sheet.bbox
                ratio=box.size.x/box.size.y if box.size.y else 1
                if not math.isfinite(ratio) or ratio<=0:
                    raise CadImportError("CAD 图框坐标异常。",422)
                page=layout.Page(420,420/ratio) if ratio>=1 else layout.Page(297*ratio,297)
                settings=layout.Settings(output_coordinate_space=pm.get_coordinate_output_space(page))
                player=Player();player.config=base.config;player.background=base.background;player.properties=base.properties
                player.records=copy.deepcopy([r for r,b in record_boxes if box.has_intersection(b)])
                player.crop_rect(box.extmin,box.extmax,max(box.size.x,box.size.y)/100_000)
                transform=layout.Layout(box,flip_y=True).get_placement_matrix(page,settings)
                player.transform(transform)
                rendered=pm.PyMuPdfRenderBackend(page,settings);player.replay(rendered)
                with pymupdf.open(stream=rendered.get_pdf_bytes(),filetype="pdf") as one:
                    width,height=one[0].rect.width,one[0].rect.height
                    output.insert_pdf(one)
                spans=_spans_on_sheet(texts,sheet,transform,width,height)
                page_records.append({"page":len(page_records)+1,"layout":space.name,"title":sheet.title,"frame_handle":sheet.handle,
                                     "overlapping_frame_handles":sheet.overlapping_handles,
                                     "attributes":sheet.attributes,"cad_bbox":[box.extmin.x,box.extmin.y,box.extmax.x,box.extmax.y],
                                     "width":width,"height":height,"spans":spans})
        if not page_records:
            output.close();raise CadImportError("CAD 中没有可渲染的二维图形或文字。",422)
        duplicates={name:n for name,n in Counter(s.title for s in frames).items() if n>1 and name!='模型空间图框'}
        if duplicates:
            warnings.add("存在同名 CAD 图框，已逐框保留；请确认有效图纸和版本，避免重复采购。")
        outside=[]
        if frames:
            for item in all_native.get("Model",[]):
                a,b,c,d=item["cad_bbox"]
                if not any(s.bbox.inside(((a+c)/2,(b+d)/2)) for s in frames):outside.append(item)
            if outside:warnings.add(f"图框外还有 {len(outside)} 条原生文字，已保存在导入报告，未自动计入分页 BOM。")
        _label_cabinet_sheet_sets(page_records, outside)
        meta={"source_format":source.suffix.lower().lstrip("."),"source_filename":display_name or source.name,
              "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"converter":conversion,
              "page_count":len(page_records),"frame_count":len(frames),"native_text_count":sum(len(v) for v in all_native.values()),
              "model_entity_count":len(doc.modelspace()),"entity_types":dict(entity_types),
              "warnings":sorted(warnings),"duplicate_titles":duplicates,"outside_text_count":len(outside),
              "requires_sheet_review":bool(duplicates or outside),"original_fonts":original_fonts,
              "text_method":"cad_native_entities_and_attributes","view_method":"ezdxf_2d_review_preview"}
        payload={"schema_version":1,"source":meta,"pages":page_records}
        native_bytes=json.dumps(payload,ensure_ascii=False).encode("utf-8")
        output.embfile_add(NATIVE_ATTACHMENT,native_bytes,filename=NATIVE_ATTACHMENT,desc="Native CAD entity evidence and page coordinates")
        output.set_metadata({"creator":"Electrical BOM CAD importer","keywords":NATIVE_MARKER,"title":display_name or source.name})
        target=output_dir/"source.pdf"
        try:output.save(target,deflate=True,garbage=4)
        finally:output.close()
        (folder/"native_entities.json").write_text(json.dumps({"source":meta,"layouts":all_native,"outside_frame_texts":outside},ensure_ascii=False,indent=2),encoding="utf-8")
        (folder/"import_report.json").write_text(json.dumps({"source":meta,"pages":[{k:v for k,v in p.items() if k!="spans"} for p in page_records],"outside_frame_texts":outside},ensure_ascii=False,indent=2),encoding="utf-8")
        return target


def native_document(doc):
    """Return validated CAD facts only for our versioned review-view container."""
    if hasattr(doc,"_electrical_bom_native_cache"):
        return doc._electrical_bom_native_cache
    payload=None
    if NATIVE_MARKER in (doc.metadata or {}).get("keywords",""):
        try:
            data=doc.embfile_get(NATIVE_ATTACHMENT)
            if len(data)>32*1024*1024:raise ValueError("native evidence is too large")
            candidate=json.loads(data)
            if candidate.get("schema_version")!=1 or len(candidate["pages"])!=len(doc):raise ValueError("invalid schema")
            if candidate['source'].get('source_format') not in {'dwg','dxf'}:raise ValueError('invalid source format')
            for p,page in zip(candidate["pages"],doc):
                if p["page"]!=page.number+1 or len(p["spans"])>20_000:raise ValueError("invalid page")
                for s in p["spans"]:
                    b=s["bbox"]
                    if not isinstance(s["text"],str) or len(b)!=4 or not all(math.isfinite(v) for v in b):raise ValueError("invalid span")
                    if not 0<=b[0]<b[2]<=page.rect.width+.01 or not 0<=b[1]<b[3]<=page.rect.height+.01:raise ValueError("invalid geometry")
                    nb=s['norm_bbox']
                    if len(nb)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in nb):raise ValueError('invalid normalized coordinates')
                    expected=[b[0]/page.rect.width,b[1]/page.rect.height,b[2]/page.rect.width,b[3]/page.rect.height]
                    if any(abs(a-v)>.002 for a,v in zip(nb,expected)):raise ValueError('inconsistent coordinates')
            payload=candidate
        except (ValueError,KeyError,TypeError,RuntimeError):
            raise CadImportError("CAD 审核视图的原生证据损坏，请从原始 DWG/DXF 重新导入。",422)
    doc._electrical_bom_native_cache=payload
    return payload


def cad_page_spans(page):
    native=native_document(page.parent)
    return native["pages"][page.number]["spans"] if native else None


def cad_join_tag_fragments(spans):
    """Some DXF writers emit overlapping TEXT glyph runs for one identifier."""
    from .notation import CAD_UNPREFIXED_TAG_RE
    fragments = [s for s in spans if s.get("source") == "cad_native" and s.get("entity_type") == "TEXT"]
    joined = []
    for first in fragments:
        if not re.fullmatch("[A-Z]{1,2}", first["text"]):
            continue
        a = first["bbox"]
        height = a[3] - a[1]
        neighbors = []
        for last in fragments:
            b = last["bbox"]
            if (last.get("layer") != first.get("layer") or last.get("block_path") != first.get("block_path")
                    or abs(b[1]-a[1]) > .06*height or abs(b[3]-a[3]) > .06*height
                    or not a[0] < b[0] <= a[2]+.05*height or b[0] < a[2]-.50*height):
                continue
            candidate = first["text"] + last["text"]
            if CAD_UNPREFIXED_TAG_RE.fullmatch(candidate) and not CAD_UNPREFIXED_TAG_RE.fullmatch(last["text"]):
                neighbors.append((last, candidate))
        if len(neighbors) != 1:
            continue
        last, candidate = neighbors[0]
        merged = {**first, "text":candidate, "source_text":candidate}
        for key in ("bbox", "norm_bbox"):
            a,b = first[key],last[key]
            merged[key] = [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
        merged["cx"] = (merged["bbox"][0]+merged["bbox"][2])/2
        merged["cy"] = (merged["bbox"][1]+merged["bbox"][3])/2
        joined.append(merged)
    return spans + joined


def cad_stacked_model_owner(model, anchors):
    """Find a tag and part code drawn as two aligned lines of one CAD label."""
    if model.get("source") != "cad_native" or not model.get("layer"):
        return None
    code = model["model_text"]
    literal = model["text"].strip().strip("()（）<>《》").strip()
    clean = re.sub(r"\s+(?:\d+(?:\.\d+)?\s*(?:A|V|KW|W|HZ)|\d+P)(?:\s+\d+P)?$", "", literal, flags=re.I)
    if clean.upper() != code.upper() or not re.search(r"[A-Z]", code, re.I) or not re.search(r"\d", code):
        return None
    # Contact / terminal addresses can also sit next to a device tag.
    if len(code) < 5 or re.fullmatch(r"[A-Z]+\d+[-.]\d+[A-Z]?|.+[-.](?:NC|NO|COM)", code, re.I):
        return None
    box = model["bbox"]
    height = box[3] - box[1]
    if box[2] - box[0] < height or height <= 0:
        return None
    owners = set()
    for anchor in anchors:
        if anchor.get("source") != "cad_native" or anchor.get("layer") != model["layer"]:
            continue
        if anchor.get("block_path", []) != model.get("block_path", []):
            continue
        other = anchor["bbox"]
        other_height = other[3] - other[1]
        if not .7 * height <= other_height <= 1.4 * height or other[2] - other[0] < other_height:
            continue
        line_height = max(height, other_height)
        gap = max(box[1] - other[3], other[1] - box[3])
        alignment = min(abs(model["cx"] - anchor["cx"]), abs(box[0] - other[0]), abs(box[2] - other[2]))
        if -.25 * line_height <= gap <= .65 * line_height and alignment <= .38 * line_height:
            owners.add(anchor["tag"])
    return next(iter(owners)) if len(owners) == 1 else None


def select_cad_sheets(source_pdf: Path, target_pdf: Path, pages: list[int]):
    """Make a new view project; never overwrite the original project's review state."""
    with pymupdf.open(source_pdf) as source:
        payload=native_document(source)
        if not payload:raise CadImportError("该项目不是 CAD 导入项目。")
        selected=sorted(set(pages))
        if not selected or any(type(p) is not int or not 1<=p<=len(source) for p in selected):
            raise CadImportError("请选择有效的 CAD 图框页码。")
        result=copy.deepcopy(payload)
        chosen=[copy.deepcopy(payload['pages'][p-1]) for p in selected]
        for i,p in enumerate(chosen,1):
            p['source_page']=p.get('source_page',p['page']);p['page']=i
        titles=Counter(p['title'] for p in chosen)
        duplicates={name:n for name,n in titles.items() if n>1}
        meta=result['source']
        meta['selected_source_pages']=[p['source_page'] for p in chosen]
        meta['original_page_count']=meta.get('original_page_count',len(source))
        meta['page_count']=len(chosen);meta['frame_count']=len(chosen)
        meta['native_text_count']=sum(len(p['spans']) for p in chosen)
        meta['duplicate_titles']=duplicates;meta['requires_sheet_review']=bool(duplicates)
        meta['warnings']=[w for w in meta['warnings'] if not w.startswith('存在同名 CAD 图框')]
        if duplicates:meta['warnings'].append('所选范围仍有同名 CAD 图框，请核对有效版本。')
        meta['warnings'].append('此项目按用户所选图框建立，源项目的审核结果保持独立；采购范围与数量仍需确认。')
        result['pages']=chosen
        with pymupdf.open() as output:
            for p in selected:output.insert_pdf(source,from_page=p-1,to_page=p-1)
            output.set_metadata(source.metadata)
            output.embfile_add(NATIVE_ATTACHMENT,json.dumps(result,ensure_ascii=False).encode('utf-8'),filename=NATIVE_ATTACHMENT)
            output.save(target_pdf,deflate=True,garbage=4)
        return result


def attach_cad_evidence(result, payload):
    """Link existing BOM evidence to native entity handles without assigning models."""
    if not payload:return
    result.source_format=payload["source"]["source_format"]
    result.source_filename=payload["source"]["source_filename"]
    result.summary["cad_import"]={k:v for k,v in payload["source"].items() if k!="original_fonts"}
    for info,raw in zip(result.pages,payload["pages"]):
        info.title=raw["title"]
        info.extraction_diagnostics["cad"]={k:v for k,v in raw.items() if k!="spans"}
    for item in [*result.itemized_bom,*result.aggregated_bom,*result.filtered_items,*result.independent_bom,*result.declared_parts]:
        for ev in item.evidences:
            if ev.cad_refs or not 1<=ev.page<=len(payload["pages"]):continue
            text=re.sub(r"\s+","",ev.raw_text).upper()
            found={}
            for s in payload["pages"][ev.page-1]["spans"]:
                t=re.sub(r"\s+","",s["text"]).upper()
                a,b,c,d=s["norm_bbox"];x,y,z,w=ev.norm_bbox
                if not (a<z and c>x and b<w and d>y) or not (t and (t in text or text in t)):continue
                key=(s["handle"],tuple(s["block_path"]))
                found[key]={k:s[k] for k in ("handle","block_path","layer","layout","cad_bbox","entity_type","attribute_tag")}
                if len(found)>=30:break
            ev.cad_refs=list(found.values())
