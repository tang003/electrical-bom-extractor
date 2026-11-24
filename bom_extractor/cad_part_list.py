"""Read CAD parts tables from detected column headers, independent of model brands."""
from __future__ import annotations

import re
import copy
import pymupdf

from .models import BOMItem, EvidenceLocation, ReviewItem, SourceType
from .notation import extract_model_strings

HEADERS = {
    "seq": {"序号", "序次", "编号", "NO.", "NO", "ITEM"},
    "tag": {"代号", "位号", "符号", "记号", "设备代号", "元件代号", "设备标识符"},
    "name": {"名称", "品名", "元件名称", "物料名称", "DESCRIPTION", "NAME"},
    "model": {"型号", "型号规格", "规格型号", "型式仕样", "型式仕様", "订货编号", "订货号", "MODEL", "PARTNUMBER"},
    "specification": {"规格", "参数", "技术参数"},
    "quantity": {"数量", "用量", "数目", "QTY", "QUANTITY"},
    "unit": {"单位", "计量单位"},
    "brand": {"品牌", "厂家", "制造商", "生产厂家"},
    "remark": {"备注", "说明"},
}


def table_spans(page,spans):
    """Read row/column layout in physical PDF coordinates even on rotated pages."""
    if not page.rotation:return spans
    physical=page.rect*page.derotation_matrix
    result=copy.deepcopy(spans)
    for span in result:
        r=pymupdf.Rect(span['bbox'])*page.derotation_matrix
        span.update(bbox=list(r),cx=(r.x0+r.x1)/2,cy=(r.y0+r.y1)/2,
                    norm_bbox=[r.x0/physical.width,r.y0/physical.height,r.x1/physical.width,r.y1/physical.height])
    return result


def table_model_tokens(value):
    """Keep every literal order-code fragment, including both sides of assemblies."""
    tokens=[]
    for part in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9.]+)*",value):
        compact=part.upper()
        if re.fullmatch(r"(?:AC|DC)?[0-9./~+*-]+(?:KVA|KW|MM2|MM|HZ|MA|V|A|W|P)(?:AC|DC)?",compact):
            continue
        if re.fullmatch(r"(?:(?:AC|DC)?\d+(?:\.\d+)?[VAWPKHZ]+/?)+",compact):
            continue
        found=extract_model_strings(part)
        if not found and re.fullmatch(r"[A-Za-z]\d{2,}[A-Za-z][A-Za-z0-9]*",part):
            found=[part]
        tokens.extend(found)
    return list(dict.fromkeys(tokens))


def detect_parts_columns(spans,page=None):
    spans=[s for s in spans if all(k in s for k in ('text','cx','cy','size','bbox'))]
    lines = []
    for span in sorted(spans, key=lambda s: s["cy"]):
        row = next((r for r in lines if abs(r[0]["cy"]-span["cy"]) <= max(r[0]["size"], span["size"])*.45), None)
        if row is None:lines.append([span])
        else:row.append(span)
    for row in lines:
        row.sort(key=lambda s:s["cx"])
        found={}
        for start in range(len(row)):
            for end in range(min(len(row),start+7),start,-1):
                text=re.sub(r"\s+","","".join(s["text"] for s in row[start:end])).upper()
                for key, names in HEADERS.items():
                    if text in names and key not in found:
                        x0=min(s["bbox"][0] for s in row[start:end]);x1=max(s["bbox"][2] for s in row[start:end])
                        found[key]={"center":(x0+x1)/2,"label":text}
        if {"name","quantity"} <= found.keys() and ('seq' in found or 'tag' in found) and ("model" in found or "specification" in found):
            if "model" not in found:found["model"]=found.pop("specification")
            ordered=sorted(found.items(),key=lambda p:p[1]["center"])
            centers=[v["center"] for _,v in ordered]
            columns={}
            for i,(key,value) in enumerate(ordered):
                columns[key]={**value,"left":(centers[i-1]+centers[i])/2 if i else centers[i]-(centers[1]-centers[0])/2,
                              "right":(centers[i]+centers[i+1])/2 if i<len(centers)-1 else centers[i]+(centers[i]-centers[i-1])/2}
            header_y=sum(s['cy'] for s in row)/len(row)
            if page is not None:
                rectangles=[]
                for drawing in page.get_drawings():
                    for part in drawing['items']:
                        if part[0]=='re':rectangles.append(part[1])
                        elif part[0]=='qu':rectangles.append(part[1].rect)
                size=max(s['size'] for s in row)
                for column in columns.values():
                    cells=[r for r in rectangles if r.x0<column['center']<r.x1 and r.y0<header_y<r.y1 and
                           size<r.height<size*6 and r.width>size*3]
                    if cells:
                        r=min(cells,key=lambda r:r.width*r.height)
                        column.update(left=r.x0,right=r.x1)
            return {"columns":columns,"header_bottom":max(s["bbox"][3] for s in row),"header_y":header_y}
    return None


def row_anchors(spans,detected,height):
    columns=detected['columns']
    key='seq' if 'seq' in columns else 'name'
    column=columns[key]
    candidates=sorted([s for s in spans if column['left']<=s['cx']<column['right'] and
        detected['header_bottom']<s['cy']<height*.9 and (key!='seq' or re.fullmatch(r'\d{1,4}',s['text'].strip()))],key=lambda s:s['cy'])
    anchors=[]
    for span in candidates:
        if not anchors or abs(span['cy']-anchors[-1]['cy'])>max(span['size'],anchors[-1]['size'])*.8:
            anchors.append(span)
    return anchors


def extract_cad_part_lists(doc, pages_info, *, reference_only=False):
    from .classifier import extract_visual_spans
    items,reviews=[],[]
    all_spans={info.page_num:extract_visual_spans(doc[info.page_num-1]) for info in pages_info}
    from .ocr_engine import DualChannelOCREngine
    for pno,spans in list(all_spans.items()):
        if len(spans)<12:all_spans[pno]=DualChannelOCREngine.extract_dual_channel_spans(doc[pno-1],spans)
    table_pages={info.page_num for info in pages_info if info.extraction_diagnostics.get("cad_parts_table")}
    from .cad_import import native_document
    native=native_document(doc)
    duplicate_titles=native['source'].get('duplicate_titles',{}) if native else {}
    for info in pages_info:
        detected=info.extraction_diagnostics.get("cad_parts_table")
        if not detected:continue
        page=doc[info.page_num-1]
        physical=page.rect*page.derotation_matrix
        w,h=physical.width,physical.height
        columns=detected["columns"]
        spans=table_spans(page,all_spans[info.page_num])
        anchors=row_anchors(spans,detected,h)
        if not anchors:
            reviews.append(ReviewItem(f"cad_table_empty_{info.page_num}","WARNING","CAD_TABLE_REVIEW","","清单表头已识别但未读得明细行",
                                      "未建立稳定行坐标，未用猜测值填充 BOM。","核对表格分页与合并单元格。",info.page_num))
        for idx,anchor in enumerate(anchors):
            top=(anchors[idx-1]["cy"]+anchor["cy"])/2 if idx else detected["header_bottom"]+1
            bottom=(anchors[idx+1]["cy"]+anchor["cy"])/2 if idx+1<len(anchors) else anchor["cy"]+(anchor["cy"]-anchors[idx-1]["cy"])/2 if idx else anchor["bbox"][3]+3
            cells={key:[] for key in columns}
            for s in spans:
                if top<=s["cy"]<bottom:
                    for key,column in columns.items():
                        if column["left"]<=s["cx"]<column["right"]:
                            cells[key].append(s);break
            values={key:" ".join(s["text"].strip() for s in sorted(group,key=lambda s:(round(s["cy"],1),s["cx"]))) for key,group in cells.items()}
            name=values.get("name","");specification=values.get("model","")
            if not name and not specification:continue
            tag=values.get("tag","")
            model_tokens=table_model_tokens(specification)
            model="+".join(model_tokens) if model_tokens else "未提取到型号(待核实)"
            quantity_text=values.get("quantity","")
            qty=re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([个只件套片米mM]*)\s*",quantity_text)
            quantity=float(qty[1]) if qty else None
            quantity_candidates=[]
            duplicate_sheet=info.title in duplicate_titles
            if duplicate_sheet and quantity is not None and not reference_only:
                quantity_candidates=[quantity]
                quantity=None
            unit=values.get("unit","") or (qty[2] if qty and qty[2] else "")
            notes=["按原表列与行坐标提取，数量仅引用数量列；未经原理图核对。"]
            if not unit:notes.append("清单未标明计量单位，需核实。")
            if not model_tokens:notes.append("规格列只有参数或未识别到完整型号，保留待选型。")
            if quantity is None and not duplicate_sheet:notes.append("未读得明确数量，保持未知。")
            if duplicate_sheet:
                notes.append(f"同名图框存在多个版本，清单数量列原文为 {quantity_text}；确定有效图框前不累计采购数量。")
            row_spans=[s for group in cells.values() for s in group]
            x0=min(s["bbox"][0] for s in row_spans);x1=max(s["bbox"][2] for s in row_spans)
            rect=[x0,max(0,top),x1,min(h,bottom)];norm=[rect[0]/w,rect[1]/h,rect[2]/w,rect[3]/h]
            visual=pymupdf.Rect(rect)*page.rotation_matrix
            rect=list(visual);norm=[visual.x0/page.rect.width,visual.y0/page.rect.height,visual.x1/page.rect.width,visual.y1/page.rect.height]
            evidence=EvidenceLocation(info.page_num,rect,norm," | ".join(values.values()),role="table_row")
            evidences=[evidence]
            # Literal one-cell tags can be linked to native schematic labels;
            # grouped ranges stay grouped and never create a quantity by expansion.
            if not reference_only and tag and re.fullmatch(r"[-+=A-Za-z0-9_.]+",tag):
                for pno,others in all_spans.items():
                    if pno in table_pages:continue
                    for s in others:
                        if s["text"].strip().upper()==tag.upper():
                            evidences.append(EvidenceLocation(pno,s["bbox"],s["norm_bbox"],s["text"],role="schematic_symbol"))
            role="assembly" if "+" in model else "terminal" if "端子" in name else "cable" if any(t in name for t in ("导线","电缆")) else "device"
            item=BOMItem(id=f"cad_table_{info.page_num}_{anchor['text']}_{idx}",seq_no=len(items)+1,tag=tag,tags=[tag] if tag else [],
                         location_code=info.location_code,
                         name=name or "清单物料",model=model,raw_model=specification,specification=specification+('；'+values['specification'] if values.get('specification') else ''),
                         brand=values.get("brand","") or "待定",unit=unit,quantity=quantity,remark=values.get("remark",""),
                         primary_page=info.page_num,primary_bbox=norm,evidences=evidences,pdf_page_label=f"第{info.page_num}页 · 清单第{anchor['text']}行",
                         source_type=SourceType.EXPLICIT.value if model_tokens else SourceType.UNRESOLVED.value,
                         needs_review=True,review_note=" ".join(notes),entity_role=role,procurement_scope="unknown",
                         quantity_status="needs_confirmation" if quantity is not None else "unknown",quantity_basis="explicit_total" if quantity is not None else "unknown",
                         quantity_candidates=quantity_candidates,
                         observations=[{'source':'drawing_parts_list','cells':values,'row_index':idx+1}])
            items.append(item)
    return items,reviews
