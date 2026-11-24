"""Read codes printed inside numbered slots of a drawn CAD controller rack."""
import re

from .models import BOMItem, EvidenceLocation, SourceType


def _walls(page):
    walls = []
    for drawing in page.get_drawings():
        for ink in drawing["items"]:
            if ink[0] == "re":
                rect = ink[1] * page.rotation_matrix
                walls.extend([(x,rect.y0,rect.y1) for x in (rect.x0,rect.x1)])
            if ink[0] == "l":
                a,b = ink[1]*page.rotation_matrix, ink[2]*page.rotation_matrix
                if abs(a.x-b.x) < .6:
                    walls.append(((a.x+b.x)/2,min(a.y,b.y),max(a.y,b.y)))
    return walls


def _slot_models(page, spans):
    native = [s for s in spans if s.get("source") == "cad_native"]
    cpu = [s for s in native if re.search(r"(?:\bCPU\b|\bPLC\b|可编程控制器)",s["text"],re.I)]
    if not cpu:return []
    digits = [s for s in native if re.fullmatch(r"\d{1,2}",s["text"])]
    walls = _walls(page)
    modules = []
    for first in sorted((s for s in digits if s["text"] in ("0","1")), key=lambda s:int(s["text"])):
        h = first["bbox"][3]-first["bbox"][1]
        # One adjacent processor or controller label must identify this row.
        if not any(0 < first["cx"]-s["cx"] < 20*h and abs(first["cy"]-s["cy"]) < 8*h for s in cpu):
            continue
        row = [first]
        for nxt in sorted(digits,key=lambda s:s["cx"]):
            if (nxt["cx"] <= row[-1]["cx"] or abs(nxt["cy"]-first["cy"]) > .45*h
                    or nxt.get("layer") != first.get("layer")):continue
            distance=nxt["cx"]-row[-1]["cx"]
            if int(nxt["text"])!=int(row[-1]["text"])+1 or not 2*h < distance < 12*h:
                continue
            if len(row)>1 and abs(distance-(row[-1]["cx"]-row[-2]["cx"]))>h:
                continue
            row.append(nxt)
        if len(row)<3 or any(s["handle"] in {m[0]["handle"] for m in modules} for s in row):
            continue
        pitch=(row[-1]["cx"]-first["cx"])/(len(row)-1)
        for position in row:
            x,y=position["cx"],position["cy"]
            sides=[(wx,top,bottom) for wx,top,bottom in walls if y <= top <= y+2*h
                   and 3*h < bottom-top < 35*h and abs(abs(wx-x)-pitch/2)<.8*h]
            left=next((w for w in sides if w[0]<x),None)
            right=next((w for w in sides if left and w[0]>x and abs(left[1]-w[1])<.3*h and abs(left[2]-w[2])<.3*h),None)
            if not left or not right:continue
            top=left[1]
            upper=top+min(.44*(left[2]-top),5*h)
            inside=sorted([s for s in native if left[0] <= s["cx"] <= right[0] and top < s["cy"] <= upper],
                          key=lambda s:(s["cy"],s["cx"]))
            # Keep literal ordering. Do not infer catalog codes from PLC pin
            # addresses, parameter fields or an empty slot.
            if not 1 <= len(inside) <= 2:continue
            if any(not re.fullmatch(r"[A-Z0-9]+(?:[-./][A-Z0-9]+)*",s["text"],re.I)
                   or not re.search("[A-Z]",s["text"],re.I) or not re.search(r"\d",s["text"])
                   or re.fullmatch(r"(?:[XYIMQ]\d{3,}|\d+(?:V|A|W|HZ))",s["text"],re.I) for s in inside):
                continue
            if len(inside)==2:
                a,b=inside
                if abs(a["bbox"][0]-b["bbox"][0])>.5*h or not 0 < b["cy"]-a["cy"] < 1.75*h:continue
            code="".join(s["text"] for s in inside)
            if len(code)>=4:modules.append((position,inside,code))
    return modules


def read_rack(page, spans, info):
    rows=[]
    reserved=set()
    for position, parts, code in _slot_models(page,spans):
        evidence=[EvidenceLocation(info.page_num,s["bbox"],s["norm_bbox"],raw_text=s["text"],role="model_text",
                                   model_binding="cad_rack_slot") for s in parts]
        evidence.append(EvidenceLocation(info.page_num,position["bbox"],position["norm_bbox"],
                                         raw_text=position["text"],role="slot_position"))
        slot=position["text"]
        rows.append(BOMItem(id=f"rack_{info.page_num}_{position['handle']}",seq_no=0,tag="",tags=[],
                            name="PLC 机架模块",model=code,raw_model=code,brand="待定",quantity=1,unit="个",
                            location_code=info.location_code,placement=info.location_code,procurement_scope="cabinet",
                            group_identity=f"{info.function_code}{info.location_code}/SLOT/{slot}",
                            primary_page=info.page_num,primary_bbox=evidence[0].norm_bbox,evidences=evidence,
                            pdf_page_label=f"第{info.page_num}页",source_type=SourceType.EXPLICIT.value,
                            quantity_basis="drawn_slot",quantity_status="by_slot",needs_review=True,
                            review_note=f"机架槽位 {slot} 内有型号实字。请核对柜图与实际供货配置。"))
        reserved.update(s["handle"] for s in parts)
    return rows,reserved
