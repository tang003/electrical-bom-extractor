"""Associate native model text with tag anchors and drawn module enclosures."""
from __future__ import annotations
import math
import re
import pymupdf
from .notation import POWER_CURRENT_LABEL, extract_model_strings, local_tag, model_compatible, tagged_model_series
from .cad_import import cad_stacked_model_owner


def symbol_enclosures(page):
    horizontal, vertical = [], []
    matrix = page.rotation_matrix
    rectangles = []
    # Use sheet-relative minima so scaled plots and landscape/portrait exports
    # describe the same device bodies. Ignore small pin boxes and page frames.
    min_width = min(70.0, page.rect.width * .075)
    min_height = min(35.0, page.rect.height * .05)

    def device_size(rect):
        return (min_width < rect.width < .98 * page.rect.width
                and min_height < rect.height < .80 * page.rect.height
                and rect.y1 <= .92 * page.rect.height)

    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] in ("re", "qu"):
                shape = item[1] * matrix
                rect = shape if item[0] == "re" else shape.rect
                if not device_size(rect):
                    continue
                if item[0] == "qu" and not all(
                    any(abs(vertex - corner) < 1 for vertex in shape)
                    for corner in (rect.tl, rect.tr, rect.bl, rect.br)
                ):
                    continue  # Winding can vary, but each corner must be present.
                rectangles.append(rect)
            elif item[0] == "l":
                a,b = item[1]*matrix,item[2]*matrix
                if abs(a.y-b.y)<1.0 and abs(a.x-b.x)>min_width:
                    horizontal.append((min(a.x,b.x),(a.y+b.y)/2,max(a.x,b.x)))
                elif abs(a.x-b.x)<1.0 and abs(a.y-b.y)>min_height:
                    vertical.append(((a.x+b.x)/2,min(a.y,b.y),max(a.y,b.y)))
    for i,(left,top,right) in enumerate(horizontal):
        for x0,bottom,x1 in horizontal[i+1:]:
            if abs(left-x0)>2 or abs(right-x1)>2:
                continue
            y0,y1=sorted((top,bottom))
            rect = pymupdf.Rect(left, y0, right, y1)
            if not device_size(rect):
                continue
            if all(any(abs(x-edge)<2 and vy0<=y0+2 and vy1>=y1-2 for x,vy0,vy1 in vertical) for edge in (left,right)):
                rectangles.append(rect)
    return rectangles


def distance_to_text(anchor, text):
    box=text["bbox"]
    dx=max(box[0]-anchor["cx"],0,anchor["cx"]-box[2])
    dy=max(box[1]-anchor["cy"],0,anchor["cy"]-box[3])
    return math.hypot(dx,dy)


def _label_side(label, body):
    """Return the part of a drawn body that this text labels, if any."""
    text = pymupdf.Rect(label["bbox"])
    x, y = label["cx"], label["cy"]
    gap = 2 * min(text.width, text.height)
    if body.x0 <= x <= body.x1:
        if body.y0 <= y <= body.y1:
            return "inside"
        if text.y0 <= body.y1 + gap and text.y1 >= body.y0 - gap:
            return "top" if y < body.y0 else "bottom"
    if body.y0 <= y <= body.y1 and text.x0 <= body.x1 + gap and text.x1 >= body.x0 - gap:
        return "left" if x < body.x0 else "right"
    return None


def _caption_owners(model, side, labelled):
    tags = {anchor["tag"] for anchor, _ in labelled}
    if len(tags) <= 1:
        return tags
    if side == "inside" or any(edge == "inside" for _, edge in labelled):
        return set()  # Multiple labels within an assembly do not establish exclusive ownership.
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    axis = 0 if side in ("top", "bottom") else 1
    # A device can have its tag above and its model below (or left/right).
    # A neighbouring label beside this body must not steal that aligned caption.
    aligned = {anchor["tag"] for anchor, edge in labelled if edge == opposite[side]
               and min(anchor["bbox"][axis+2], model["bbox"][axis+2]) >= max(anchor["bbox"][axis], model["bbox"][axis])}
    if aligned and not any(edge == side for _, edge in labelled):
        return aligned
    return tags  # Ambiguous edge captions remain unbound.


_DEVICE_CAPTION = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{0,18}(?:控制器|探测器|报警器|蜂鸣器|接触器|继电器|"
    r"断路器|保护器|变送器|传感器|互感器|变频器|电源|电表|空调|灯|按钮盒|按钮|开关|插座|模块|轴卡)"
)


def device_captions(spans, anchors, bodies):
    """Bind a printed device description to a single named symbol or a stacked tag label."""
    body_anchors = [[] for _ in bodies]
    for anchor in anchors:
        tag_box = pymupdf.Rect(anchor['bbox'])
        adjacent = []
        for index, body in enumerate(bodies):
            side = _label_side(anchor, body)
            if side:
                dx = max(body.x0 - tag_box.x1, tag_box.x0 - body.x1, 0)
                dy = max(body.y0 - tag_box.y1, tag_box.y0 - body.y1, 0)
                adjacent.append((math.hypot(dx, dy), index))
        if adjacent:
            best = min(distance for distance, _ in adjacent)
            for distance, index in adjacent:
                if distance <= best + 2:
                    body_anchors[index].append(anchor['tag'])

    owners = {}
    for span in spans:
        if not (_DEVICE_CAPTION.fullmatch(span['text']) or re.fullmatch(r'(?:[A-Z]+\+)?RCBO', span['text'], re.I)):
            continue
        x0,y0,x1,y1 = span['bbox']
        font = y1-y0
        # A common EPLAN label stack right-aligns the function under the device tag.
        direct = {a['tag'] for a in anchors
                  if 0 <= y0-a['bbox'][3] <= 2*font
                  and min(abs(x0-a['bbox'][0]),abs(x1-a['bbox'][2])) <= font}
        declared = direct or {tag for body,tags in zip(bodies,body_anchors)
                              if _label_side(span,body) and len(set(tags))==1 for tag in tags}
        if len(declared)==1:
            owners.setdefault(next(iter(declared)),[]).append(span)
    return owners


def _rated_series_owner(model, anchors, ratings):
    """Recognise a three-line caption: indexed family, rating, complete order code."""
    mx0,my0,mx1,my1=model['bbox']
    height=my1-my0
    matches=set()
    for a in anchors:
        if not tagged_model_series(a['tag'],model['model_text']):
            continue
        ax0,ay0,ax1,ay1=a['bbox']
        if not 0 < my0-ay1 < 2.5*height:
            continue
        for r in ratings:
            rx0,ry0,rx1,ry1=r['bbox']
            if not (ay1-.25*height <= ry0 and ry1 <= my0+.25*height):
                continue
            if min(max(mx0,rx0,ax0)-min(mx0,rx0,ax0),max(mx1,rx1,ax1)-min(mx1,rx1,ax1)) <= .5*height:
                matches.add(a['tag'])
    return next(iter(matches)) if len(matches)==1 else None


def bind_models(page, spans, anchors):
    """Each model occurrence has one owner; repeated symbols of that device share it."""
    enclosures=symbol_enclosures(page)
    body_labels = [(body, [(a, side) for a in anchors if (side := _label_side(a, body))])
                   for body in enclosures]
    embedded = {}
    for anchor in anchors:
        if anchor.get('native_tag') and anchor.get('source_text'):
            embedded.setdefault(anchor['source_text'], []).append(anchor)
    bindings={}
    unbound=[]
    models=[]
    printed_tags = {local_tag(a['tag']).lstrip('-') for a in anchors}
    # Matching phase labels for a branch identify its conductors, e.g. U208/V208/W208.
    labels_on_sheet={s['text'].strip().upper() for s in spans}
    phase_wires=set()
    for label in labels_on_sheet:
        phase=re.fullmatch(r'[UVW](\d+[A-Z]?)',label)
        if phase:
            phases={prefix+phase[1] for prefix in 'UVW'}
            if len(phases & labels_on_sheet)>=2:
                phase_wires.update(phases)
    for span in spans:
        # PDF engines can return a single text span for a tag and its part
        # number even when they were written as adjacent text objects.
        text = span["text"]
        if text.strip().upper() in phase_wires:
            continue
        if span.get("source") == "cad_native" and span.get("attribute_tag", "").strip().upper() in {
            "图号", "图纸编号", "图名", "图纸名称", "图幅", "比例", "DRAWING NO", "DRAWING_NUMBER", "SHEET_NO"
        }:
            continue
        labels = [a for a in embedded.get(text, ()) if pymupdf.Rect(a['bbox']).intersects(pymupdf.Rect(span['bbox']))]
        inline = labels[0] if len(labels) == 1 and text.startswith(labels[0]['native_tag'] + ' ') else None
        if inline:
            text = text[len(inline['native_tag']):].lstrip()
        elif text.strip().upper() in printed_tags:
            continue  # A bare repetition of this sheet's device tag is not an order code.
        for value in extract_model_strings(text):
            model={**span,"model_text":value}
            if inline:
                model['inline_owner'] = inline['tag']
            if re.search(r"\b[1-4]P$",value,re.I):
                ratings=[other for other in spans if re.fullmatch(r"\d+A",other["text"],re.I)
                         and math.hypot(other["cx"]-span["cx"],other["cy"]-span["cy"])<32]
                if ratings:
                    rating=min(ratings,key=lambda other:math.hypot(other["cx"]-span["cx"],other["cy"]-span["cy"]))
                    model["model_text"]=value+" "+rating["text"]
                    model["text"]=span["text"]+" "+rating["text"]
                    for key in ("bbox","norm_bbox"):
                        a,b=span[key],rating[key]
                        model[key]=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
            models.append(model)
    limit=max(65.0,.22*math.hypot(page.rect.width,page.rect.height))
    ratings=[s for s in spans if POWER_CURRENT_LABEL.fullmatch(s['text'].strip())]
    for model in models:
        if 'inline_owner' in model:
            owner = model['inline_owner']
            if model_compatible(owner, model['model_text']):
                bindings.setdefault(owner, []).append({**model, "association": "inline", "needs_review": False})
            else:
                unbound.append(model)
            continue
        owner=_rated_series_owner(model,anchors,ratings) if ratings else None
        if owner and model_compatible(owner,model['model_text']):
            bindings.setdefault(owner, []).append({**model,'association':'label_stack','needs_review':False})
            continue
        ranked=[]
        model_enclosures = [rect for rect in enclosures if pymupdf.Point(model["cx"],model["cy"]) in rect]
        # An identifier and a part number may be on opposite sides of the same
        # symbol. A unique labelled outline is better evidence than the distance
        # across that symbol to an unrelated coil or switch.
        owners = set()
        for body, labelled in body_labels:
            if model_enclosures and body not in model_enclosures:
                continue
            side = _label_side(model, body)
            if side:
                owners.update(_caption_owners(model, side, labelled))
        if owners:
            if len(owners) == 1:
                owner = next(iter(owners))
                if model_compatible(owner, model["model_text"]):
                    bindings.setdefault(owner, []).append({**model, "association": "enclosure", "needs_review": False})
                    continue
            # Between two separately labelled bodies or at an incompatible
            # body: retain the text for review instead of giving it to a neighbour.
            unbound.append(model)
            continue
        stacked = cad_stacked_model_owner(model, anchors) if not model_enclosures else None
        if stacked and model_compatible(stacked, model["model_text"]):
            bindings.setdefault(stacked, []).append({**model, "association": "cad_label", "needs_review": False})
            continue
        for anchor in anchors:
            if not model_compatible(anchor["tag"],model["model_text"]):
                continue
            # A nearby bare serial or wire number alone cannot identify the model
            # of an implicitly prefixed PDF tag. Strong inline/boxed matches above
            # already retain positively owned catalogue references.
            if (anchor.get('native_tag') and anchor.get('source')!='cad_native'
                    and not anchor['native_tag'].startswith(('-', '+', '='))
                    and re.fullmatch(r'[A-Z]{1,8}\d+[A-Z]{0,3}',model['model_text'],re.I)
                    and not model['model_text'].upper().startswith(('LRN','LRD','LC1'))):
                continue
            distance=distance_to_text(anchor,model)
            common = any(pymupdf.Rect(anchor["bbox"]).intersects(rect + (-30, -20, 30, 20))
                         for rect in model_enclosures)
            # Text inside a boxed device cannot select an unrelated nearby switch
            # outside that box merely because the true device has no printed tag.
            if model_enclosures and not common:
                continue
            # Bare proximity can join the model of an unlabelled network device
            # to a different tag across the sheet. Restrict it to the local text
            # group; enclosed symbols can still associate over a large distance.
            label_height=max(anchor['bbox'][3]-anchor['bbox'][1],model['bbox'][3]-model['bbox'][1])
            if distance>min(limit, max(8.0, 8*label_height)) and not common:
                continue
            score=distance*(.25 if common else 1.0)
            ranked.append((score,anchor, bool(common)))
        ranked.sort(key=lambda record:record[0])
        if not ranked:
            unbound.append(model)
            continue
        score,anchor,enclosed=ranked[0]
        competitor=next((r for r in ranked[1:] if r[1]["tag"]!=anchor["tag"]),None)
        if competitor and competitor[0]-score<5:
            unbound.append(model)
            continue
        bindings.setdefault(anchor["tag"],[]).append({**model,"association":"enclosure_candidate" if enclosed else "spatial",
            "needs_review":not enclosed and distance_to_text(anchor,model)>75})
    return bindings, models, unbound, enclosures
