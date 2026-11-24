"""Read-only row markers for parts tables; never infer rows or procurement quantities."""
from .reconciliation import norm, tag_members


def row_regions(rows, page_num):
    return [(row,e.norm_bbox) for row in rows for e in row.evidences
            if e.page==page_num and e.role=='table_row' and e.norm_bbox and len(e.norm_bbox)==4]


def table_candidates(page, rows, spans):
    width,height=float(page.rect.width),float(page.rect.height)
    result=[];seen=set()
    for row,region in row_regions(rows,page.number+1):
        if row.id in seen:continue
        seen.add(row.id)
        label=row.tag or row.name
        matches=[s for s in spans if norm(s['text'])==norm(label) and
                 region[0]-.002<=s['cx']/width<=region[2]+.002 and region[1]<=s['cy']/height<=region[3]] if label else []
        if matches:
            # Match only this row's own text. Do not grow into nearby tag/model cells.
            box=[min(s['norm_bbox'][0] for s in matches),min(s['norm_bbox'][1] for s in matches),
                 max(s['norm_bbox'][2] for s in matches),max(s['norm_bbox'][3] for s in matches)]
            box=[max(region[0],box[0]-1/width),max(region[1],box[1]-1/height),
                 min(region[2],box[2]+1/width),min(region[3],box[3]+1/height)]
        else:box=list(region)
        if box[2]<=box[0] or box[3]<=box[1]:continue
        members=list(dict.fromkeys(member for value in (row.tags or [row.tag]) for member in tag_members(value) if member))
        member_labels=[m[1:] if not row.tag.startswith(('=','+','-')) and m.startswith('-') else m for m in members]
        result.append({'tag':row.tag,'display_label':row.tag or row.name or f'清单第{row.seq_no}项',
                       'source_item_id':row.id,'table_members':members,'table_member_labels':member_labels,'table_row_index':row.seq_no,
                       'reference_name':row.name,'reference_model':row.model,'reference_quantity':row.quantity,
                       'reference_row_box':list(region),'role':'table_row','detector_source':'PARTS_TABLE_ROW',
                       'bbox':[round(box[0]*width,2),round(box[1]*height,2),round(box[2]*width,2),round(box[3]*height,2)],
                       'norm_bbox':box,'anchor_norm_bbox':box,'cx':(box[0]+box[2])*width/2,'cy':(box[1]+box[3])*height/2,
                       'nearby_texts':[s for s in [row.tag,row.name,row.model] if s],
                       'model_candidates':[row.model] if row.model and not row.model.startswith('未提取') else []})
    return result
