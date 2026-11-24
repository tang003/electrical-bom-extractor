"""Literal table extraction is an explicit workflow, separate from schematic facts."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

from .models import ExtractionResult, ReviewItem

TABLE_MODE='TABLE_EXTRACT'
CHECK_MODE='SCHEMATIC_CHECK'
TABLE_POLICY='parts-table-only-v1'


def table_policy(result):
    return result.summary.get('table_extraction',{}).get('policy')==TABLE_POLICY


def _overlapping_cad_pages(cad,infos):
    """Only collapse the same native entities at the same physical CAD location."""
    if not cad:return infos,[]
    kept=[];skipped=[];seen=[]
    for info in infos:
        page=cad['pages'][info.page_num-1]
        signature=sorted((s.get('handle',''),str(s.get('block_path',[])),s['text']) for s in page['spans'])
        box=page['cad_bbox'];area=(box[2]-box[0])*(box[3]-box[1])
        duplicate=None
        for prior,other,key in seen:
            a=other['cad_bbox'];larger=(a[2]-a[0])*(a[3]-a[1])
            same_bounds=all(abs(x-y)<.001 for x,y in zip(a,box))
            nested=all((a[0]-.001<=box[0],a[1]-.001<=box[1],a[2]+.001>=box[2],a[3]+.001>=box[3])) and .87<area/larger<=1.001
            if signature==key and page['layout']==other['layout'] and (same_bounds or nested):
                duplicate=prior;break
        if duplicate is not None:skipped.append({'page':info.page_num,'same_physical_page':duplicate})
        else:kept.append(info);seen.append((info.page_num,page,signature))
    return kept,skipped


def read_declared_tables(doc,infos,selected_pages=None):
    from .cad_import import native_document
    from .cad_part_list import extract_cad_part_lists
    from .part_list_extractor import extract_from_part_list_pdf
    detected=[p for p in infos if p.page_type=='PART_LIST']
    selected=set(selected_pages) if selected_pages is not None else None
    candidates=[p for p in detected if selected is None or p.page_num in selected]
    candidates,duplicates=_overlapping_cad_pages(native_document(doc),candidates)
    def generic_table(p):
        detected=p.extraction_diagnostics.get('cad_parts_table')
        if not detected:return False
        cols=detected['columns']
        # The EPLAN reader understands multi-line placement and device fields.
        return not (cols.get('tag',{}).get('label')=='设备标识符' and cols.get('model',{}).get('label') in {'订货编号','订货号'})
    generic=[p for p in candidates if generic_table(p)]
    other=[p for p in candidates if p not in generic]
    rows=[];reviews=[]
    if generic:
        found,checks=extract_cad_part_lists(doc,generic,reference_only=True)
        rows.extend(found);reviews.extend(checks)
    if other:
        found,checks=extract_from_part_list_pdf(doc,other,table_only=True)
        rows.extend(found);reviews.extend(checks)
    rows.sort(key=lambda i:(i.primary_page,i.seq_no))
    for index,item in enumerate(rows,1):
        item.id=f'declared_{index}_{item.id}';item.seq_no=index;item.model_origin='drawing_parts_list'
    unread=[p.page_num for p in candidates if not any(i.primary_page==p.page_num for i in rows)]
    status='partial' if rows and unread else 'loaded' if rows else 'unreadable' if candidates else 'absent'
    return rows,reviews,{'status':status,'detected_pages':[p.page_num for p in detected],
        'selected_pages':[p.page_num for p in candidates],'unreadable_pages':unread,
        'skipped_overlapping_pages':duplicates,'row_count':len(rows)}


def table_result(doc,infos,source,project_id,project_name,drawing_no,folder,selected_pages=None):
    from .cad_import import attach_cad_evidence,native_document
    from .procurement import procurement_summary
    rows,reviews,report=read_declared_tables(doc,infos,selected_pages)
    # Immutable source rows and editable working rows never share objects.
    working=copy.deepcopy(rows)
    for item in working:
        cells=next((o['cells'] for o in item.observations if 'cells' in o),{})
        if cells and not item.model.startswith('未提取'):
            item.model=cells['model']
        item.review_note='仅提取部件表，未经原理图核对。'+item.review_note
    if report['status']!='loaded':
        message='未检测到所选范围内的部件表，未改用原理图识别或生成猜测清单。' if report['status']=='absent' else '部分表格未读得明细，请核对所选图页和表格版式。'
        reviews.append(ReviewItem('table_read_status','WARNING','TABLE_EXTRACTION','','表格提取需要核对',message,'核对图页或另建原理图核对项目。',1))
    cad=native_document(doc)
    if cad:
        for index,warning in enumerate(cad['source']['warnings']):
            reviews.append(ReviewItem(f'table_cad_{index}','WARNING','CAD_IMPORT_REVIEW','','CAD导入提醒',warning,'核对原图与导入报告。',1,scope='project'))
    report.update(policy=TABLE_POLICY,schematic_verified=False,requested_pages=selected_pages)
    result=ExtractionResult(project_id,source.name,project_name,drawing_no,'PART_LIST_MODE',len(infos),
        pages=infos,itemized_bom=working,declared_parts=rows,review_items=reviews,
        summary={'workflow_mode':TABLE_MODE,'extraction_mode_label':'仅提取部件表（未经原理图核对）',
                 'table_extraction':report,'declared_reference_status':report['status'],'total_pages':len(infos),
                 'output_dir':str(folder),'itemized_count':len(working),'aggregated_count':0,
                 'filtered_count':0,'review_count':len(reviews),'source_breakdown':dict(Counter(i.source_type.lower() for i in working))})
    attach_cad_evidence(result,cad)
    result.summary['procurement']=procurement_summary(result)
    (folder/'declared_parts.json').write_text(json.dumps([i.to_dict() for i in rows],ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'table_extraction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
