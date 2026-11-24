from __future__ import annotations

import asyncio
import copy
import io
import json
from pathlib import Path
import subprocess

import ezdxf
import pymupdf
import pytest
from fastapi import UploadFile

import backend.app as backend
from bom_extractor.cad_import import (
    CadImportError, NATIVE_ATTACHMENT, NATIVE_MARKER, _dwg_to_dxf,
    attach_cad_evidence, native_document, prepare_cad, select_cad_sheets,
)
from bom_extractor.cad_part_list import table_model_tokens
from bom_extractor.classifier import extract_visual_spans
from bom_extractor.models import ExtractionResult
from bom_extractor.ocr_engine import DualChannelOCREngine
from bom_extractor.pipeline import run_extraction_pipeline


@pytest.fixture(autouse=True)
def no_cad_ocr(monkeypatch):
    monkeypatch.setattr(DualChannelOCREngine,"get_ocr",lambda:None)


def drawing(path: Path, duplicate=False):
    doc=ezdxf.new("R2018")
    frame=doc.blocks.new("BORDER_TEST")
    frame.add_lwpolyline([(0,0),(420,0),(420,297),(0,297)],close=True)
    frame.add_attdef("TITLE",(285,12),height=4)
    for index,offset in enumerate((0,470)):
        insert=doc.modelspace().add_blockref("BORDER_TEST",(offset,0))
        insert.add_auto_attribs({"TITLE":"测试清单" if duplicate else f"测试清单{index+1}"})
        for text,x,y in [("元件清单",180,278),("序号",20,260),("代号",80,260),("名称",145,260),("规格",245,260),("数量",350,260),("单位",390,260),
                         ("1",20,240),(f"R{index+1}",80,240),("继电器",145,240),("R42A+P19C",245,240),("2",350,240),("个",390,240),
                         ("2",20,220),(f"FU{index+1}",80,220),("熔断器",145,220),("2A",245,220),("个",390,220)]:
            doc.modelspace().add_text(text,dxfattribs={"height":4,"insert":(x+offset,y)})
    doc.saveas(path)
    return path


def test_dxf_native_text_attributes_and_geometry_are_readable(tmp_path):
    original=drawing(tmp_path/"source.dxf")
    before=original.read_bytes()
    pdf=prepare_cad(original,tmp_path)
    with pymupdf.open(pdf) as doc:
        assert len(doc)==2
        payload=native_document(doc)
        assert payload['source']['source_format']=='dxf'
        spans=extract_visual_spans(doc[0])
        assert any(s['text']=='测试清单1' and s['attribute_tag']=='TITLE' for s in spans)
        assert any(s['text']=='R42A+P19C' and s['handle'] for s in spans)
        assert not doc[0].get_text().strip(), 'CAD facts must not depend on OCR of the preview'
        assert all(0<=v<=1 for s in spans for v in s['norm_bbox'])
        assert all(s['layout']=='Model' for s in spans)
    assert original.read_bytes()==before
    assert (tmp_path/'cad/import_report.json').is_file()


def test_cad_parts_rows_keep_combinations_unknown_quantities_and_evidence(tmp_path):
    pdf=prepare_cad(drawing(tmp_path/'source.dxf'),tmp_path)
    result=run_extraction_pipeline(pdf,output_dir=tmp_path/'extracted',project_id='cad_test')
    assert result.extraction_mode=='SCHEMATIC_MODE'
    assert not result.itemized_bom and not result.independent_bom
    assert len(result.declared_parts)==4
    assert {r['status'] for r in result.reconciliation}=={'TABLE_ONLY'}
    relay=next(i for i in result.declared_parts if i.tag=='R1')
    assert relay.model=='R42A+P19C'
    assert relay.raw_model=='R42A+P19C'
    assert relay.quantity==2
    assert relay.evidences[0].cad_refs
    assert next(i for i in result.declared_parts if i.tag=='FU1').quantity is None
    assert next(i for i in result.declared_parts if i.tag=='FU1').model.startswith('未提取')
    restored=ExtractionResult.from_dict(result.to_dict())
    assert restored.source_format=='dxf'
    assert restored.declared_parts[0].evidences[0].cad_refs==result.declared_parts[0].evidences[0].cad_refs


def test_table_tokens_keep_all_parts_and_reject_electrical_values():
    assert table_model_tokens('3P 50A（BW50EAG）+(BW9BTAA-S3W)')==['BW50EAG','BW9BTAA-S3W']
    assert table_model_tokens('Q38B+Q61P')==['Q38B','Q61P']
    assert table_model_tokens('AC220V/DC24V 240W')==[]
    assert table_model_tokens('3P 630A')==[]


def test_duplicate_sheets_do_not_silently_add_purchase_counts(tmp_path):
    pdf=prepare_cad(drawing(tmp_path/'source.dxf',duplicate=True),tmp_path)
    result=run_extraction_pipeline(pdf,output_dir=tmp_path/'all',project_id='all')
    assert not result.itemized_bom
    relays=[i for i in result.declared_parts if i.name=='继电器']
    assert len(relays)==2
    assert all(i.quantity==2 for i in relays), 'Reference keeps printed quantities; it never becomes procurement quantity'
    before=pdf.read_bytes()
    selected=tmp_path/'selected.pdf'
    subset=select_cad_sheets(pdf,selected,[2])
    assert subset['source']['selected_source_pages']==[2]
    assert not subset['source']['duplicate_titles']
    assert not subset['source']['requires_sheet_review']
    chosen=run_extraction_pipeline(selected,output_dir=tmp_path/'chosen',project_id='chosen')
    assert not chosen.itemized_bom and len(chosen.declared_parts)==2
    assert next(i for i in chosen.declared_parts if i.name=='继电器').quantity==2
    assert all(e.page==1 for i in chosen.declared_parts for e in i.evidences)
    assert pdf.read_bytes()==before


def test_upload_restart_and_selection_keep_the_original_project(tmp_path):
    source=drawing(tmp_path/'drawing.dxf',duplicate=True)
    response=asyncio.run(backend.upload_pdf(UploadFile(io.BytesIO(source.read_bytes()),filename='工程图.DXF')))
    pid=response['project_id']
    assert response['source_filename']=='工程图.DXF'
    assert response['source_format']=='dxf'
    state=backend.OUTPUT_DIR/pid/'project_state.json'
    before=state.read_bytes()
    selected=asyncio.run(backend.create_cad_selection(pid,backend.CadSheetSelection(pages=[1])))
    assert selected['project_id']!=pid and selected['page_count']==1
    assert state.read_bytes()==before
    assert (backend.OUTPUT_DIR/selected['project_id']/'source.dxf').read_bytes()==source.read_bytes()
    restored=backend.ProjectStore();restored.initialize_defaults()
    assert restored.results[pid].source_format=='dxf'
    assert restored.results[pid].page_count==2
    assert restored.pdf_paths[pid].name=='source.pdf'
    assert backend.download_source(pid).filename=='工程图.DXF'
    assert Path(backend.cad_import_report(pid).path).is_file()


def test_failed_cad_upload_is_not_retried_as_pdf_on_restart(tmp_path,monkeypatch):
    source=drawing(tmp_path/'drawing.dxf')
    monkeypatch.setattr(backend,'run_extraction_pipeline',lambda *a,**k:(_ for _ in ()).throw(ValueError('stop')))
    with pytest.raises(backend.HTTPException):
        asyncio.run(backend.upload_pdf(UploadFile(io.BytesIO(source.read_bytes()),filename='broken.dxf')))
    assert not backend.store.results
    folder=next(backend.OUTPUT_DIR.iterdir())
    assert (folder/'source.pdf').is_file() and (folder/'import_failure.json').is_file()
    fail=copy.deepcopy((folder/'import_failure.json').read_bytes())
    restored=backend.ProjectStore();restored.initialize_defaults()
    assert not restored.results and folder.name in restored.load_errors
    assert (folder/'import_failure.json').read_bytes()==fail


def test_missing_dwg_provider_and_converter_timeout_are_explicit(tmp_path,monkeypatch):
    source=tmp_path/'input.dwg';source.write_bytes(b'AC1032'+b'\0'*50)
    monkeypatch.delenv('CAD_DWG_COMMAND_JSON',raising=False)
    monkeypatch.setenv('CAD_CONVERTER_DLL',str(tmp_path/'missing.dll'))
    with pytest.raises(CadImportError) as missing:_dwg_to_dxf(source,tmp_path)
    assert missing.value.status_code==503
    monkeypatch.setenv('CAD_DWG_COMMAND_JSON',json.dumps(['configured-tool','{input}','{output}']))
    def timeout(*a,**k):raise subprocess.TimeoutExpired(a[0],5)
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(CadImportError) as failed:_dwg_to_dxf(source,tmp_path)
    assert failed.value.status_code==422 and '超时' in str(failed.value)


def test_invalid_native_evidence_is_rejected_instead_of_treated_as_empty_pdf():
    with pymupdf.open() as document:
        document.new_page(width=100,height=100)
        document.set_metadata({'keywords':NATIVE_MARKER})
        document.embfile_add(NATIVE_ATTACHMENT,json.dumps({'schema_version':1,'source':{},'pages':[{'page':1,'spans':[{'text':'BAD','bbox':[0,0,900,900]}]}]}).encode())
        with pytest.raises(CadImportError):native_document(document)


def test_paper_layout_maps_modelspace_native_text(tmp_path):
    doc=ezdxf.new('R2018')
    doc.modelspace().add_text('ABC-1234',dxfattribs={'insert':(10,10),'height':1})
    space=doc.layout();space.add_text('布局电气图',dxfattribs={'insert':(10,190),'height':4})
    space.add_viewport(center=(100,100),size=(180,160),view_center_point=(10,10),view_height=20)
    doc.saveas(tmp_path/'layout.dxf')
    pdf=prepare_cad(tmp_path/'layout.dxf',tmp_path)
    with pymupdf.open(pdf) as rendered:
        assert len(rendered)==1
        spans=extract_visual_spans(rendered[0])
        assert any(s['text']=='ABC-1234' and s['block_path'][0].startswith('VIEWPORT:') for s in spans)
