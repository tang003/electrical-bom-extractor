// Exercise the actual Vue setup with in-memory fixtures; never call a running project API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
let app, request;
const storage = new Map();
const context = vm.createContext({
  console,
  URLSearchParams,
  localStorage: {getItem:key=>storage.get(key), setItem:(key,value)=>storage.set(key,value)},
  window: {},
  setTimeout:()=>0,
  fetch:(...args)=>request(...args),
  confirm:()=>true,
  FormData:class { constructor(){this.fields=new Map();} append(k,v){this.fields.set(k,v);} },
});
vm.runInContext(fs.readFileSync(path.join(root,'frontend/assets/vue.global.prod.js'),'utf8'), context);
context.document = {activeElement:null, querySelector:()=>null, getElementById:()=>null};
context.Vue.onMounted = ()=>{};
context.Vue.createApp = options=>({mount:()=>{app=options.setup();}});
vm.runInContext(fs.readFileSync(path.join(root,'frontend/assets/app.js'),'utf8'), context);

function fixture(id='fixture') {
  const rows = [
    {id:'unknown',seq_no:1,name:'电缆',tag:'-W1',model:'',specification:'3*70+2*35',quantity:null,procurement_scope:'unknown',source_type:'UNRESOLVED'},
    {id:'zero',seq_no:2,name:'接触器',tag:'-KM1',model:'LC1D',quantity:0,procurement_scope:'cabinet',source_type:'EXPLICIT'},
    {id:'known',seq_no:3,name:'熔断器',tag:'-FU1',model:'RT18',quantity:3,procurement_scope:'field',source_type:'EXPLICIT'},
  ];
  return {project_id:id,extraction_mode:'SCHEMATIC_MODE',summary:{},pages:[],aggregated_bom:rows,itemized_bom:rows,review_items:[{status:'PENDING'},{status:'IGNORED'}]};
}
const response = data=>({ok:true,json:async()=>data});

(async()=>{
  assert.equal(app.leftPaneWidth.value,60,'Give the drawing most of the default comparison view');
  assert.equal(app.overviewExpanded.value,false,'Keep the drawing workspace tall by default');
  app.toggleOverview();
  assert.equal(app.overviewExpanded.value,true);
  assert.equal(JSON.parse(storage.get('electrical-bom-workspace')).overviewExpanded,true);
  app.toggleOverview();
  const anchor=[.2,.2,.25,.25];
  const centered=app.drawingBoxStyle([.5,.5,.505,.506],.018,.018);
  assert.ok(Math.abs(parseFloat(centered.left)-49.35)<.001 && Math.abs(parseFloat(centered.top)-49.4)<.001,
    'Small terminal hitboxes must expand around the symbol center, not only right and down');
  const somItem=(id,tag,role='device',evidenceRole='main')=>({id,tag,tags:[tag],seq_no:1,name:'元件',model:'',entity_role:role,quantity_basis:'unique_tag',evidences:[{page:1,norm_bbox:anchor,role:evidenceRole}]});
  const somRows=[somItem('local','-K1'),somItem('scope-a','=A-K2'),somItem('scope-b','=B-K2'),somItem('contact','-K3','device','contact'),somItem('uncertain','-K4','device','contact_or_symbol')];
  app.projectData.value={...fixture(),pages:[{page_num:1}],itemized_bom:somRows,aggregated_bom:somRows};
  app.currentProjectId.value='fixture';app.currentPage.value=1;
  const candidate=(id,tag,other={})=>({som_id:id,som_label:'#'+id,tag,norm_bbox:anchor,nearby_texts:[],...other});
  const topo={project_id:'fixture',view_mode:'original',page_num:1,candidates:[candidate(1,'-K1'),candidate(2,'-K2'),candidate(3,'=A-K2'),candidate(4,'-K3'),candidate(5,'-K4'),candidate(6,'-NEW1',{model_candidates:['MODEL-999']})],excluded_candidates:[candidate(7,'-M4',{excluded:true,exclusion_reason:'边框微小标注'})]};
  app.pageTopology.value=topo;
  assert.equal(app.filteredSomCandidates.value.length,6,'Default shows every valid candidate, including those without a model');
  assert.deepEqual(Array.from(app.filteredSomCandidates.value,c=>c.kind),['linked','pending','linked','reference','linked','pending']);
  assert.equal(app.somCounts.value.excluded,1);
  assert.equal(app.somCounts.value.reference,1,'An ambiguous contact_or_symbol role must not be labelled reference');
  const beforeSom=JSON.stringify(app.projectData.value);
  app.somFilter.value='pending';assert.equal(app.filteredSomCandidates.value.length,2);
  app.somShowExcluded.value=true;assert.equal(app.filteredSomCandidates.value.length,3);
  app.selectSomCandidate(app.filteredSomCandidates.value.find(c=>c.excluded));
  assert.equal(app.selectedItemId.value,'','Excluded annotations cannot select or add a BOM item');
  app.somFilter.value='ALL';app.somShowExcluded.value=false;
  app.searchQuery.value='hidden by filter';app.selectSomCandidate(app.filteredSomCandidates.value[0]);
  assert.equal(app.selectedItemId.value,'local');assert.equal(app.searchQuery.value,'');
  assert.equal(JSON.stringify(app.projectData.value),beforeSom,'Filters and candidate clicks never change quantities or saved evidence');
  app.projectData.value.pages[0].function_code='=B';
  assert.equal(app.filteredSomCandidates.value.find(c=>c.tag==='-K2').matched_item_id,'scope-b','Use the complete page scope instead of choosing the first local tag');
  app.clearActiveBox();
  app.showSomLayer.value=true;
  let topologyPending={};request=url=>new Promise(resolve=>{topologyPending[url]=resolve;});
  const pageOne=app.loadPageTopology();app.currentPage.value=2;const pageTwo=app.loadPageTopology();
  topologyPending['/api/projects/fixture/pages/2/topology?view=original'](response({page_num:2,candidates:[]}));await pageTwo;
  topologyPending['/api/projects/fixture/pages/1/topology?view=original'](response(topo));await pageOne;
  assert.equal(app.pageTopology.value.page_num,2,'A late topology response must never place page-one boxes on page two');
  assert.equal(app.somFilter.value,'ALL');assert.equal(app.somShowExcluded.value,false);
  app.showSomLayer.value=false;app.pageTopology.value=null;app.currentPage.value=1;
  app.projectData.value={...fixture(),pages:[{page_num:1},{page_num:2}]};
  const oldImageUrl=app.pageImageUrl.value;
  const imageEvent=(url,connected=true)=>({currentTarget:{isConnected:connected,getAttribute:()=>url}});
  assert.equal(app.pageImageReady.value,false);
  app.onPageImageLoad(imageEvent(oldImageUrl));assert.equal(app.pageImageReady.value,true);
  await app.jumpToLocation(2,anchor,'page two');
  assert.equal(app.pageImageReady.value,false,'New boxes stay hidden until the new page image has loaded');
  app.onPageImageLoad(imageEvent(oldImageUrl));assert.equal(app.pageImageReady.value,false,'An old image completion cannot enable new-page boxes');
  app.onPageImageLoad(imageEvent(app.pageImageUrl.value));assert.equal(app.pageImageReady.value,true);
  app.onPageImageError(imageEvent(oldImageUrl));assert.equal(app.pageImageError.value,'');
  app.onPageImageError(imageEvent(app.pageImageUrl.value));assert.match(app.pageImageError.value,/加载失败/);
  app.retryPageImage();assert.equal(app.pageImageReady.value,false);
  app.currentPage.value=1;
  app.onPageImageLoad(imageEvent(oldImageUrl,false));assert.equal(app.pageImageReady.value,false,'Detached images cannot expose overlays after navigating back');
  const imageTimers = new Map();
  const imageRequests = [];
  const originalTimeout = context.setTimeout;
  context.setTimeout = (callback, delay) => { const id = Symbol(); if (delay === 180) imageTimers.set(id, callback); return id; };
  context.clearTimeout = id => imageTimers.delete(id);
  context.Image = class {
    constructor() { imageRequests.push(this); }
    async decode() {}
  };
  app.zoomPercent.value = 100;
  app.onPageImageLoad(imageEvent(app.pageImageUrl.value));
  const previewUrl = app.pageImageUrl.value;
  app.zoomPercent.value = 200;
  await context.Vue.nextTick();
  for (const callback of imageTimers.values()) callback();
  imageTimers.clear();
  assert.equal(imageRequests.length, 1);
  assert.equal(app.pageImageUrl.value, previewUrl, 'Zoom preserves the visible preview and its DOM identity');
  assert.equal(app.pageImageReady.value, true, 'High-resolution loading must not blank a readable drawing');
  await imageRequests[0].onload();
  const sharpUrl = app.displayedPageImageUrl.value;
  assert.match(sharpUrl, /dpi=220/);
  app.zoomPercent.value = 100;
  await context.Vue.nextTick();
  assert.equal(app.displayedPageImageUrl.value, sharpUrl, 'Zooming out reuses the decoded high-resolution image');
  app.currentPage.value = 2;
  await imageRequests[0].onload();
  assert.equal(app.displayedPageImageUrl.value, app.pageImageUrl.value, 'Late high-resolution images cannot replace a different page');
  assert.equal(app.pageImageReady.value, false);
  context.setTimeout = originalTimeout;
  await app.jumpToLocation(1,null,'no location');assert.equal(app.activeBox.value,null,'Never invent a fallback evidence rectangle');
  const cadWarning={issue_type:'CAD_IMPORT_REVIEW',page:1,norm_bbox:[.1,.1,.3,.3]};
  assert.equal(app.canLocateReview(cadWarning),false,'Legacy default CAD warning boxes are not genuine drawing evidence');
  app.currentPage.value=2;app.selectReviewCard(cadWarning);
  assert.equal(app.currentPage.value,2);assert.equal(app.activeBox.value,null);
  app.currentPage.value=1;app.overviewExpanded.value=false;
  const grouped=somItem('grouped','QF1,2');const ranged=somItem('ranged','FU1~2');
  app.projectData.value={...fixture(),pages:[{page_num:1}],itemized_bom:[grouped,ranged],aggregated_bom:[grouped,ranged]};
  app.pageTopology.value={...topo,candidates:[candidate(1,'QF2'),candidate(2,'FU2')],excluded_candidates:[]};
  assert.deepEqual(Array.from(app.filteredSomCandidates.value,c=>c.matched_item_id),['grouped','ranged'],'Grouped table tags may link to symbols without changing table rows or quantities');
  assert.equal(app.projectData.value.itemized_bom.length,2);
  const padded=somItem('padded','QF01~02');
  app.projectData.value.itemized_bom=[padded];app.projectData.value.aggregated_bom=[padded];
  app.pageTopology.value={...topo,candidates:[candidate(1,'QF02'),candidate(2,'QF2')],excluded_candidates:[]};
  assert.deepEqual(Array.from(app.filteredSomCandidates.value,c=>c.matched_item_id),['padded',''],'Leading zeros in full identifiers must survive grouped-tag lookup');
  const shared=somItem('shared','QF1;2/4至5');shared.tag_members=['-QF1','-QF2','-QF4','-QF5'];
  app.projectData.value.itemized_bom=[shared];app.projectData.value.aggregated_bom=[shared];
  app.pageTopology.value={...topo,candidates:[candidate(1,'QF4')],excluded_candidates:[]};
  assert.equal(app.filteredSomCandidates.value[0].matched_item_id,'shared','Use the shared backend membership interpretation');
  app.projectData.value.summary={table_extraction:{policy:'parts-table-only-v1'}};
  app.pageTopology.value={...topo,candidates:[candidate(1,'QF1;2/4至5',{detector_source:'PARTS_TABLE_ROW',source_item_id:'shared'})],excluded_candidates:[]};
  assert.equal(app.filteredSomCandidates.value[0].kind,'linked');
  app.projectData.value.summary={independent_analysis:{policy:'independent-schematic-v1'}};
  assert.equal(app.filteredSomCandidates.value[0].kind,'reference','Original table markers must never count as independent schematic recognition');
  assert.equal(app.filteredSomCandidates.value[0].matched_item_id,'');
  app.pageTopology.value=null;

  const point=(page,x,y,role='device')=>({page,role,norm_bbox:[x,y,x+.02,y+.02],raw_text:'symbol'});
  const left={...somItem('left','-K1'),seq_no:20,evidences:[point(2,.4,.4),point(1,.1,.104),point(1,.1,.5)]};
  const right={...somItem('right','-K2'),seq_no:3,evidences:[point(1,.8,.10)]};
  const lower={...somItem('lower','-K3'),seq_no:9,evidences:[point(1,.2,.6),point(1,.05,.01,'model_text')]};
  const unmapped={...somItem('unmapped','-K4'),primary_page:1,primary_bbox:anchor,evidences:[]};
  const merged={...left,id:'merged',member_ids:['left','right']};
  app.projectData.value={...fixture(),itemized_bom:[lower,right,unmapped,left],aggregated_bom:[merged]};
  app.pageTopology.value={...topo,candidates:[candidate(1,'-K1',{norm_bbox:left.evidences[2].norm_bbox})],excluded_candidates:[]};
  app.activeTab.value='primary';
  app.selectSomCandidate(app.filteredSomCandidates.value[0]);
  assert.equal(app.activeTab.value,'secondary');
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>it.id),['left','lower','right'],'Circuit order follows drawing columns before vertical position');
  assert.equal(app.pagePositions.value.get('left').number,1);
  assert.equal(app.pagePositions.value.get('lower').number,2);
  app.pageOrderMode.value='visual';
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>it.id),['left','right','lower'],'Visual order remains available for comparing a page top to bottom');
  app.pageOrderMode.value='circuit';
  app.projectData.value.pages=[{page_num:1,page_type:'INSTALL_LAYOUT'}];
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>it.id),['left','right','lower'],'Layout pages retain spatial reading order');
  app.projectData.value.pages=[];
  assert.equal(app.selectedItemId.value,'left','Clicking a drawing opens its individual item even when the purchase row groups devices');
  assert.equal(app.currentEvIndex.value,1,'A click on the second occurrence keeps that location selected');
  app.showSomLayer.value=true;
  const savedBox=app.filteredSomCandidates.value.find(c=>c.matched_item_id==='right');
  assert.equal(savedBox.detector_source,'BOM_EVIDENCE','An item with a saved symbol position is clickable even if the topology detector missed it');
  app.selectSomCandidate(savedBox);
  assert.equal(app.selectedItemId.value,'right');
  assert.equal(app.filteredSomCandidates.value.filter(c=>c.matched_item_id==='left').length,2,'Overlapping detector and saved evidence produce one marker per occurrence');
  app.showSomLayer.value=false;
  app.clearActiveBox();
  app.selectBomRow(left);
  assert.equal(app.currentPage.value,1,'Row navigation stays on the drawing being checked');
  assert.equal(app.activeBox.value.norm_bbox[0],.1);
  assert.equal(app.focusedCandidate.value,null,'A hidden candidate layer retains the standalone evidence highlight');
  app.showSomLayer.value=true;
  assert.equal(app.focusedCandidate.value?.matched_item_id,'left');
  app.selectEvidenceByIndex(1);
  assert.equal(app.focusedCandidate.value?.key,'1','Highlight the candidate at the chosen occurrence, including after stepping through evidence');
  app.somFilter.value='pending';
  assert.equal(app.focusedCandidate.value,null,'Filtered candidates cannot replace the visible focus box');
  app.somFilter.value='ALL';
  await app.locateItemEvidence(left,{...left.evidences[2],role:'model_text'});
  assert.equal(app.focusedCandidate.value,null,'Exact model text remains independently locatable');
  await app.locateItemEvidence(merged,left.evidences[2]);
  assert.equal(app.focusedCandidate.value?.key,'1','A summary row can focus its visible member at the same position');
  await app.locateItemEvidence(right,left.evidences[2]);
  assert.equal(app.focusedCandidate.value,null,'A nearby box belonging to another device cannot absorb the highlight');
  app.showSomLayer.value=false;
  app.searchQuery.value='-K2';
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>app.pagePositions.value.get(it.id).number),[3]);
  app.resetFilters();app.currentPage.value=2;
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>it.id),['left']);
  app.currentPage.value=3;
  assert.equal(app.currentTableItems.value.length,0,'A page without drawing evidence does not inherit the previous page');
  app.followDrawingPage.value=false;
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>it.id),['lower','right','unmapped','left']);
  app.activeTab.value='primary';
  assert.deepEqual(Array.from(app.currentTableItems.value,it=>it.id),['merged']);
  app.clearActiveBox();app.currentPage.value=1;app.pageTopology.value=null;
  app.projectData.value=fixture();
  app.currentProjectId.value='fixture';
  assert.equal(app.issueCounts.value.ALL,3);
  assert.equal(app.issueCounts.value.MODEL,1);
  assert.equal(app.issueCounts.value.QUANTITY,1,'Zero is a known quantity');
  assert.equal(app.issueCounts.value.SCOPE,1);
  app.focusIssue('QUANTITY');
  assert.equal(app.currentTableItems.value.length,1);
  assert.equal(app.currentTableItems.value[0].id,'unknown');
  app.resetFilters();
  app.searchQuery.value='3*70+2*35';
  assert.equal(app.currentTableItems.value[0].tag,'-W1','Search original specifications');
  app.resetFilters();
  assert.equal(app.filtersActive.value,false);
  assert.equal(app.visibleReviews.value.length,1);
  app.reviewStatusFilter.value='ALL';
  assert.equal(app.visibleReviews.value.length,2);
  const initial=app.projectData.value;
  const noise=Array.from({length:100},(_,i)=>({id:'raw'+i,status:'PENDING',issue_type:'UNBOUND_MODEL',page:1+(i%2)}));
  app.projectData.value={...initial,itemized_bom:[],review_items:noise};
  app.reviewStatusFilter.value='PENDING';
  assert.equal(app.visibleReviews.value.length,0,'Unbound tokens are coverage evidence, not procurement decisions');
  app.reviewStatusFilter.value='COVERAGE';
  assert.equal(app.visibleReviews.value.length,2,'Coverage checks preserve both pages');
  assert.equal(app.visibleReviews.value[0].members.length,50);
  app.reviewStatusFilter.value='PENDING';
  app.aiJob.value={status:'running'};
  assert.equal(app.visibleReviews.value.length,0,'Coverage findings are provisional while the first BOM is generated');
  app.aiJob.value=null;
  app.reviewStatusFilter.value='ALL';
  assert.equal(app.visibleReviews.value.length,100,'Original evidence stays accessible');
  app.projectData.value=initial;
  app.setWorkspaceMode('bom');
  assert.equal(JSON.parse(storage.get('electrical-bom-workspace')).mode,'bom');

  let pending = {};
  request = url=>new Promise(resolve=>{pending[url]=resolve;});
  const oldLoad=app.loadProject('old');
  const newLoad=app.loadProject('new');
  pending['/api/projects/new'](response(fixture('new')));
  await newLoad;
  pending['/api/projects/old'](response(fixture('old')));
  await oldLoad;
  assert.equal(app.projectData.value.project_id,'new','Late responses must not replace the selected project');
  assert.equal(app.currentProjectId.value,'new');

  let savedPayload;
  request=async(_url,options)=>{savedPayload=JSON.parse(options.body); return response(fixture('new'));};
  app.startEdit(app.projectData.value.aggregated_bom[1]);
  await app.saveEdit(app.drawerItem.value);
  assert.equal(savedPayload.quantity,0);
  assert.equal('seq_no' in savedPayload,false,'Do not fall back to a reused row number');
  app.startEdit(app.projectData.value.aggregated_bom[0]);
  app.editForm.value.quantity='';
  await app.saveEdit(app.drawerItem.value);
  assert.equal(savedPayload.quantity,null);
  app.startEdit(app.projectData.value.aggregated_bom[2]);
  app.editForm.value.remark = 'checked';
  await app.saveEdit(app.drawerItem.value);
  assert.equal(savedPayload.expected_values.quantity, 3);
  assert.equal(savedPayload.remark, 'checked');
  assert.equal('quantity' in savedPayload, false, 'Saving a note must not confirm a quantity');

  app.startEdit(app.projectData.value.aggregated_bom[2]);
  let finishSave;
  request = () => new Promise(resolve => { finishSave = resolve; });
  const lateSave = app.saveEdit(app.drawerItem.value);
  app.currentProjectId.value = 'another';
  app.projectData.value = fixture('another');
  finishSave(response(fixture('new')));
  await lateSave;
  assert.equal(app.projectData.value.project_id, 'another', 'A late save must not reopen the previous project');
  app.currentProjectId.value = 'new';
  app.projectData.value = fixture('new');

  request=async()=>{throw new Error('offline');};
  app.startEdit(app.projectData.value.aggregated_bom[0]);
  app.editForm.value.specification='retain draft';
  await app.saveEdit(app.drawerItem.value);
  assert.equal(app.editForm.value.specification,'retain draft');
  assert.equal(app.editingId.value,'unknown','A failed save must keep the editor open');
  assert.equal(app.savingEdit.value,false);
  app.projectData.value={...fixture('new'),aggregated_bom:[],itemized_bom:[]};
  assert.equal(app.drawerItem.value.id,'unknown','Keep the draft when background results regroup');
  assert.equal(app.editTargetChanged.value,true);
  let calls=0;
  request=async()=>{calls++;throw new Error('must not submit stale row');};
  await app.saveEdit(app.drawerItem.value);
  assert.equal(calls,0);
  app.cancelEdit();
  assert.equal(app.drawerItem.value,null);
  const independent={...fixture('new'),independent_bom:[],declared_parts:[],filtered_items:[],summary:{independent_analysis:{policy:'independent-schematic-v1',schematic_pages:[1],excluded_table_pages:[2],mixed_table_pages:[]}},reconciliation:[{id:'r',signature:'s',status:'TABLE_ONLY',decision:'PENDING',schematic_ids:[],declared_ids:['d'],schematic:[],declared:[{model:'TABLE-100'}],issues:[]},{id:'done',status:'MATCH',decision:'CONFIRM'}]};
  app.projectData.value=independent;
  app.currentPage.value=2;
  assert.equal(app.currentPageEligible.value,false);
  app.showAnalysisView.value=true;
  assert.match(app.pageImageUrl.value,/analysis-image/);
  assert.equal(app.reconciliationRows.value.length,1);
  app.reconciliationFilter.value='ALL';
  assert.equal(app.reconciliationRows.value.length,2);
  assert.equal(app.canAdoptReference(independent.reconciliation[0]),false);
  let resolveAction;
  request=()=>new Promise(resolve=>{resolveAction=resolve;});
  const action=app.handleReconciliation(independent.reconciliation[0],'CONFIRM');
  app.currentProjectId.value='other';app.projectData.value=fixture('other');
  resolveAction(response(independent));await action;
  assert.equal(app.projectData.value.project_id,'other','Late comparison actions must not replace another project');
  const quick={...fixture('quick'),extraction_mode:'PART_LIST_MODE',pages:[{page_num:1,page_type:'PART_LIST'}],declared_parts:[],summary:{table_extraction:{policy:'parts-table-only-v1',detected_pages:[1],selected_pages:[1],skipped_overlapping_pages:[]}}};
  app.projectData.value=quick;app.currentProjectId.value='quick';
  assert.equal(app.isTableExtraction.value,true);
  assert.equal(app.legacyTableProject.value,false,'Quick extraction is not a historical project');
  assert.equal(app.sourceLabel('EXPLICIT'),'部件表原文');
  app.openTableExtraction();
  assert.deepEqual(Array.from(app.tableSelectedPages.value),[1]);
  app.tablePageChoice.value='SELECT';app.tableSelectedPages.value=[];
  calls=0;request=async()=>{calls++;throw new Error('empty selection must not submit');};
  await app.createTableProject();assert.equal(calls,0);
  app.showTableDialog.value=false;
  const file={name:'drawing.pdf'};const event={target:{files:[file],value:'file'}};
  app.uploadWorkflow.value='TABLE_EXTRACT';app.handleFileUpload(event);
  assert.equal(app.uploadWorkflow.value,'SCHEMATIC_CHECK','Every upload defaults to independent checking');
  assert.equal(app.showUploadDialog.value,true);
  assert.equal(calls,0,'Choosing a file must wait for the explicit workflow selection');
  app.uploadWorkflow.value='TABLE_EXTRACT';let submitted;
  request=async(url,options)=>{
    if(url==='/api/projects/upload'){submitted=options.body;return response(quick);}
    if(url==='/api/projects')return response([{project_id:'quick'}]);
    if(url==='/api/project-load-errors')return response({});
    if(url==='/api/projects/quick')return response(quick);
    throw new Error('unexpected '+url);
  };
  await app.submitUpload();
  assert.equal(submitted.fields.get('workflow_mode'),'TABLE_EXTRACT');
  assert.equal(app.pendingUpload.value,null);
  assert.equal(app.showUploadDialog.value,false);
  context.window.location={href:''};
  app.downloadExcel();assert.equal(context.window.location.href,'/api/projects/quick/export/excel');
  app.downloadExcelDiagnostics();assert.equal(context.window.location.href,'/api/projects/quick/export/excel/diagnostics');
  const terminalRow={id:'x2_pt6',seq_no:1,tag:'-X2:1~2',tags:['-X2:1~2'],terminal_strip_tag:'-X2',
    name:'接线端子',model:'PT6',quantity:2,evidences:[
      {page:1,norm_bbox:[.1,.1,.13,.13],raw_text:'-X2',role:'contact_or_symbol'},
      {page:1,norm_bbox:[.2,.2,.21,.21],raw_text:'1',role:'terminal_symbol'},
      {page:1,norm_bbox:[.3,.2,.31,.21],raw_text:'2',role:'terminal_symbol'},
      {page:1,norm_bbox:[.2,.18,.21,.19],raw_text:'PT6',role:'model_text'}]};
  app.projectData.value={...fixture('term'),itemized_bom:[terminalRow],aggregated_bom:[terminalRow]};
  app.currentProjectId.value='term';app.currentPage.value=1;
  app.selectedItemId.value=terminalRow.id;app.selectedEvidences.value=terminalRow.evidences;
  app.activeBox.value={project_id:'term',view_mode:app.currentImageView.value,page:1,norm_bbox:[.2,.2,.21,.21]};
  assert.deepEqual(Array.from(app.uniqueTagEvidences.value,ev=>ev.raw_text),['1','2'],
    'Terminal evidence navigation must show every counted position');
  assert.equal(app.terminalHighlights.value.length,2,
    'Selecting a terminal material row must highlight all of its position symbols');
  app.currentPage.value=2;assert.equal(app.terminalHighlights.value.length,0);
  const u1={...somItem('u1','-U1'),evidences:[point(1,.19,.2,'main_circuit')]};
  const km1={...somItem('km1','-KM1'),evidences:[point(1,.3,.7,'main_circuit')]};
  app.projectData.value={...fixture('body'),pages:[{page_num:1,page_type:'MAIN_CIRCUIT'}],itemized_bom:[u1,km1],aggregated_bom:[u1,km1]};
  app.currentProjectId.value='body';app.currentPage.value=1;app.showAnalysisView.value=false;
  app.pageTopology.value={...topo,project_id:'body',view_mode:'original',candidates:[candidate(8,'-U1, -KM1',
    {norm_bbox:[.22,.19,.42,.4],anchor_norm_bbox:[.19,.2,.21,.22],bbox_source:'vector_device_outline'})],excluded_candidates:[]};
  app.activeTab.value='secondary';app.followDrawingPage.value=true;app.showSomLayer.value=true;
  assert.equal(app.filteredSomCandidates.value[0].matched_item_id,'u1',
    'A combined OCR label must use its unique tag anchor to identify the device');
  assert.equal(app.filteredSomCandidates.value[0].display_label,'-U1');
  assert.equal(app.filteredSomCandidates.value.filter(c=>c.matched_item_id==='u1').length,1,
    'The body outline replaces the small tag marker instead of duplicating it');
  app.pageTopology.value={...app.pageTopology.value,candidates:[candidate(8,'-U1, -KM1',
    {norm_bbox:[.1,.19,.3,.8],anchor_norm_bbox:[.19,.2,.21,.22],detector_source:'VECTOR_SYMBOL+TAG_ANCHOR'})]};
  assert.ok(app.filteredSomCandidates.value[0].norm_bbox[3]-app.filteredSomCandidates.value[0].norm_bbox[1]<.1,
    'An unverified legacy candidate must not stretch from U1 down into KM1');
  const printedGrid=Array.from({length:10},(_,number)=>({number,x:.065+number*.088}));
  const km2={...somItem('km2','-KM2'),evidences:[
    {page:2,role:'main_circuit',norm_bbox:[.44,.65,.48,.69],raw_text:'-KM2'},
    {page:2,role:'cross_reference',norm_bbox:[.47,.70,.49,.71],raw_text:'/16.7',
      cad_refs:[{sheet:16,column:7,target_pages:[16],resolved:true}]},
    {page:16,role:'coil',norm_bbox:[.68,.2,.72,.24],raw_text:'-KM2'}]};
  app.projectData.value={...fixture('grid'),page_count:16,pages:[
    {page_num:2,sheet_no:'2',page_type:'MAIN_CIRCUIT',extraction_diagnostics:{drawing_grid_columns:printedGrid}},
    {page_num:16,sheet_no:'16',page_type:'CONTROL_CIRCUIT',extraction_diagnostics:{drawing_grid_columns:printedGrid}}],
    itemized_bom:[km2],aggregated_bom:[km2]};
  app.currentProjectId.value='grid';app.currentPage.value=2;app.showSomLayer.value=false;
  assert.equal(app.pageGridBands.value.length,10);
  const jump=app.crossPageLinks.value.find(link=>link.evidence.raw_text==='/16.7');
  assert.equal(jump.targetPage,16);
  assert.deepEqual(Array.from(jump.targetBox),[.68,.2,.72,.24]);
  await app.jumpToCrossReference(jump);
  assert.equal(app.currentPage.value,16);
  assert.equal(app.activeGridColumn.value.number,7);
  assert.equal(app.selectedItemId.value,'km2');
  assert.deepEqual(Array.from(app.activeBox.value.norm_bbox),[.68,.2,.72,.24]);
  const preview={...fixture('preview'),page_count:7,pages:Array.from({length:7},(_,i)=>({page_num:i+1})),
    itemized_bom:[terminalRow],aggregated_bom:[terminalRow]};
  context.window.location={search:'?project=preview&page=7&tag=-X14'};
  app.currentProjectId.value='';
  request=async url=>response(url==='/api/projects'?[{project_id:'other'},{project_id:'preview'}]:
    url==='/api/project-load-errors'?{}:url==='/api/projects/preview'?preview:{});
  await app.fetchProjects();
  assert.equal(app.currentProjectId.value,'preview');
  assert.equal(app.currentPage.value,7);
  assert.equal(app.searchQuery.value,'-X14');
  console.log('Frontend checks passed (drafts, races, reconciliation, workflow, compact and diagnostic exports)');
})().catch(error=>{console.error(error); process.exitCode=1;});
