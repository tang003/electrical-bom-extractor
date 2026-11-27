const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');

function editor({qwenConfig=null}={}) {
  let state, focused, payload;
  const requests=[];
  const context = vm.createContext({console, URLSearchParams, setTimeout:()=>0, clearTimeout:()=>{},
    localStorage:{getItem:()=>null, setItem:()=>{}},
    document:{createElement:()=>({}), activeElement:null, querySelector:()=>({focus:()=>{focused='name';}}),
      getElementById:id=>({focus:()=>{focused=id;}})},
    confirm:()=>false,
    fetch:async (url, options) => {
      requests.push({url, method:options?.method || 'GET'});
      if (url === '/api/qwen-config' && !options) {
        return {ok:true, json:async()=>qwenConfig || {has_api_key:false,key_source:'none',can_edit:true}};
      }
      if (options?.method === 'PUT') {
        payload = JSON.parse(options.body);
        const project = JSON.parse(JSON.stringify(state.projectData.value));
        project.itemized_bom[0].name = payload.name;
        return {ok:true, json:async()=>project};
      }
      return {ok:true, json:async()=>({records:[]})};
    }});
  vm.runInContext(fs.readFileSync(path.join(root,'frontend/assets/vue.global.prod.js'),'utf8'),context);
  context.Vue.createApp = options=>({mount:()=>{state=options.setup();}});
  context.Vue.onMounted=()=>{};
  context.Vue.watch=()=>{};
  vm.runInContext(fs.readFileSync(path.join(root,'frontend/assets/app.js'),'utf8'),context);
  const item={id:'drive',seq_no:1,tag:'-D1',name:'伺服驱动器',model:'XML-230-18',quantity:1,
    unit:'个',procurement_scope:'unknown',entity_role:'device',evidences:[],source_type:'EXPLICIT'};
  state.currentProjectId.value='project';
  state.projectData.value={project_id:'project',extraction_mode:'SCHEMATIC_MODE',pages:[],summary:{},
    itemized_bom:[item],aggregated_bom:[],filtered_items:[]};
  state.activeTab.value='secondary';
  state.followDrawingPage.value=false;
  state.currentPage.value=8;
  return {state,item,context,requests,get focused(){return focused;},get payload(){return payload;}};
}

test('server-managed Qwen key is read-only in the configuration dialog', async()=>{
  const e=editor({qwenConfig:{has_api_key:true,key_source:'environment',can_edit:false}}), s=e.state;
  await s.openVlmModal('item');
  assert.equal(s.qwenHasKey.value,true);
  assert.equal(s.qwenKeySource.value,'environment');
  assert.equal(s.qwenKeyEditable.value,false);
  s.qwenKeyValue.value='should-not-be-saved';
  await s.saveQwenKey();
  assert.equal(e.requests.some(request=>request.url==='/api/qwen-config' && request.method==='PUT'),false);
});

test('combined editor focuses the issue field and protects an unsaved draft', async()=>{
  const e=editor(), s=e.state;
  s.openItemIssue(e.item,'待选型');
  await e.context.Vue.nextTick();
  assert.equal(e.focused,'bom-editor-model');
  assert.equal(s.editDirty.value,false);
  s.editForm.value.name='X轴伺服驱动器';
  assert.equal(s.editDirty.value,true);
  assert.equal(s.closeItemPanel(),false);
  assert.equal(s.editForm.value.name,'X轴伺服驱动器');
  assert.equal(s.editingId.value,e.item.id);
});

test('save sends only changed fields with conflict guards and preserves list context', async()=>{
  const e=editor(), s=e.state;
  s.openItemDetails(e.item);
  s.searchQuery.value='D1';
  s.editForm.value.name='X轴伺服驱动器';
  await s.saveEdit(e.item);
  assert.equal(e.payload.name,'X轴伺服驱动器');
  assert.equal('quantity' in e.payload,false);
  assert.equal('model' in e.payload,false);
  assert.equal(e.payload.expected_values.name,'伺服驱动器');
  assert.equal(e.payload.view_mode,'itemized');
  assert.equal(s.currentPage.value,8);
  assert.equal(s.searchQuery.value,'D1');
  assert.equal(s.selectedItemId.value,e.item.id);
  assert.equal(s.editingId.value,'');
});

test('list count states the scope and a pristine editor does not submit', async()=>{
  const e=editor(), s=e.state;
  assert.equal(s.tableCountLabel.value,'全项目 1 项');
  s.searchQuery.value='missing';
  assert.equal(s.tableCountLabel.value,'全项目匹配 0 / 1 项');
  s.openItemDetails(e.item);
  await s.saveEdit(e.item);
  assert.equal(e.payload,undefined);
  assert.equal(s.closeItemPanel(),true);
});

test('model ownership and component issues are distinct from missing selection',()=>{
  const e=editor(),s=e.state;
  e.item.model='未提取到型号(待核实)';
  e.item.evidences=[{role:'model_candidate',raw_text:'RELAY-12',page:1}];
  assert.equal(s.modelPendingLabel(e.item),'型号关联待核');
  assert.equal(s.matchesIssue(e.item,'MODEL'),false);
  assert.equal(s.matchesIssue(e.item,'ASSOCIATION'),true);
  e.item.model='RELAY-12';e.item.component_parts=[{kind:'socket',status:'PENDING',relationship:'separate_part'}];
  assert.equal(s.matchesIssue(e.item,'COMPONENTS'),true);
  assert.ok(s.itemIssues(e.item).includes('本体/附件待核'));
});

test('evidence previews keep the editor and its unsaved fields open', async () => {
  const e = editor(), s = e.state;
  s.projectData.value.pages = [{page_num: 2, width: 842, height: 595}];
  e.item.evidences = [
    {page: 2, raw_text: 'XML-230-18', norm_bbox: [.84, .45, .98, .48]},
    {page: 2, raw_text: '230 V'},
  ];
  s.openItemDetails(e.item);
  s.editForm.value.name = 'X轴伺服驱动器';
  s.toggleEvidencePreview(0);
  await e.context.Vue.nextTick();
  assert.equal(s.evidencePreview.value.url, '/api/projects/project/pages/2/image?dpi=150');
  assert.ok(s.evidencePreview.value.markerStyle);
  assert.equal(s.editingId.value, e.item.id);
  assert.equal(s.editDirty.value, true);
  assert.equal(s.currentPage.value, 8);

  s.toggleEvidencePreview(1);
  assert.equal(s.evidencePreview.value.evidence.raw_text, '230 V');
  assert.equal(s.evidencePreview.value.markerStyle, undefined);
  s.toggleEvidencePreview(1);
  assert.equal(s.evidencePreview.value, null);
  assert.equal(s.editForm.value.name, 'X轴伺服驱动器');
});
