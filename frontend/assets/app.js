const { createApp, ref, computed, onMounted, nextTick, watch } = Vue;

const SOURCE_LABELS = {
  EXPLICIT: "原图明文",
  UNRESOLVED: "型号待核实",
  VISION_RECOGNIZED: "视觉识别待核实",
  TOPOLOGY_INFERRED: "拓扑推导",
  RULE_COMPLETED: "规则补全",
  ESTIMATED_AUX: "工程估算",
  MANUAL_EDITED: "人工已改",
};

function somTag(value) {
  const tag=String(value || '').normalize('NFKC').trim().toUpperCase();
  return tag && !/^[=+-]/.test(tag) ? '-' + tag : tag;
}

function somItemTags(item) {
  if(Array.isArray(item.tag_members)) return [...new Set([...(item.tags?.length ? item.tags : [item.tag]),...item.tag_members,item.terminal_strip_tag].map(somTag).filter(Boolean))];
  const result=new Set();
  if (item.terminal_strip_tag) result.add(somTag(item.terminal_strip_tag));
  for (const value of item.tags?.length ? item.tags : [item.tag]) {
    const tag=somTag(value);result.add(tag);
    const range=tag.match(/^(.*?[A-Z])(\d+)[~～至](?:([A-Z]+))?(\d+)$/);
    if (range && (!range[3] || range[1].endsWith(range[3])) && Number(range[4])>=Number(range[2]) && Number(range[4])-Number(range[2])<=100) {
      for(let n=Number(range[2]);n<=Number(range[4]);n++) result.add(range[1]+(/^0\d/.test(range[2]) ? String(n).padStart(range[2].length,'0') : n));
    }
    const parts=tag.split(/[,，、]/);const prefix=parts[0]?.match(/^(.*?[A-Z])\d+$/)?.[1];
    if(parts.length>1 && prefix) for(const part of parts) result.add(/^\d+$/.test(part.trim()) ? prefix+part.trim() : somTag(part));
  }
  return [...result].filter(Boolean);
}

function hasDrawingBox(box) {
  return Array.isArray(box) && box.length===4 && box.every(Number.isFinite) && box[0]>=0 && box[1]>=0 && box[2]<=1 && box[3]<=1 && box[2]>box[0] && box[3]>box[1];
}

function centeredDrawingBox(box, minWidth, minHeight) {
  if (!hasDrawingBox(box)) return box;
  const width=Math.max(minWidth,box[2]-box[0]),height=Math.max(minHeight,box[3]-box[1]);
  const left=Math.max(0,Math.min(1-width,(box[0]+box[2]-width)/2));
  const top=Math.max(0,Math.min(1-height,(box[1]+box[3]-height)/2));
  return [left,top,left+width,top+height];
}

function drawingBoxStyle(box, minWidth, minHeight) {
  const [left,top,right,bottom]=centeredDrawingBox(box,minWidth,minHeight);
  return {left:(left*100)+'%',top:(top*100)+'%',width:((right-left)*100)+'%',height:((bottom-top)*100)+'%'};
}

function candidateDisplayBox(candidate, ownerBox) {
  const box=candidate.norm_bbox, anchor=candidate.anchor_norm_bbox;
  if (candidate.bbox_source === 'vector_device_outline' || candidate.detector_source !== 'VECTOR_SYMBOL+TAG_ANCHOR' ||
      !hasDrawingBox(box) || !hasDrawingBox(anchor)) return box;
  const tooTall=box[3]-box[1]>.16 && box[3]-box[1]>8*(anchor[3]-anchor[1]);
  const tooWide=box[2]-box[0]>.22 && box[2]-box[0]>8*(anchor[2]-anchor[0]);
  const display=tooTall || tooWide
    ? [Math.max(0,anchor[0]-.012),Math.max(0,anchor[1]-.018),
       Math.min(1,anchor[2]+.012),Math.min(1,anchor[3]+.018)] : box;
  if (!hasDrawingBox(ownerBox) || candidate.bbox_source!=='tag_and_nearby_geometry') return display;
  // Nearby wire text can stretch a text-anchored box into the preceding device.
  return [Math.max(display[0],Math.min(anchor[0],ownerBox[0])-.006),
          Math.max(display[1],Math.min(anchor[1],ownerBox[1])-.012),display[2],display[3]];
}

function somOverlap(a,b) {
  if (!a || !b || a.length!==4 || b.length!==4) return false;
  const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])) * Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  const smaller=Math.min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]));
  return smaller>0 && intersection/smaller>=.3;
}

function terminalStripCandidateKey(candidate) {
  if (candidate.detector_source === 'PARTS_TABLE_ROW' || candidate.excluded) return '';
  return somTag(candidate.terminal_strip_tag || candidate.tag || candidate.device_tag)
    .match(/^(.*-X\d+[A-Z]?)(?::\d+(?:[~～至-]\d+)?)?$/)?.[1] || '';
}

function orderedTerminalLocations(locations) {
  const numeric=locations.filter(location=>location.role==='terminal_symbol' && /^\d+$/.test(String(location.raw_text || '').trim()));
  if (!numeric.length) return locations;
  const extras=locations.filter(location=>!numeric.includes(location));
  const seen=new Set();
  const repeated=numeric.some(location=>{
    const key=location.page+':'+String(location.raw_text).trim();
    if (seen.has(key)) return true;
    seen.add(key);return false;
  });
  const ordered=[...numeric].sort((a,b)=>a.page-b.page || (repeated
    ? a.norm_bbox[1]-b.norm_bbox[1] || a.norm_bbox[0]-b.norm_bbox[0]
    : Number(a.raw_text)-Number(b.raw_text)));
  return [...ordered,...extras];
}

function compactTerminalCandidateLabels(candidates) {
  const display=candidates.map(candidate=>({...candidate,terminalCompact:false,terminalBadgeVisible:true,
    terminalStripLabel:terminalStripCandidateKey(candidate)}));
  const byStrip=new Map();
  display.forEach((candidate,index)=>{
    const key=candidate.terminalStripLabel;
    if (!key || !hasDrawingBox(candidate.norm_bbox)) return;
    if (!byStrip.has(key)) byStrip.set(key,[]);
    const box=hasDrawingBox(candidate.anchor_norm_bbox) ? candidate.anchor_norm_bbox : candidate.norm_bbox;
    byStrip.get(key).push({index,x:(box[0]+box[2])/2,y:(box[1]+box[3])/2});
  });
  for (const entries of byStrip.values()) {
    const pending=[...entries];
    while (pending.length) {
      const group=[pending.shift()];
      for (let i=0;i<group.length;i++) {
        for (let j=pending.length-1;j>=0;j--) {
          const dx=Math.abs(group[i].x-pending[j].x),dy=Math.abs(group[i].y-pending[j].y);
          if ((dx<=.09 && dy<=.025) || (dx<=.025 && dy<=.09)) group.push(pending.splice(j,1)[0]);
        }
      }
      if (group.length<2) continue;
      const width=Math.max(...group.map(entry=>entry.x))-Math.min(...group.map(entry=>entry.x));
      const height=Math.max(...group.map(entry=>entry.y))-Math.min(...group.map(entry=>entry.y));
      group.sort((a,b)=>width>=height ? a.x-b.x || a.y-b.y : a.y-b.y || a.x-b.x);
      group.forEach((entry,index)=>{
        display[entry.index].terminalCompact=true;
        display[entry.index].terminalBadgeVisible=index===0;
      });
    }
  }
  return display;
}

function readWorkspacePreferences() {
  try {
    const value = JSON.parse(localStorage.getItem('electrical-bom-workspace') || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}

function saveWorkspacePreferences(value) {
  try { localStorage.setItem('electrical-bom-workspace', JSON.stringify({...readWorkspacePreferences(), ...value})); } catch {}
}

createApp({
  setup() {
    const projects = ref([]);
    const currentRole = ref('engineer');
    const currentProjectId = ref("");
    const projectData = ref(null);
    const pageReferenceIndex = ref({project_id:'',links:[]});
    const loading = ref(false);
    const loadingMessage = ref('正在加载项目…');
    const showUploadDialog=ref(false);
    const pendingUpload=ref(null);
    const uploadWorkflow=ref('SCHEMATIC_CHECK');
    const showTableDialog=ref(false);
    const tablePageChoice=ref('AUTO');
    const tableSelectedPages=ref([]);
    const showCadSheets = ref(false);
    const cadSelectedPages = ref([]);
    watch(() => projectData.value?.project_id, () => {
      cadSelectedPages.value = (projectData.value?.pages || []).map(p => p.page_num);
      showCadSheets.value = false;
    });

    // Split pane width (%)
    const preferences = readWorkspacePreferences();
    const drawingLayoutUpdated = preferences.drawingLayoutVersion === 2;
    const leftPaneWidth = ref(Math.max(28, Math.min(78, Number(drawingLayoutUpdated ? preferences.paneWidth : 60) || 60)));
    const workspaceMode = ref(drawingLayoutUpdated && ['compare','bom','drawing'].includes(preferences.mode) ? preferences.mode : 'compare');
    const overviewExpanded = ref(drawingLayoutUpdated && preferences.overviewExpanded === true);
    saveWorkspacePreferences({drawingLayoutVersion:2, paneWidth:leftPaneWidth.value, mode:workspaceMode.value, overviewExpanded:overviewExpanded.value});
    const isResizingSplit = ref(false);

    // PDF viewer & interactive box-drawing state
    const pdfViewportRef = ref(null);
    const pdfStageRef = ref(null);
    const currentPage = ref(1);
    const showDrawingGrid = ref(true);
    const activeGridColumn = ref(null);
    const loadedPageImage = ref('');
    const pageImageError = ref('');
    const imageRetry = ref(0);
    const highResolutionImage = ref('');
    let highResolutionRequest = '';
    const showAnalysisView = ref(false);
    const reconciliationFilter = ref('PENDING');
    const reconciliationBusy = ref(false);
    const independentInfo = computed(() => projectData.value?.summary?.independent_analysis || null);
    const tableInfo=computed(() => projectData.value?.summary?.table_extraction || null);
    const isTableExtraction=computed(() => tableInfo.value?.policy === 'parts-table-only-v1');
    const legacyTableProject = computed(() => projectData.value?.extraction_mode === 'PART_LIST_MODE' && !independentInfo.value && !isTableExtraction.value);
    const currentPageEligible = computed(() => !independentInfo.value || independentInfo.value.schematic_pages.includes(currentPage.value));
    const reconciliationLabels = {MATCH:'已提取字段一致',DIFFERENCE:'存在差异',TABLE_ONLY:'仅原表有记录',TABLE_ACCESSORY:'原表另列附件',SCHEMATIC_ONLY:'仅原理图有记录',AMBIGUOUS:'对应关系待核',UNLINKED_REFERENCE:'原表无对应位号',REFERENCE_UNAVAILABLE:'暂无可用原表'};
    const pendingReconciliation=computed(()=>(projectData.value?.reconciliation || []).filter(r=>
      r.decision==='PENDING' && !['MATCH','REFERENCE_UNAVAILABLE'].includes(r.status)));
    const reconciliationRows = computed(() => reconciliationFilter.value==='PENDING' ? pendingReconciliation.value :
      (projectData.value?.reconciliation || []).filter(r => reconciliationFilter.value==='ALL' || r.status===reconciliationFilter.value));
    const decisionLabel = value => ({PENDING:'待处理',CONFIRM:'已确认差异',KEEP_SCHEMATIC:'保留原理图结果',ADOPT_MODEL:'已人工采纳原表型号',ADD_REFERENCE:'已人工补入，数量待核'}[value] || value);
    const workingItem = row => projectData.value?.itemized_bom.concat(projectData.value?.filtered_items || []).find(i => i.id === row.schematic_ids[0]);
    function canAdoptReference(row) { return !['AMBIGUOUS','TABLE_ACCESSORY'].includes(row.status) && row.schematic_ids.length === 1 && row.declared_ids.length === 1 && !row.issues.some(i => ['ASSEMBLY_REVIEW','PARTIAL_TAG_COVERAGE'].includes(i)) && !!workingItem(row) && !!row.declared[0]?.model && !/^(未提取|图纸未注|待核|待定)/.test(row.declared[0].model); }
    function locateComparison(fact, reference) {
      const evidence=fact.evidences?.[0]; if (!evidence) return;
      showAnalysisView.value = !reference; showSomLayer.value = false;
      if (workspaceMode.value === 'bom') setWorkspaceMode('compare');
      jumpToLocation(evidence.page,evidence.norm_bbox,fact.tag || fact.name);
    }
    async function handleReconciliation(row, action) {
      if (reconciliationBusy.value) return;
      if (action === 'ADOPT_MODEL' && !confirm('采纳原表型号到采购明细？独立识别快照保持原样，数量与范围需要分别确认。')) return;
      if (action === 'ADD_REFERENCE' && !confirm('将这条原表记录作为人工物料补入？数量会留空待确认，且不会计入独立识别结果。')) return;
      const pid=currentProjectId.value; reconciliationBusy.value=true;
      try {
        const response=await fetch(`/api/projects/${pid}/reconciliation/${row.id}/action`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({signature:row.signature,action,expected_working_model:workingItem(row)?.model})});
        const data=await response.json();
        if (!response.ok) {showToast(data.detail || '核对处理失败');return;}
        if (pid === currentProjectId.value) projectData.value=data;
        showToast('处理已保存，独立识别快照保留。');
      } catch {showToast('连接失败，请重试。');} finally {reconciliationBusy.value=false;}
    }
    async function createIndependentProject() {
      if (loading.value) return;
      const pid=currentProjectId.value;
      loading.value=true;loadingMessage.value='正在屏蔽图内清单并建立独立分析项目，原审核结果保留…';
      try {
        const response=await fetch(`/api/projects/${pid}/independent-analysis`,{method:'POST'});
        const data=await response.json();
        if (!response.ok) {showToast(data.detail || '建立项目失败');return;}
        await fetchProjects(); await loadProject(data.project_id);activeTab.value='reconciliation';
        showToast('独立分析项目已建立；可启动 AI 完善原理图识别。');
      } catch {showToast('建立项目失败，请检查服务。');} finally {loading.value=false;}
    }
    function openTableExtraction() {
      showTableDialog.value=true;tablePageChoice.value='AUTO';
      tableSelectedPages.value=(projectData.value?.pages || []).filter(p=>p.page_type==='PART_LIST').map(p=>p.page_num);
    }
    async function createTableProject() {
      if (loading.value || (tablePageChoice.value==='SELECT' && !tableSelectedPages.value.length)) return;
      const pid=currentProjectId.value;
      loading.value=true;loadingMessage.value='正在建立部件表提取项目，原项目和人工审核记录保留…';
      try {
        const response=await fetch(`/api/projects/${pid}/table-extraction`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pages:tablePageChoice.value==='AUTO'?null:tableSelectedPages.value})});
        const data=await response.json();
        if (!response.ok) {showToast(data.detail || '提取失败');return;}
        showTableDialog.value=false;
        await fetchProjects();await loadProject(data.project_id);setWorkspaceMode('compare');
        showToast(`已建立部件表提取项目，共 ${data.itemized_bom.length} 行；未经原理图核对。`);
      } catch {showToast('提取连接失败，请重试。');} finally {loading.value=false;}
    }
    const zoomPercent = ref(100);
    const ZOOM_MIN = 25;
    const ZOOM_MAX = 400;
    const autoFocusZoom = ref(true);
    const isPanning = ref(false);
    const isDrawBoxMode = ref(false);
    const drawingBoxRect = ref(null); // [x0, y0, x1, y1] in 0..1 normalized space
    const boxCropBase64 = ref("");
    const boxRecognizedTokens = ref([]);
    const tokenFillTarget = ref("model");
    let drawStartX = 0;
    let drawStartY = 0;
    let panStartX = 0;
    let panStartY = 0;
    let panScrollLeft = 0;
    let panScrollTop = 0;
    let hasMoved = false;

    const activeBox = ref(null); // { page, norm_bbox: [x0,y0,x1,y1], label }
    const selectedItemId = ref("");
    const selectedDrawingHit = ref(null); // Exact drawing occurrence, which may share one BOM row with other terminals.
    let drawingHitSequence = 0;
    const selectedEvidences = ref([]);
    const currentEvIndex = ref(0);
    const evDedupeByTag = ref(true);
    const evExpanded = ref(false);

    // Right pane tabs & filters
    const activeTab = ref("primary"); // "primary" | "secondary" | "inventory" | "reviews" | "filtered"
    const deviceInventory = ref(null);
    const inventoryLoading = ref(false);
    const inventoryFamily = ref('ALL');
    const inventoryFamilies = computed(() => [...new Set((deviceInventory.value?.devices || []).map(item => item.family))].sort());
    const inventoryItems = computed(() => (deviceInventory.value?.devices || []).filter(item =>
      inventoryFamily.value === 'ALL' || item.family === inventoryFamily.value));
    const followDrawingPage = ref(true);
    const pageOrderMode = ref('circuit'); // circuit columns, or visual rows
    const searchQuery = ref("");
    const sourceFilter = ref("ALL");
    const categoryFilter = ref("ALL"); // "ALL" | "PRIMARY" | "AUX"
    const issueFilter = ref('ALL');
    const reviewStatusFilter = ref('PENDING');
    const expandedReview = ref('');

    // Inline editing state
    const editingId = ref("");
    const editForm = ref({});
    const detailItemId = ref('');
    const savingEdit = ref(false);
    const editInitialValues = ref('');
    const editItemSnapshot = ref(null);
    let panelReturnFocus = null;
    let projectLoadVersion = 0;

    // Modals & SoM
    const showAddModal = ref(false);
    const addForm = ref({
      tag: "-NEW1",
      brand: "施耐德",
      name: "断路器",
      model: "",
      unit: "个",
      quantity: 1,
      remark: "工程师网页新增",
      primary_page: 1,
    });

    const showRulesModal = ref(false);
    const rulesJsonText = ref("");
    const showVlmModal = ref(false);
    const aiConfigTab = ref('page');
    const aiConfigError = ref('');
    const vlmForm = ref({
      provider: "本地模型接口",
      base_url: "http://localhost:60563/v1",
      model: "gpt-6.1-sol",
      api_protocol: "responses",
      reasoning_effort: "medium",
      review_reasoning_effort: "high",
      auto_run_on_upload: true,
      max_high_review_pages: 6,
      api_key: "",
    });
    const showSomLayer = ref(true);
    const somFilter = ref('ALL');
    const somShowExcluded = ref(false);
    const selectedSomId = ref('');
    const topologyLoading = ref(false);
    const topologyError = ref('');
    const somKinds = [
      {key:'linked',label:'已关联',description:'完整位号已关联清单，不代表已人工审核'},
      {key:'pending',label:'待核',description:'未关联清单、对应不唯一或元件证据不足'},
      {key:'reference',label:'参考',description:'明确的辅助触点或重复引用位置，不单独增加采购数量'},
      {key:'filtered',label:'备查',description:'已识别并列入范围外备查，采购范围仍可核对'},
    ];
    const pageTopology = ref(null);
    let topologyRequestVersion = 0;
    const showYoloModal = ref(false);
    const yoloStatus = ref(null);
    const aiLoading = ref(false);
    const aiJob = ref(null);
    const aiBusy = computed(() => aiLoading.value || ["queued", "running", "reviewing", "merging"].includes(aiJob.value?.status));
    const aiStatusLabel = computed(() => ({queued:"排队中",running:"逐页识别中",reviewing:"疑点复核中",merging:"合并中",completed:"已完成",partial:"部分完成",failed:"失败",cancelled:"已取消",interrupted:"已中断"}[aiJob.value?.status] || ""));
    let aiPollVersion = 0;
    watch([() => currentProjectId.value, () => projectData.value?.summary?.ai_job_id], ([pid, jid]) => {
      const version = ++aiPollVersion;
      aiJob.value = null;
      if (pid && jid && projectData.value?.project_id === pid) pollAiJob(pid, jid, version);
    });

    async function pollAiJob(pid, jid, version) {
      if (version !== aiPollVersion) return;
      try {
        const response = await fetch(`/api/projects/${pid}/ai-jobs/${jid}`);
        if (!response.ok || version !== aiPollVersion) return;
        const job = await response.json();
        if (version !== aiPollVersion || pid !== currentProjectId.value) return;
        aiJob.value = job;
        if (["queued", "running", "reviewing", "merging"].includes(aiJob.value.status)) {
          setTimeout(() => pollAiJob(pid, jid, version), 2000);
        } else if (["completed", "partial"].includes(aiJob.value.status)) {
          const latest = await fetch(`/api/projects/${pid}`);
          if (latest.ok) {
            const data = await latest.json();
            if (version === aiPollVersion) applyProjectResponse(data);
          }
        }
      } catch (error) {
        if (version === aiPollVersion) setTimeout(() => pollAiJob(pid, jid, version), 4000);
      }
    }

    async function startAiExtraction(pages = null) {
      if (!displayedProjectIsCurrent() || aiBusy.value) return;
      const pid = currentProjectId.value;
      aiLoading.value = true;
      try {
        const response = await fetch(`/api/projects/${currentProjectId.value}/ai-extract`, {
          method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({pages}),
        });
        const job = await response.json();
        if (!response.ok) { alert(job.detail || "无法启动 AI 任务"); return; }
        if (pid !== currentProjectId.value || !displayedProjectIsCurrent()) return;
        projectData.value.summary.ai_job_id = job.id;
        showToast("AI 后台任务已启动，可以继续查看图纸和审核物料。");
      } finally { aiLoading.value = false; }
    }

    async function cancelAiJob() {
      if (!displayedProjectIsCurrent() || aiJob.value?.project_id !== currentProjectId.value) return;
      const response = await fetch(`/api/projects/${currentProjectId.value}/ai-jobs/${aiJob.value.id}/cancel`, {method:"POST"});
      if (response.ok) {
        const job = await response.json();
        if (job.project_id === currentProjectId.value && job.id === aiJob.value?.id) aiJob.value = job;
      }
    }
    const toastMessage = ref("");

    function showToast(msg) {
      toastMessage.value = msg;
      setTimeout(() => {
        if (toastMessage.value === msg) toastMessage.value = "";
      }, 3200);
    }

    async function fetchProjects() {
      const res = await fetch("/api/projects");
      projects.value = await res.json();
      const recovery = await fetch("/api/project-load-errors");
      if (recovery.ok) {
        const errors = await recovery.json();
        if (Object.keys(errors).length) showToast(`部分项目无法恢复，原文件已保留：${Object.keys(errors).join("、")}`);
      }
      if (projects.value.length > 0 && !currentProjectId.value) {
        const remembered = readWorkspacePreferences().projectId;
        const params = new URLSearchParams(window.location?.search || '');
        const requested = params.get('project');
        const chosen = projects.value.find(p => p.project_id === requested)?.project_id ||
          projects.value.find(p => p.project_id === remembered)?.project_id || projects.value[0].project_id;
        await loadProject(chosen);
        if (requested === chosen) {
          const page = Number(params.get('page'));
          if (Number.isInteger(page) && page >= 1 && page <= (projectData.value?.page_count || 0)) {
            currentPage.value = page;
            if (showSomLayer.value) await loadPageTopology();
          }
          if (params.get('tag')) searchQuery.value = params.get('tag');
        }
      } else if (projects.value.length === 0) {
        currentProjectId.value = "";
        projectData.value = null;
      }
    }

    async function clearAllProjects() {
      if (!confirm("将所有项目移入回收站？之后可以在项目管理中恢复。")) return;
      const res = await fetch("/api/projects/clear", { method: "POST" });
      if (res.ok) {
        projects.value = [];
        currentProjectId.value = "";
        projectData.value = null;
        activeBox.value = null;
        selectedItemId.value = "";
        selectedEvidences.value = [];
        showToast("项目已移入回收站，可在项目管理中恢复");
      }
    }

    async function openInventory() {
      activeTab.value = 'inventory';
      if (deviceInventory.value || inventoryLoading.value || !currentProjectId.value) return;
      const projectId = currentProjectId.value;
      inventoryLoading.value = true;
      try {
        const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/device-inventory`);
        if (!response.ok) throw new Error('位号盘点加载失败');
        const data = await response.json();
        if (currentProjectId.value === projectId) deviceInventory.value = data;
      } catch (error) {
        showToast(error.message || '位号盘点加载失败');
      } finally {
        inventoryLoading.value = false;
      }
    }

    function locateInventoryItem(item) {
      const location = item.occurrences.find(entry => projectData.value?.pages?.[entry.page - 1]?.page_type !== 'INSTALL_LAYOUT') || item.occurrences[0];
      if (location) jumpToLocation(location.page, location.bbox, item.tag);
    }

    async function loadProject(projId) {
      if (!projId) return;
      const version = ++projectLoadVersion;
      loading.value = true;
      loadingMessage.value = '正在加载项目…';
      try {
        currentProjectId.value = projId;
        const res = await fetch(`/api/projects/${projId}`);
        if (!res.ok) throw new Error('项目加载失败，请重试');
        const data = await res.json();
        if (version !== projectLoadVersion) return;
        projectData.value = data;
        pageReferenceIndex.value = {project_id:'',links:[]};
        loadPageReferences(projId, version);
        deviceInventory.value = null;
        inventoryFamily.value = 'ALL';
        showAddModal.value = false;
        saveWorkspacePreferences({projectId:projId});
        currentPage.value = 1;
        showAnalysisView.value = false;
        reconciliationFilter.value = 'PENDING';
        zoomPercent.value = 100;
        activeBox.value = null;
        selectedItemId.value = "";
        selectedEvidences.value = [];
        editingId.value = "";
        detailItemId.value = '';
        fieldClassification.value = null;
        fieldHistory.value = {item_id:'', records:[]};
        editForm.value = {};
        editItemSnapshot.value = null;
        searchQuery.value = '';
        sourceFilter.value = 'ALL';
        issueFilter.value = 'ALL';
        reviewStatusFilter.value = 'PENDING';
        expandedReview.value = '';
        followDrawingPage.value = !isPartListMode.value;
        const hasOnlySummary = !data.itemized_bom?.length && data.aggregated_bom?.length;
        activeTab.value = workspaceMode.value === 'compare' && !isPartListMode.value && !hasOnlySummary ? 'secondary' : 'primary';
        categoryFilter.value = "ALL";
        if (showSomLayer.value) {
          await loadPageTopology();
        }
      } catch (error) {
        if (version === projectLoadVersion) {
          currentProjectId.value = projectData.value?.project_id || '';
          showToast(error.message || '项目加载失败');
        }
      } finally {
        if (version === projectLoadVersion) loading.value = false;
      }
    }

    async function loadPageReferences(projectId, version) {
      try {
        const response=await fetch(`/api/projects/${encodeURIComponent(projectId)}/page-references`);
        if (!response.ok) return;
        const index=await response.json();
        if (version===projectLoadVersion && projectData.value?.project_id===projectId)
          pageReferenceIndex.value=index;
      } catch (_) {
        // BOM evidence links remain available if a drawing has no readable index.
      }
    }

    const currentPageInfo = computed(() => {
      if (!projectData.value || !projectData.value.pages) return null;
      return projectData.value.pages.find((p) => p.page_num === currentPage.value) || null;
    });

    const gridColumnsForPage = page => (projectData.value?.pages || [])
      .find(info => info.page_num === page)?.extraction_diagnostics?.drawing_grid_columns || [];
    const pageGridColumns = computed(() => gridColumnsForPage(currentPage.value)
      .filter(entry => Number.isInteger(entry.number) && Number.isFinite(entry.x) && entry.x > 0 && entry.x < 1)
      .sort((a,b) => a.x-b.x));
    const pageGridBands = computed(() => pageGridColumns.value.map((entry,index,columns) => ({
      number:entry.number, center:entry.x,
      left:index ? (columns[index-1].x+entry.x)/2 : 0,
      right:index+1<columns.length ? (entry.x+columns[index+1].x)/2 : 1,
    })));
    const crossPageLinks = computed(() => {
      const links=[], seen=new Set();
      for (const item of [...(projectData.value?.itemized_bom || []),...(projectData.value?.filtered_items || [])]) {
        for (const evidence of item.evidences || []) {
          if (evidence.page !== currentPage.value || evidence.role !== 'cross_reference' || !hasDrawingBox(evidence.norm_bbox)) continue;
          for (const reference of evidence.cad_refs || []) {
            const sheet=Number(reference.sheet), column=Number(reference.column);
            const pages=(reference.target_pages?.length ? reference.target_pages :
              (projectData.value?.pages || []).filter(info => Number(info.sheet_no) === sheet).map(info => info.page_num));
            const targetPage=pages.length === 1 ? pages[0] : null;
            if (!targetPage || !Number.isInteger(column)) continue;
            const key=[item.id,evidence.page,evidence.norm_bbox.join(','),targetPage,column].join(':');
            if (seen.has(key)) continue;
            seen.add(key);
            const gridX=gridColumnsForPage(targetPage).find(entry => entry.number === column)?.x;
            const targets=(item.evidences || []).filter(entry => entry.page === targetPage && hasDrawingBox(entry.norm_bbox) &&
              ['coil','main_circuit','instrument_module','device','contact_or_symbol'].includes(entry.role));
            targets.sort((a,b) => (Number.isFinite(gridX)
              ? Math.abs((a.norm_bbox[0]+a.norm_bbox[2])/2-gridX)-Math.abs((b.norm_bbox[0]+b.norm_bbox[2])/2-gridX) : 0));
            const target=targets[0];
            const targetBox=target && (!Number.isFinite(gridX) ||
              Math.abs((target.norm_bbox[0]+target.norm_bbox[2])/2-gridX)<.12) ? target.norm_bbox : null;
            links.push({key,item,evidence,sheet,column,targetPage,targetBox,gridX});
          }
        }
      }
      if (pageReferenceIndex.value.project_id === currentProjectId.value) {
        for (const reference of pageReferenceIndex.value.links || []) {
          if (reference.source_page !== currentPage.value) continue;
          if (links.some(link => link.targetPage === reference.target_page &&
              link.column === reference.column && somOverlap(link.evidence.norm_bbox, reference.source_bbox))) continue;
          links.push({key:'index:'+reference.key,item:null,
            evidence:{page:reference.source_page,norm_bbox:reference.source_bbox,raw_text:reference.source_text},
            sheet:reference.sheet,column:reference.column,targetPage:reference.target_page,
            targetBox:null,gridX:reference.grid_x,columnVerified:reference.column_verified});
        }
      }
      return links;
    });
    const visibleCrossPageLinks = computed(() => zoomPercent.value >= 75 ? crossPageLinks.value : []);
    function crossPageLinkStyle(link) {
      const box=link.evidence.norm_bbox;
      const x=box[2] < .95 ? box[2]+.002 : Math.max(.01,box[0]-.025);
      return {left:(x*100)+'%',top:(((box[1]+box[3])/2)*100)+'%'};
    }

    const pageImageUrl = computed(() => {
      if (!currentProjectId.value || !currentPage.value) return "";
      const version = independentInfo.value?.view_sha256 || '';
      return `/api/projects/${currentProjectId.value}/pages/${currentPage.value}/${showAnalysisView.value && independentInfo.value ? 'analysis-image' : 'image'}?dpi=150&v=${version}&retry=${imageRetry.value}`;
    });
    const displayedPageImageUrl = computed(() => highResolutionImage.value || pageImageUrl.value);
    const currentImageView = computed(() => showAnalysisView.value && independentInfo.value ? 'analysis' : 'original');
    const pageImageReady = computed(() => !!pageImageUrl.value && loadedPageImage.value===pageImageUrl.value && projectData.value?.project_id===currentProjectId.value);
    watch(pageImageUrl,()=>{
      loadedPageImage.value='';pageImageError.value='';
      highResolutionImage.value='';highResolutionRequest='';
    },{flush:'sync'});
    watch([pageImageUrl, pageImageReady, zoomPercent], ([url, ready, zoom], previous, onCleanup) => {
      if (!ready || zoom < 165 || highResolutionImage.value || highResolutionRequest === url) return;
      const timer = setTimeout(() => {
        highResolutionRequest = url;
        const image = new Image();
        image.onload = async () => {
          try {
            if (image.decode) await image.decode();
            if (pageImageUrl.value === url && pageImageReady.value) highResolutionImage.value = image.src;
          } catch { /* Keep the readable preview if high-resolution decoding fails. */ }
        };
        image.src = url.replace('dpi=150', 'dpi=220');
      }, 180);
      onCleanup(() => clearTimeout(timer));
    });
    watch([currentProjectId,currentPage,showAnalysisView],()=>{
      activeBox.value=null;selectedSomId.value='';selectedDrawingHit.value=null;drawingBoxRect.value=null;isDrawBoxMode.value=false;
      pageTopology.value=null;++topologyRequestVersion;topologyLoading.value=false;topologyError.value='';
    },{flush:'sync'});
    function onPageImageLoad(event) {
      const img=event.currentTarget;
      if (img.isConnected && img.getAttribute('src')===displayedPageImageUrl.value) {loadedPageImage.value=pageImageUrl.value;pageImageError.value='';}
    }
    function onPageImageError(event) {
      if (!event.currentTarget.isConnected || event.currentTarget.getAttribute('src') !== displayedPageImageUrl.value) return;
      if (highResolutionImage.value) { highResolutionImage.value=''; return; }
      pageImageError.value='图纸加载失败，请重试';
    }
    function retryPageImage() { imageRetry.value++; }
    function toggleAnalysisView() {
      showAnalysisView.value=!showAnalysisView.value;
      if(showSomLayer.value) loadPageTopology();
    }

    const detectedSomCandidates = computed(() => {
      if (!pageTopology.value || pageTopology.value.page_num!==currentPage.value || pageTopology.value.project_id!==currentProjectId.value || pageTopology.value.view_mode!==currentImageView.value) return [];
      const scope=somTag((currentPageInfo.value?.function_code || '')+(currentPageInfo.value?.location_code || ''));
      const items=projectData.value?.itemized_bom?.length ? projectData.value.itemized_bom : projectData.value?.aggregated_bom || [];
      const filteredItems=projectData.value?.filtered_items || [];
      const detected=[...(pageTopology.value.candidates || []),...(pageTopology.value.excluded_candidates || []).map(c=>({...c,excluded:true}))].map(c=>{
        const tags=[...new Set([c.device_tag,c.tag].filter(Boolean).flatMap(value=>
          [somTag(value),...String(value).split(/[,，、]/).map(somTag)]).filter(Boolean))];
        const qualified=new Set(tags.filter(t=>/^[=+]/.test(t)));
        if (!qualified.size && scope) tags.forEach(t=>qualified.add(scope+t));
        const evidenceAtAnchor=item=>(item.evidences || []).filter(e=>e.page===currentPage.value && !['table_row','schematic_grid','model_text','model_candidate','device_description','device_annotation','specification'].includes(e.role) && somOverlap(c.anchor_norm_bbox || c.norm_bbox,e.norm_bbox));
        const matchesTag=item=>somItemTags(item).some(value=>{
          const tag=somTag(value);
          if (qualified.size) return qualified.has(tag);
          if (tags.includes(tag)) return true;
          const local=tag.match(/-\d*[A-Z][A-Z0-9]*(?:[.-]\d+)*$/)?.[0];
          return tags.includes(local) && evidenceAtAnchor(item).length>0;
        });
        const matches=items.filter(matchesTag);
        const tableRow=c.detector_source==='PARTS_TABLE_ROW';
        const sourceItem=tableRow && (isTableExtraction.value || legacyTableProject.value) ? items.find(i=>i.id===c.source_item_id) : null;
        const anchoredMatches=matches.filter(candidate=>evidenceAtAnchor(candidate).some(e=>!['cross_reference','duplicate_reference','model_inference'].includes(e.role)));
        const item=tableRow ? sourceItem || null : matches.length===1 ? matches[0] : anchoredMatches.length===1 ? anchoredMatches[0] : null;
        const filteredMatches=!item && !matches.length && !tableRow ? filteredItems.filter(matchesTag) : [];
        const filteredItem=filteredMatches.length===1 ? filteredMatches[0] : null;
        const stripTag=terminalStripCandidateKey(c);
        const terminalGroupRows=!item && stripTag && somTag(c.tag || c.device_tag)===stripTag && matches.length>1 &&
          matches.every(candidate=>candidate.entity_role==='terminal' && somTag(candidate.terminal_strip_tag)===stripTag)
          ? matches : [];
        const terminalStripAnchor=!tableRow && stripTag && somTag(c.tag || c.device_tag)===stripTag &&
          (item?.terminal_strip_tag || terminalGroupRows.length) && hasDrawingBox(c.anchor_norm_bbox);
        const roles=item ? evidenceAtAnchor(item).map(e=>e.role) : [];
        const mainEstablished=item && item.entity_role!=='contact' && item.quantity_basis!=='contact_only';
        const isReference=(tableRow && !sourceItem) || roles.some(r=>['cross_reference','duplicate_reference'].includes(r) || (mainEstablished && ['contact','auxiliary_contact'].includes(r)));
        const contactOnly=item && (item.entity_role==='contact' || item.quantity_basis==='contact_only') && !isReference;
        const kind=c.excluded ? 'excluded' : isReference ? 'reference' : (item && !contactOnly) || terminalGroupRows.length ? 'linked' : filteredItem ? 'filtered' : 'pending';
        const status=terminalGroupRows.length ? '端子排已关联（分型号）' : {linked:'已关联清单',pending:'待核候选',reference:'辅助 / 参考位置',filtered:'范围外备查',excluded:'已排除标注'}[kind];
        const reason=c.excluded ? c.exclusion_reason || '疑似非元件标注，默认不显示' : tableRow ? '按部件表原始行定位；位号展开仅用于关联，不拆改原表行和数量。' : terminalGroupRows.length ? '端子排位号对应多种端子材料；请按各端子位置分别查看型号和数量。' : isReference ? '此处为明确的触点或引用证据，不重复增加采购数量。' : contactOnly ? '目前只有触点证据，采购本体或组件关系仍需核对。' : item ? '完整位号已对应清单；关联成功不代表人工审核通过。' : filteredItem ? '已识别到位号，当前列在范围外备查；是否纳入采购清单仍需按项目范围核对。' : matches.length>1 ? '完整位号对应多条清单记录，尚不能唯一关联。' : '尚未按完整位号建立清单关联；有型号线索也不代表已确认。';
        const ownerEvidence=item && evidenceAtAnchor(item).find(e=>['main_circuit','coil','device','instrument_module','contact_or_symbol'].includes(e.role));
        return {...c,kind,status,reason,
          norm_bbox:terminalStripAnchor ? centeredDrawingBox(c.anchor_norm_bbox,.032,.028) : centeredDrawingBox(candidateDisplayBox(c,ownerEvidence?.norm_bbox),.018,.018),
          terminalStripAnchor:Boolean(terminalStripAnchor),
          display_label:item && tags.length>1 ? item.tag : c.display_label,
          terminal_strip_tag:item?.terminal_strip_tag || (terminalGroupRows.length ? stripTag : ''),
          terminal_group_rows:terminalGroupRows.map(row=>({id:row.id,tag:row.tag,model:row.model,quantity:row.quantity,unit:row.unit})),
          terminal_group_summary:terminalGroupRows.map(row=>`${row.tag} ${row.model || '待选型'} × ${row.quantity ?? '待核'}${row.unit || '片'}`).join('；'),
          matched_item_id:item?.id || '',matched_filtered_id:filteredItem?.id || '',item_name:item?.name || filteredItem?.name || (terminalGroupRows.length ? '端子排（分型号）' : c.reference_name) || '',item_model:item?.model || filteredItem?.model || c.reference_model || '',key:String(c.som_id ?? c.som_label)};
      });
      const nearby=detected.filter(c=>c.matched_item_id && c.bbox_source==='tag_and_nearby_geometry' &&
        hasDrawingBox(c.anchor_norm_bbox) && hasDrawingBox(c.norm_bbox) && !c.terminal_strip_tag);
      nearby.sort((a,b)=>(a.anchor_norm_bbox[0]+a.anchor_norm_bbox[2])-(b.anchor_norm_bbox[0]+b.anchor_norm_bbox[2]));
      for (let i=1;i<nearby.length;i++) {
        const left=nearby[i-1],right=nearby[i];
        const ay=(left.anchor_norm_bbox[1]+left.anchor_norm_bbox[3])/2;
        const by=(right.anchor_norm_bbox[1]+right.anchor_norm_bbox[3])/2;
        if (Math.abs(ay-by)>.025 || left.norm_bbox[2]<=right.norm_bbox[0] ||
            left.norm_bbox[3]<=right.norm_bbox[1] || right.norm_bbox[3]<=left.norm_bbox[1]) continue;
        const boundary=(left.norm_bbox[2]+right.norm_bbox[0])/2;
        if (boundary-left.norm_bbox[0]>=.018 && right.norm_bbox[2]-boundary>=.018) {
          left.norm_bbox=[...left.norm_bbox.slice(0,2),boundary,left.norm_bbox[3]];
          right.norm_bbox=[boundary,right.norm_bbox[1],...right.norm_bbox.slice(2)];
        }
      }
      return detected;
    });
    const somCandidates = computed(() => {
      const detected = detectedSomCandidates.value;
      if (currentImageView.value !== 'original' || !showSomLayer.value) return detected;
      const candidates = [...detected];
      const terminalRows=new Map();
      for (const item of projectData.value?.itemized_bom || []) {
        if (item.entity_role!=='terminal' || !item.terminal_strip_tag) continue;
        const strip=somTag(item.terminal_strip_tag);
        if (!terminalRows.has(strip)) terminalRows.set(strip,[]);
        terminalRows.get(strip).push(item);
      }
      for (const [strip,rows] of terminalRows) {
        const localStrip=strip.match(/-X\d+[A-Z]?$/)?.[0] || strip;
        const headers=new Map();
        for (const item of rows) for (const evidence of item.evidences || []) {
          if (evidence.page!==currentPage.value || evidence.role!=='main_circuit' ||
              ![strip,localStrip].includes(somTag(evidence.raw_text)) || !hasDrawingBox(evidence.norm_bbox)) continue;
          headers.set(evidence.norm_bbox.join(','),evidence.norm_bbox);
        }
        for (const [position,box] of headers) {
          if (candidates.some(candidate=>[strip,localStrip].includes(terminalStripCandidateKey(candidate)) &&
              somOverlap(candidate.anchor_norm_bbox || candidate.norm_bbox,box))) continue;
          const key='terminal-header-'+strip+'-'+position;
          candidates.push({key,som_label:key,tag:strip,display_label:strip,
            norm_bbox:centeredDrawingBox(box,.032,.028),anchor_norm_bbox:box,
            terminalStripAnchor:true,terminal_strip_tag:strip,
            terminal_group_rows:rows.length>1 ? rows.map(row=>({id:row.id,tag:row.tag,model:row.model,quantity:row.quantity,unit:row.unit})) : [],
            terminal_group_summary:rows.length>1 ? rows.map(row=>`${row.tag} ${row.model || '待选型'} × ${row.quantity ?? '待核'}${row.unit || '片'}`).join('；') : '',
            matched_item_id:rows.length===1 ? rows[0].id : '',
            item_name:rows.length===1 ? rows[0].name : '端子排（分型号）',item_model:rows.length===1 ? rows[0].model : '',
            kind:'linked',status:rows.length===1 ? '清单已定位' : '端子排已关联（分型号）',
            reason:'按原图位号文字坐标定位；各端子位置和型号仍逐位核对。',detector_source:'BOM_STRIP_HEADER'});
        }
      }
      if (!pageOrderActive.value) return candidates;
      for (const {item,locations} of pagePositions.value.values()) {
        for (const [index,location] of locations.entries()) {
          const box = location.norm_bbox;
          if (candidates.some(c=>!c.excluded && c.matched_item_id===item.id && somOverlap(c.anchor_norm_bbox || c.norm_bbox,box))) continue;
          const reference = ['contact','auxiliary_contact','cross_reference','duplicate_reference'].includes(location.role);
          const key = 'evidence-' + item.id + '-' + index;
          const terminalNumber=item.terminal_strip_tag && ['terminal_symbol','terminal_position'].includes(location.role) && /^\d+[A-Z]?$/.test(String(location.raw_text || '').trim())
            ? String(location.raw_text).trim() : '';
          candidates.push({key, som_label:key, tag:item.tag,
            display_label:terminalNumber ? item.terminal_strip_tag + ':' + terminalNumber : item.tag || item.name,
            terminal_strip_tag:item.terminal_strip_tag || '',
            norm_bbox:centeredDrawingBox(box,.018,.018),anchor_norm_bbox:box,
            matched_item_id:item.id, item_name:item.name, item_model:item.model, kind:reference ? 'reference' : 'linked',
            status:reference ? '清单引用位置' : '清单已定位', reason:'按本页已保存的图纸坐标定位。', detector_source:'BOM_EVIDENCE'});
        }
      }
      return candidates;
    });
    const somCounts = computed(() => {
      const counts={ALL:0,linked:0,pending:0,reference:0,filtered:0,excluded:0};
      somCandidates.value.forEach(c=>{counts[c.kind]++;if(c.kind!=='excluded')counts.ALL++;});
      return counts;
    });
    const filteredSomCandidates = computed(() => somCandidates.value.filter(c=>c.kind==='excluded' ? somShowExcluded.value : somFilter.value==='ALL' || c.kind===somFilter.value));
    const overlaySomCandidates = computed(() => compactTerminalCandidateLabels(filteredSomCandidates.value));
    const focusedCandidate = computed(() => {
      const box = activeBox.value;
      if (!showSomLayer.value || !box?.itemIds?.length || box.page !== currentPage.value ||
          box.project_id !== currentProjectId.value || box.view_mode !== currentImageView.value) return null;
      const matches = filteredSomCandidates.value.filter(candidate => !candidate.excluded &&
        box.itemIds.includes(candidate.matched_item_id) &&
        somOverlap(candidate.anchor_norm_bbox || candidate.norm_bbox, box.norm_bbox));
      return matches.length === 1 ? matches[0] : null;
    });
    const highlightedCandidateKey = computed(() => focusedCandidate.value?.key || selectedSomId.value);
    const somFilterLabel = computed(() => somFilter.value==='ALL' ? '全部候选' : somKinds.find(k=>k.key===somFilter.value)?.label || '全部候选');
    const visibleSomBranches = computed(() => {
      const visible=new Set(filteredSomCandidates.value.filter(c=>!c.excluded).map(c=>c.som_label));
      return (pageTopology.value?.branches || []).map(b=>({...b,node_labels:b.node_labels.filter(n=>visible.has(n)),chain:b.chain.filter((_,i)=>visible.has(b.node_labels[i]))})).filter(b=>b.node_labels.length>1);
    });

    async function loadPageTopology() {
      if (!currentProjectId.value || !currentPage.value || !showSomLayer.value) return;
      const pid=currentProjectId.value, page=currentPage.value, view=currentImageView.value, version=++topologyRequestVersion;
      pageTopology.value=null; selectedSomId.value=''; topologyError.value=''; topologyLoading.value=true;
      somFilter.value='ALL';somShowExcluded.value=false;
      const current=()=>version===topologyRequestVersion && pid===currentProjectId.value && page===currentPage.value && view===currentImageView.value && showSomLayer.value;
      try {
        const res=await fetch(`/api/projects/${pid}/pages/${page}/topology?view=${view}`);
        if (!res.ok) throw new Error('候选框读取失败，请重试');
        const data=await res.json();
        if (current()) pageTopology.value={...data,project_id:pid,view_mode:view};
      } catch (error) { if (current()) topologyError.value=error.message || '候选框读取失败，请重试'; }
      finally { if (version===topologyRequestVersion) topologyLoading.value=false; }
    }

    async function changePage(p) {
      if (!projectData.value) return;
      const clamped = Math.max(1, Math.min(projectData.value.page_count, p));
      const previousLeft = pdfViewportRef.value?.scrollLeft || 0;
      clearActiveBox();drawingBoxRect.value=null;
      currentPage.value = clamped;
      nextTick(() => {
        const viewport=pdfViewportRef.value;
        if (viewport) { viewport.scrollLeft=previousLeft;viewport.scrollTop=0; }
      });
      if (showSomLayer.value) {
        await loadPageTopology();
      }
    }

    async function onPageChanged() {
      clearActiveBox();drawingBoxRect.value=null;
      nextTick(() => { if (pdfViewportRef.value) pdfViewportRef.value.scrollTop=0; });
      if (showSomLayer.value) {
        await loadPageTopology();
      }
    }

    async function toggleSomLayer() {
      showSomLayer.value = !showSomLayer.value;
      if (showSomLayer.value) {
        await loadPageTopology();
      } else {
        ++topologyRequestVersion;topologyLoading.value=false;topologyError.value='';selectedSomId.value='';
        pageTopology.value = null;
      }
    }

    // Interactive SoM Box Click -> Reverse-Highlight BOM Row in Right Table
    function selectSomCandidate(cand) {
      if (highlightedCandidateKey.value === cand.key) {
        clearActiveBox();
        return;
      }
      activeBox.value = null; selectedSomId.value=cand.key;
      selectedItemId.value='';selectedEvidences.value=[];selectedDrawingHit.value=null;
      if (!projectData.value) return;
      if (cand.excluded) return;
      if (cand.matched_filtered_id) {
        const matched=(projectData.value.filtered_items || []).find(item=>item.id===cand.matched_filtered_id);
        if (matched) {
          if (workspaceMode.value==='drawing') setWorkspaceMode('compare');
          activeTab.value='filtered';
          selectedItemId.value=matched.id;
          selectedEvidences.value=matched.evidences || [];
          const clickedBox=cand.anchor_norm_bbox || cand.norm_bbox;
          currentEvIndex.value=Math.max(0,displayedEvidences.value.findIndex(ev=>ev.page===currentPage.value && somOverlap(clickedBox,ev.norm_bbox)));
          nextTick(()=>document.getElementById('bom-row-'+matched.id)?.scrollIntoView({behavior:'smooth',block:'center'}));
          return;
        }
      }
      if (cand.terminal_group_rows?.length) {
        if (workspaceMode.value === 'drawing') setWorkspaceMode('compare');
        activeTab.value=detailTab.value;followDrawingPage.value=true;
        resetFilters();searchQuery.value=cand.terminal_strip_tag;
        nextTick(() => document.getElementById('bom-row-' + cand.terminal_group_rows[0].id)?.scrollIntoView({behavior:'smooth',block:'center'}));
        return;
      }
      if (workspaceMode.value === 'compare' && cand.matched_item_id && projectData.value.itemized_bom?.some(it=>it.id===cand.matched_item_id)) {
        activeTab.value = detailTab.value;
        followDrawingPage.value = true;
      }
      let list=activeTab.value==='secondary' ? secondaryBomList.value : primaryBomList.value;
      let matched=list.find(it=>it.id===cand.matched_item_id || it.member_ids?.includes(cand.matched_item_id));
      if (!matched && cand.matched_item_id) {
        activeTab.value='primary';list=primaryBomList.value;
        matched=list.find(it=>it.id===cand.matched_item_id || it.member_ids?.includes(cand.matched_item_id));
      }
      if (matched) {
        resetFilters();
        if (!['primary','secondary'].includes(activeTab.value)) activeTab.value='primary';
        selectedItemId.value = matched.id;
        selectedEvidences.value = pageOrderActive.value ? pagePositions.value.get(matched.id)?.locations || [] : matched.evidences || [];
        const clickedBox = cand.anchor_norm_bbox || cand.norm_bbox;
        currentEvIndex.value = Math.max(0, displayedEvidences.value.findIndex(ev => ev.page === currentPage.value && somOverlap(clickedBox,ev.norm_bbox)));
        const terminalEvidence = matched.terminal_strip_tag && displayedEvidences.value.find(ev =>
          ev.page === currentPage.value && ['terminal_symbol','terminal_position'].includes(ev.role) &&
          /^\d+[A-Z]?$/.test(String(ev.raw_text || '').trim()) && somOverlap(clickedBox, ev.norm_bbox));
        const terminalNumber = terminalEvidence ? String(terminalEvidence.raw_text).trim() :
          String(cand.display_label || cand.tag || '').match(/:(\d+[A-Z]?)$/i)?.[1] || '';
        const label = terminalNumber && matched.terminal_strip_tag
          ? `${matched.terminal_strip_tag}:${terminalNumber}` : String(cand.display_label || cand.tag || matched.tag || '').trim();
        selectedDrawingHit.value = {itemId:matched.id,label,page:currentPage.value,sequence:++drawingHitSequence};
        nextTick(() => {
          const rowEl = document.getElementById("bom-row-" + matched.id);
          if (rowEl) {
            rowEl.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        });
      }
    }

    let wheelAccumulated = 0;
    let wheelLastAt = -Infinity;
    let wheelDirection = 0;
    let wheelGestureUsed = false;

    // Unmodified wheel turns one page per gesture; Ctrl/Meta-wheel zooms at the pointer.
    function handleCanvasWheel(e) {
      const vp = pdfViewportRef.value;
      if (!vp || isDrawBoxMode.value || isPanning.value) return;

      if (!e.ctrlKey && !e.metaKey) {
        if (e.shiftKey) { vp.scrollLeft += e.deltaY || e.deltaX || 0; return; }
        if (Math.abs(e.deltaX || 0) > Math.abs(e.deltaY || 0)) { vp.scrollLeft += e.deltaX; return; }
        const delta = e.deltaY * (e.deltaMode === 1 ? 32 : e.deltaMode === 2 ? vp.clientHeight : 1);
        const direction = Math.sign(delta);
        if (!direction || !projectData.value) return;
        const now = Number.isFinite(e.timeStamp) ? e.timeStamp : Date.now();
        if (now-wheelLastAt > 280 || direction !== wheelDirection) {
          wheelAccumulated=0;wheelGestureUsed=false;
        }
        wheelLastAt=now;wheelDirection=direction;
        if (wheelGestureUsed) return;
        wheelAccumulated += delta;
        if (Math.abs(wheelAccumulated) < 80) return;
        wheelAccumulated=0;wheelGestureUsed=true;
        const nextPage=currentPage.value+direction;
        if (nextPage>=1 && nextPage<=projectData.value.page_count) changePage(nextPage);
        return;
      }

      const rect = vp.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const contentX = vp.scrollLeft + mouseX;
      const contentY = vp.scrollTop + mouseY;

      const oldZoom = zoomPercent.value;
      const rawDelta = e.deltaY * (e.deltaMode === 1 ? 32 : e.deltaMode === 2 ? vp.clientHeight : 1);
      if (!rawDelta) return;
      const amount = Math.max(2, Math.min(20, Math.round(Math.abs(rawDelta) * .25)));
      const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, oldZoom + (rawDelta < 0 ? amount : -amount)));
      if (newZoom === oldZoom) return;

      const scaleRatio = newZoom / oldZoom;
      zoomPercent.value = newZoom;

      nextTick(() => {
        vp.scrollLeft = contentX * scaleRatio - mouseX;
        vp.scrollTop = contentY * scaleRatio - mouseY;
      });
    }

    function toggleDrawBoxMode() {
      isDrawBoxMode.value = !isDrawBoxMode.value;
      drawingBoxRect.value = null;
      if (isDrawBoxMode.value) {
        showToast("✏️ 已进入【鼠标拉框补录】模式：请在图纸任意元件上按住左键拖出矩形框！");
      }
    }

    function _getNormalizedStagePoint(e) {
      const stage = pdfStageRef.value;
      if (!stage) return null;
      const rect = stage.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      return [nx, ny];
    }

    let panDownTime = 0;
    let panDownX = 0;
    let panDownY = 0;

    function clearActiveBox() {
      selectedSomId.value='';
      activeBox.value = null;
      activeGridColumn.value = null;
      selectedItemId.value = "";
      selectedDrawingHit.value = null;
      selectedEvidences.value = [];
      currentEvIndex.value = 0;
    }

    function handleCanvasClick(e) {
      // Blank canvas click intentionally does NOT cancel activeBox.
      // Highlighting stays locked until user presses Esc, clicks corner ✕, or re-clicks table row.
    }

    function handleEvScrollWheel(e) {
      const el = e.currentTarget;
      if (!el) return;
      el.scrollLeft += (e.deltaY || e.deltaX) * 1.5;
    }

    // Click-and-Drag Panning OR Interactive Box Drawing
    function startPan(e) {
      if (e.button !== 0) return;
      panDownTime = Date.now();
      panDownX = e.clientX;
      panDownY = e.clientY;
      if (isDrawBoxMode.value) {
        const pt = _getNormalizedStagePoint(e);
        if (!pt) return;
        drawStartX = pt[0];
        drawStartY = pt[1];
        drawingBoxRect.value = [pt[0], pt[1], pt[0], pt[1]];
        return;
      }
      const vp = pdfViewportRef.value;
      if (!vp) return;
      isPanning.value = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      panScrollLeft = vp.scrollLeft;
      panScrollTop = vp.scrollTop;
    }

    function movePan(e) {
      if (isDrawBoxMode.value && drawingBoxRect.value) {
        const pt = _getNormalizedStagePoint(e);
        if (!pt) return;
        const x0 = Math.min(drawStartX, pt[0]);
        const y0 = Math.min(drawStartY, pt[1]);
        const x1 = Math.max(drawStartX, pt[0]);
        const y1 = Math.max(drawStartY, pt[1]);
        drawingBoxRect.value = [x0, y0, x1, y1];
        return;
      }
      if (!isPanning.value) return;
      const vp = pdfViewportRef.value;
      if (!vp) return;
      const dx = e.clientX - panStartX;
      const dy = e.clientY - panStartY;
      vp.scrollLeft = panScrollLeft - dx;
      vp.scrollTop = panScrollTop - dy;
    }

    async function endPan() {
      if (isDrawBoxMode.value && drawingBoxRect.value) {
        const [x0, y0, x1, y1] = drawingBoxRect.value;
        drawingBoxRect.value = null;
        if (x1 - x0 >= 0.008 && y1 - y0 >= 0.008) {
          await recognizeDrawnBoxOnPage([x0, y0, x1, y1]);
        }
        return;
      }
      isPanning.value = false;
    }

    async function recognizeDrawnBoxOnPage(normBbox) {
      if (!currentProjectId.value) return;
      activeBox.value = {
        page: currentPage.value,
        norm_bbox: normBbox,
        label: "✏️ 人工框选区域",
      };
      try {
        const res = await fetch(
          `/api/projects/${currentProjectId.value}/pages/${currentPage.value}/box-recognize`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ norm_bbox: normBbox }),
          }
        );
        if (!res.ok) {
          showToast("⚠️ 框选区域解析失败");
          return;
        }
        const data = await res.json();
        boxCropBase64.value = data.crop_base64 || "";
        boxRecognizedTokens.value = data.all_tokens || [];
        tokenFillTarget.value = "model";
        const sug = data.suggested || {};
        addForm.value = {
          tag: sug.tag || `-P${currentPage.value}_NEW`,
          brand: sug.brand || "施耐德",
          name: sug.name || "电气元器件",
          model: sug.model || "",
          unit: sug.unit || "个",
          quantity: sug.quantity || 1,
          remark: `图纸第${currentPage.value}页人工框选提取 [${normBbox.map((v) => v.toFixed(3)).join(",")}]`,
          primary_page: currentPage.value,
          primary_bbox: data.norm_bbox || normBbox,
        };
        isDrawBoxMode.value = false;
        showAddModal.value = true;
      } catch (err) {
        console.error(err);
      }
    }

    // Double-Click to Toggle 220% Zoom at Cursor
    function handleCanvasDblClick(e) {
      const vp = pdfViewportRef.value;
      if (!vp) return;
      const rect = vp.getBoundingClientRect();
      const relX = (vp.scrollLeft + (e.clientX - rect.left)) / Math.max(1, vp.scrollWidth);
      const relY = (vp.scrollTop + (e.clientY - rect.top)) / Math.max(1, vp.scrollHeight);

      const targetZoom = zoomPercent.value < 160 ? 220 : 100;
      zoomPercent.value = targetZoom;

      nextTick(() => {
        vp.scrollLeft = Math.max(0, relX * vp.scrollWidth - rect.width / 2);
        vp.scrollTop = Math.max(0, relY * vp.scrollHeight - rect.height / 2);
      });
    }

    function stepZoom(delta) {
      setZoomAndCenter(Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoomPercent.value + delta)));
    }

    function fitDrawingPage() {
      const viewport=pdfViewportRef.value;
      const page=currentPageInfo.value;
      if (!viewport || !page?.width || !page?.height) return;
      const usableWidth=Math.max(1,viewport.clientWidth-40);
      const usableHeight=Math.max(1,viewport.clientHeight-40);
      const pageHeightAtWidth=usableWidth*page.height/page.width;
      setZoomAndCenter(Math.max(ZOOM_MIN,Math.min(100,Math.floor(usableHeight/pageHeightAtWidth*100))));
    }

    function resetZoom() {
      zoomPercent.value = 100;
      nextTick(() => {
        const vp = pdfViewportRef.value;
        if (vp) {
          vp.scrollLeft = 0;
          vp.scrollTop = 0;
        }
      });
    }

    function setZoomAndCenter(targetZoom) {
      const vp = pdfViewportRef.value;
      if (!vp) {
        zoomPercent.value = targetZoom;
        return;
      }
      const centerRatioX = (vp.scrollLeft + vp.clientWidth / 2) / Math.max(1, vp.scrollWidth);
      const centerRatioY = (vp.scrollTop + vp.clientHeight / 2) / Math.max(1, vp.scrollHeight);
      zoomPercent.value = targetZoom;
      nextTick(() => {
        vp.scrollLeft = Math.max(0, centerRatioX * vp.scrollWidth - vp.clientWidth / 2);
        vp.scrollTop = Math.max(0, centerRatioY * vp.scrollHeight - vp.clientHeight / 2);
      });
    }

    function panCenterToActiveBox() {
      const box = activeBox.value;
      if (!box?.norm_bbox) return;
      nextTick(() => {
        setTimeout(() => {
          const vp = pdfViewportRef.value;
          const stage = vp ? vp.querySelector(".pdf-stage-wrapper") : null;
          if (!vp || !stage || activeBox.value !== box) return;
          const [x0, y0, x1, y1] = box.norm_bbox;
          const cx = (x0 + x1) / 2;
          const cy = (y0 + y1) / 2;
          const targetLeft = stage.offsetLeft + cx * stage.clientWidth - vp.clientWidth / 2;
          const targetTop = stage.offsetTop + cy * stage.clientHeight - vp.clientHeight / 2;
          vp.scrollTo({
            left: Math.max(0, targetLeft),
            top: Math.max(0, targetTop),
            behavior: "smooth",
          });
        }, 30);
      });
    }

    const focusOnActiveBox = panCenterToActiveBox;

    // Draggable Left-Right Split Resizer
    function startSplitResize(e) {
      e.preventDefault();
      isResizingSplit.value = true;
      const onMouseMove = (moveEvent) => {
        const pct = (moveEvent.clientX / window.innerWidth) * 100;
        leftPaneWidth.value = Math.max(28, Math.min(78, Math.round(pct)));
      };
      const onMouseUp = () => {
        isResizingSplit.value = false;
        saveWorkspacePreferences({paneWidth:leftPaneWidth.value});
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      };
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    }

    async function openYoloModal() {
      const res = await fetch("/api/yolo-dataset");
      if (res.ok) {
        yoloStatus.value = await res.json();
      }
      showYoloModal.value = true;
    }

    async function archiveCurrentProjectToYolo() {
      if (!currentProjectId.value) return;
      const res = await fetch(`/api/projects/${currentProjectId.value}/yolo-archive`, {
        method: "POST",
      });
      if (res.ok) {
        yoloStatus.value = await res.json();
        showToast(
          `🧬 已将当前项目沉淀至 YOLOv11 训练集（累计 ${yoloStatus.value.image_count} 页 / ${yoloStatus.value.total_bounding_boxes} 个标注框）！`
        );
      }
    }

    const isPartListMode = computed(() => {
      return projectData.value && projectData.value.extraction_mode === "PART_LIST_MODE";
    });

    const primaryBomList = computed(() => {
      if (!projectData.value) return [];
      return isPartListMode.value
        ? projectData.value.itemized_bom || []
        : projectData.value.aggregated_bom || [];
    });

    const secondaryBomList = computed(() => {
      if (!projectData.value) return [];
      return isPartListMode.value
        ? projectData.value.aggregated_bom || []
        : projectData.value.itemized_bom || [];
    });

    const baseBomList = computed(() => {
      return activeTab.value === "secondary" ? secondaryBomList.value : primaryBomList.value;
    });

    const detailTab = computed(() => isPartListMode.value ? 'primary' : 'secondary');
    const pageOrderActive = computed(() => activeTab.value === detailTab.value && followDrawingPage.value);
    const circuitPageOrder = computed(() => ['MAIN_CIRCUIT','CONTROL_CIRCUIT','PLC_WIRING','INSTRUMENT_WIRING','TERMINAL_DIAGRAM'].includes(currentPageInfo.value?.page_type) || !currentPageInfo.value?.page_type);
    const useCircuitOrder = computed(() => pageOrderMode.value === 'circuit' && circuitPageOrder.value);
    const pagePositions = computed(() => {
      const page = currentPage.value;
      const candidatesByItem = new Map();
      for (const candidate of detectedSomCandidates.value) {
        const box = hasDrawingBox(candidate.anchor_norm_bbox) ? candidate.anchor_norm_bbox : candidate.norm_bbox;
        if (candidate.excluded || !candidate.matched_item_id || !hasDrawingBox(box)) continue;
        const id = candidate.matched_item_id;
        if (!candidatesByItem.has(id)) candidatesByItem.set(id, []);
        candidatesByItem.get(id).push({page, norm_bbox:box, raw_text:candidate.display_label || candidate.tag || '', device_tag:candidate.tag});
      }
      const points = [];
      const supportingRoles = new Set(['model_text','model_inference','model_candidate','device_description','device_annotation','specification','schematic_grid','control_wire_estimate','cross_reference']);
      for (const item of projectData.value?.itemized_bom || []) {
        const anchors = [];
        const hasTerminalSymbols = item.terminal_strip_tag && (item.evidences || []).some(e => e.page === page && e.role === 'terminal_symbol');
        for (const evidence of item.evidences || []) {
          if (evidence.page !== page || !hasDrawingBox(evidence.norm_bbox) || supportingRoles.has(evidence.role)) continue;
          if (hasTerminalSymbols && evidence.role !== 'terminal_symbol') continue;
          anchors.push(evidence);
        }
        for (const candidate of candidatesByItem.get(item.id) || []) {
          if (!anchors.some(anchor => somOverlap(anchor.norm_bbox,candidate.norm_bbox))) anchors.push(candidate);
        }
        for (const location of anchors) points.push({item, location, x:location.norm_bbox[0], y:location.norm_bbox[1]});
      }
      const column = x => pageGridColumns.value.length >= 8
        ? pageGridColumns.value.reduce((best,entry,index,columns) =>
            Math.abs(entry.x-x)<Math.abs(columns[best].x-x) ? index : best,0)
        : Math.max(0, Math.min(9, Math.floor(x * 10)));
      points.sort((a,b) => useCircuitOrder.value
        ? column(a.x)-column(b.x) || a.y-b.y || a.x-b.x || a.item.seq_no-b.item.seq_no
        : a.y-b.y || a.x-b.x || a.item.seq_no-b.item.seq_no);
      const positions = new Map();
      const append = ({item,location}) => {
        if (!positions.has(item.id)) positions.set(item.id,{item, number:positions.size+1, locations:[]});
        positions.get(item.id).locations.push(location);
      };
      if (useCircuitOrder.value) {
        points.forEach(append);
      } else {
        for (let start=0; start<points.length;) {
          let end=start+1;
          while (end<points.length && points[end].y-points[start].y<=0.006) end++;
          points.slice(start,end).sort((a,b) => a.x-b.x || a.y-b.y || a.item.seq_no-b.item.seq_no).forEach(append);
          start=end;
        }
      }
      const groups=new Map();
      for (const [id,entry] of positions) {
        const strip=entry.item.entity_role==='terminal' && entry.item.terminal_strip_tag
          ? somTag(entry.item.terminal_strip_tag) : '';
        const key=strip ? 'terminal:'+strip : 'item:'+id;
        if (!groups.has(key)) groups.set(key,[]);
        groups.get(key).push([id,entry]);
      }
      const firstTerminalNumber=item=>{
        const numbers=(item.terminal_numbers || []).map(Number).filter(Number.isFinite);
        if (numbers.length) return Math.min(...numbers);
        const printed=String(item.tag || '').match(/:(\d+)/);
        return printed ? Number(printed[1]) : Infinity;
      };
      const groupedPositions=new Map();
      for (const entries of groups.values()) {
        if (entries.length>1 && entries[0][1].item.terminal_strip_tag) {
          entries.sort((a,b)=>firstTerminalNumber(a[1].item)-firstTerminalNumber(b[1].item) || a[1].number-b[1].number);
        }
        for (const [id,entry] of entries) groupedPositions.set(id,{...entry,
          locations:entry.item.terminal_strip_tag ? orderedTerminalLocations(entry.locations) : entry.locations,
          number:groupedPositions.size+1});
      }
      return groupedPositions;
    });
    const drawingAnchorRoles = new Set(['main_circuit','coil','device','instrument_module',
      'contact_or_symbol','terminal_symbol','terminal_position','main']);
    function itemDrawingAnchor(item) {
      const pageLocation=pageOrderActive.value ? pagePositions.value.get(item.id)?.locations?.[0] : null;
      if (pageLocation && hasDrawingBox(pageLocation.norm_bbox)) return pageLocation;
      const anchors=(item.evidences || []).filter(ev => drawingAnchorRoles.has(ev.role) && hasDrawingBox(ev.norm_bbox));
      return anchors.find(ev => ev.page === item.primary_page) || anchors[0] || null;
    }
    function itemDrawingReference(item) {
      const anchor=itemDrawingAnchor(item);
      const page=Number(anchor?.page || item.primary_page) || 1;
      const info=(projectData.value?.pages || []).find(entry => entry.page_num === page);
      const sheet=String(info?.sheet_no || '').trim();
      const differentSheet=/^\d+$/.test(sheet) && Number(sheet)!==page;
      const grid=gridColumnsForPage(page).filter(entry => Number.isInteger(entry.number) &&
        Number.isFinite(entry.x) && entry.x > 0 && entry.x < 1).sort((a,b) => a.x-b.x);
      let column=null;
      if (anchor && grid.length>=8) {
        const box=anchor.norm_bbox;
        const spacing=(grid[grid.length-1].x-grid[0].x)/(grid.length-1);
        const x=(box[0]+box[2])/2;
        if (spacing>0 && box[2]-box[0]<=spacing*1.5 &&
            x>=grid[0].x-spacing/2 && x<=grid[grid.length-1].x+spacing/2) {
          column=grid.reduce((best,entry) => Math.abs(entry.x-x)<Math.abs(best.x-x) ? entry : best).number;
        }
      }
      const position=column===null ? '' : `.${column}`;
      const label=differentSheet ? `图${sheet}${position} · P${page}` : `P${page}${position}`;
      const title=column===null ? `PDF 第 ${page} 页${differentSheet ? ` · 图纸第 ${sheet} 页` : ''}；分栏位置未确认`
        : `${differentSheet ? `图纸第 ${sheet} 页 · 第 ${column} 列 · ` : ''}PDF 第 ${page} 页${differentSheet ? '' : ` · 第 ${column} 列`}；点击定位原图证据`;
      return {page,column,anchor,label,title};
    }
    const tableScopeItems = computed(() => pageOrderActive.value ? Array.from(pagePositions.value.values(),entry=>entry.item) : baseBomList.value);

    const tableCountLabel = computed(() => {
      const total = baseBomList.value.length, scope = tableScopeItems.value.length, count = currentTableItems.value.length;
      if (pageOrderActive.value) return filtersActive.value
        ? `本页匹配 ${count} / ${scope} 项 · 全项目 ${total} 项`
        : `本页 ${scope} 项 · 全项目 ${total} 项`;
      return filtersActive.value ? `全项目匹配 ${count} / ${total} 项` : `全项目 ${total} 项`;
    });

    const isAuxItem = (it) => {
      return (
        it.category === "auxiliary" ||
        ["导线", "端子", "仿威图电柜", "人工"].includes((it.name || "").trim())
      );
    };

    const primaryCatCount = computed(() => tableScopeItems.value.filter((it) => !isAuxItem(it)).length);
    const auxCatCount = computed(() => tableScopeItems.value.filter((it) => isAuxItem(it)).length);

    function needsModel(item) {
      return !item.model || /^(未提取|图纸未注|待定)/.test(item.model);
    }
    function modelPendingLabel(item) {
      const state = item.readiness?.model_state;
      if (state && state.code !== 'KNOWN') return state.label;
      return (item.evidences || []).some(e => ['model_text','model_candidate'].includes(e.role)) && needsModel(item)
        ? '型号关联待核' : '待选型';
    }
    function hasPendingComponents(item) {
      return (item.component_parts || []).some(p => p.kind !== 'main' && !p.materialized_item_id &&
        (p.status !== 'CONFIRMED' || p.relationship !== 'integral'));
    }
    function componentLabel(kind) {
      return {main:'本体',socket:'底座',contact_block:'触点模块',lamp_module:'灯模块',accessory:'附件',unknown:'部件待核'}[kind] || '部件';
    }
    async function materializeComponent(item, part) {
      if (!displayedProjectIsCurrent() || editDirty.value || savingEdit.value) return;
      savingEdit.value = true;
      try {
        const response = await fetch(`/api/projects/${currentProjectId.value}/bom/items/${encodeURIComponent(item.id)}/components/${encodeURIComponent(part.id)}/materialize`, {method:'POST'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '部件未补入');
        if (!applyProjectResponse(data)) return;
        savingEdit.value = false;
        refreshPanelAfterAction();
        showToast('已补入独立采购部件，数量留空待核定。');
      } catch (error) { showToast(error.message || '部件未补入，请重试'); }
      finally { savingEdit.value = false; }
    }
    function componentEvidence(item, evidence) {
      return locateItemEvidence(item, {...evidence, bbox:evidence.bbox || []});
    }
    function matchesIssue(item, issue) {
      if (issue === 'MODEL') return needsModel(item) && modelPendingLabel(item) === '待选型';
      if (issue === 'ASSOCIATION') return needsModel(item) && modelPendingLabel(item) === '型号关联待核';
      if (issue === 'COMPONENTS') return hasPendingComponents(item);
      if (issue === 'QUANTITY') return item.quantity == null || ['unknown','needs_confirmation'].includes(item.quantity_status);
      if (issue === 'SCOPE') return !item.procurement_scope || item.procurement_scope === 'unknown';
      if (issue === 'REVIEW') return item.needs_review || item.readiness?.status === 'NEEDS_REVIEW';
      return true;
    }
    const issueLabels = {ALL:'全部物料', MODEL:'待选型', ASSOCIATION:'型号关联待核', COMPONENTS:'本体/附件待核', QUANTITY:'数量待核', SCOPE:'范围待核', REVIEW:'待审核'};
    const issueCounts = computed(() => {
      const rows = primaryBomList.value;
      return {ALL:rows.length, ...Object.fromEntries(['MODEL','ASSOCIATION','COMPONENTS','QUANTITY','SCOPE','REVIEW'].map(key => [key, rows.filter(item => matchesIssue(item,key)).length]))};
    });
    function setWorkspaceMode(mode) {
      workspaceMode.value = mode;
      saveWorkspacePreferences({mode});
    }
    function toggleOverview() {
      overviewExpanded.value = !overviewExpanded.value;
      saveWorkspacePreferences({overviewExpanded:overviewExpanded.value});
    }
    function focusIssue(issue) {
      activeTab.value = 'primary';
      issueFilter.value = issueFilter.value === issue ? 'ALL' : issue;
      searchQuery.value = ''; sourceFilter.value = 'ALL'; categoryFilter.value = 'ALL';
      if (workspaceMode.value === 'drawing') setWorkspaceMode('compare');
    }
    function resetFilters() {
      searchQuery.value = ''; sourceFilter.value = 'ALL'; categoryFilter.value = 'ALL'; issueFilter.value = 'ALL';
    }
    const filtersActive = computed(() => Boolean(searchQuery.value || sourceFilter.value !== 'ALL' || categoryFilter.value !== 'ALL' || issueFilter.value !== 'ALL'));
    function itemIssues(item) {
      const issues = [];
      if (matchesIssue(item,'MODEL')) issues.push('待选型');
      if (matchesIssue(item,'ASSOCIATION')) issues.push('型号关联待核');
      if (hasPendingComponents(item)) issues.push('本体/附件待核');
      if (matchesIssue(item,'QUANTITY')) issues.push('数量待核');
      if (matchesIssue(item,'SCOPE')) issues.push('范围待核');
      if (item.parent_tag || ['assembly','contact'].includes(item.entity_role)) issues.push('组件待核');
      if (!issues.length && matchesIssue(item,'REVIEW')) issues.push('待审核');
      return issues;
    }
    function scopeLabel(value) { return {cabinet:'柜内采购',field:'现场采购',unknown:'范围待核',excluded:'不在采购范围'}[value] || '范围待核'; }
    function roleLabel(value) { return {device:'独立器件',coil:'线圈 / 器件',contact:'仅触点',terminal:'端子材料',cable:'电缆材料',assembly:'成套 / 组件待核',accessory:'附件'}[value] || '独立器件'; }
    function evidencePages(item) { return [...new Set((item.evidences || []).map(e => e.page))]; }
    const drawingAnnotationRoles={main_circuit:'位号',device_description:'器件说明',device_annotation:'功能 / 接口说明',model_text:'原图型号',model_candidate:'型号线索',specification:'技术参数'};
    function drawingAnnotations(item) {
      if (!item) return [];
      const seen=new Set();
      return (item.evidences || []).filter(ev=>{
        if (!drawingAnnotationRoles[ev.role] || !String(ev.raw_text || '').trim() || !hasDrawingBox(ev.norm_bbox)) return false;
        if (ev.role==='main_circuit' && somTag(ev.raw_text)!==somTag(item.tag)) return false;
        const key=[ev.page,ev.raw_text.trim(),ev.norm_bbox.join(',')].join('|');
        if (seen.has(key)) return false;
        seen.add(key);return true;
      }).sort((a,b)=>a.page-b.page || a.norm_bbox[1]-b.norm_bbox[1] || a.norm_bbox[0]-b.norm_bbox[0])
        .map(ev=>({...ev,annotation_label:drawingAnnotationRoles[ev.role]}));
    }
    function drawingAnnotationSummary(item) {
      if ((item.member_ids || []).length>1) return '';
      const notes=drawingAnnotations(item).filter(ev=>['device_description','device_annotation'].includes(ev.role))
        .map(ev=>ev.raw_text.trim()).filter(Boolean);
      return [...new Set(notes)].join('；');
    }
    function hasDistinctSpecification(item) {
      const normalize=value=>String(value || '').normalize('NFKC').replace(/\s+/g,'').toUpperCase();
      return Boolean(item.specification && normalize(item.specification)!==normalize(item.model));
    }
    const rawReviews = computed(() => projectData.value?.review_items || []);
    const reviewPages = computed(() => new Map((projectData.value?.pages || []).map(p=>[p.page_num,p])));
    const reviewGroups = computed(() => {
      const cards=[], groups=new Map();
      const substantive=new Set(rawReviews.value.filter(r=>r.status==='PENDING' && r.issue_type!=='AI_RECOGNIZED'
          && (r.tag || r.proposed_data?.tag)).map(r=>r.tag || r.proposed_data.tag));
      for (const r of rawReviews.value) {
        if (r.status!=='PENDING' || (r.severity==='INFO' && (!r.action_type || r.action_type==='CONFIRM'))) continue;
        if (r.issue_type==='AI_RECOGNIZED' && r.tag && substantive.has(r.tag) && !r.proposed_data) continue;
        if (isPageCheck(r)) continue;
        const groupable=['UNBOUND_MODEL','AI_EVIDENCE_CHECK','AI_REGION_REVIEW','AI_RECOGNIZED','UNRESOLVED_MODEL','TEMPLATE_QUANTITY','QUANTITY_UNCERTAIN'];
        if (!groupable.includes(r.issue_type)) {
          cards.push(r);continue;
        }
        const key=r.page+'-'+r.issue_type;
        if (!groups.has(key)) groups.set(key,[]);
        groups.get(key).push(r);
      }
      const names={UNBOUND_MODEL:'核对本页可能漏列的物料',AI_EVIDENCE_CHECK:'复查本页 AI 识别证据',AI_REGION_REVIEW:'复查未定位的设备',
        AI_RECOGNIZED:'核实 AI 识别的设备',UNRESOLVED_MODEL:'补充设备型号',TEMPLATE_QUANTITY:'核实共用图的采购套数',QUANTITY_UNCERTAIN:'核对实际采购数量'};
      for (const [key,members] of groups) {
        const first=members[0];
        if (members.length===1) {cards.push(first);continue;}
        cards.push({...first,id:'group-'+key,tag:'',norm_bbox:null,members,
          title:names[first.issue_type]+' · 第 '+first.page+' 页',
          description:members.length+' 条相关记录',suggestion:'展开查看涉及的位号与图纸位置。'});
      }
      return cards.sort((a,b)=>(b.severity==='CONFLICT')-(a.severity==='CONFLICT') || a.page-b.page);
    });
    function isPageCheck(r) {
      return ['UNBOUND_MODEL','PAGE_COVERAGE'].includes(r.issue_type) ||
        (['AI_EVIDENCE_CHECK','AI_REGION_REVIEW'].includes(r.issue_type) && !r.tag && !r.proposed_data);
    }
    const coverageReviews=computed(()=>{
      const byPage=new Map();
      for (const r of rawReviews.value) {
        if (r.status!=='PENDING' || !isPageCheck(r)) continue;
        if (!byPage.has(r.page)) byPage.set(r.page,[]);
        byPage.get(r.page).push(r);
      }
      return [...byPage.entries()].sort((a,b)=>a[0]-b[0]).map(([page,members])=>members.length===1 ? members[0] :
        {...members[0],id:'coverage-'+page,tag:'',norm_bbox:null,title:'核查第 '+page+' 页的设备与原文',
         description:members.length+' 条证据',suggestion:'展开证据定位到图纸，检查是否遗漏设备或误读标注。',members});
    });
    const awaitingFirstBom = computed(() => aiBusy.value && !(projectData.value?.itemized_bom || []).length);
    const pagesToCheck = computed(() => {
      const pages=new Set(coverageReviews.value.map(r=>r.page));
      for (const error of aiJob.value?.errors || []) if (error.stage==='extract') pages.add(error.page);
      return [...pages].sort((a,b)=>a-b);
    });
    const visibleReviews = computed(() => reviewStatusFilter.value==='COVERAGE' ? coverageReviews.value : reviewStatusFilter.value==='PENDING'
      ? reviewGroups.value
      : rawReviews.value.filter(r=>reviewStatusFilter.value==='ALL' || r.status===reviewStatusFilter.value));
    function unboundLocations(rev) {
      return rev.issue_type==='UNBOUND_MODEL' && !rev.members
        ? reviewPages.value.get(rev.page)?.extraction_diagnostics?.unbound_evidence || [] : [];
    }
    const drawerItem = computed(() => {
      if (editingId.value && editItemSnapshot.value) return editItemSnapshot.value;
      const id = editingId.value || detailItemId.value;
      return [...(projectData.value?.itemized_bom || []), ...(projectData.value?.aggregated_bom || []), ...(projectData.value?.filtered_items || [])].find(item => item.id === id) || null;
    });
    const fieldClassification = ref(null);
    const fieldClassificationBusy = ref(false);
    const fieldClassificationError = ref('');
    const fieldHistory = ref({item_id:'', records:[]});
    const fieldHistoryBusy = ref(false);
    const fieldHistoryError = ref('');
    let fieldHistoryVersion = 0;
    const fieldPreview = ref(null);
    const fieldPreviewBusy = ref(false);
    const qwenKeyValue = ref('');
    const qwenHasKey = ref(false);
    const qwenKeySource = ref('none');
    const qwenKeySaving = ref(false);
    const qwenTestBusy = ref(false);
    const qwenTestResult = ref(null);
    async function openQwenKeyEditor() {
      await openVlmModal('item');
    }
    async function saveQwenKey() {
      if (!qwenKeyValue.value.trim() || qwenKeySaving.value) return;
      qwenKeySaving.value=true;
      try {
        const response=await fetch('/api/qwen-config',{method:'PUT',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({api_key:qwenKeyValue.value.trim()})});
        const data=await response.json();
        if (!response.ok) throw new Error(data.detail || '百炼 Key 保存失败');
        qwenHasKey.value=data.has_api_key;qwenKeySource.value=data.key_source || 'saved';qwenKeyValue.value='';aiConfigError.value='';
        qwenTestResult.value=null;
        showToast('百炼 Key 已保存在当前计算机的加密凭据中');
      } catch (error) {aiConfigError.value=error.message || '百炼 Key 保存失败';}
      finally {qwenKeySaving.value=false;}
    }
    async function testQwenConnection() {
      if (qwenTestBusy.value || (!qwenHasKey.value && !qwenKeyValue.value.trim())) return;
      qwenTestBusy.value=true;qwenTestResult.value=null;aiConfigError.value='';
      const testingInput=Boolean(qwenKeyValue.value.trim());
      try {
        const response=await fetch('/api/qwen-config/test',{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({api_key:testingInput ? qwenKeyValue.value.trim() : null})});
        const data=await response.json();
        if (!response.ok) throw new Error(data.detail || '连通性测试失败');
        qwenTestResult.value={model:data.model || 'qwen3-vl-plus',source:testingInput ? '当前输入' : '已保存'};
      } catch (error) {aiConfigError.value=error.message || '连通性测试失败';}
      finally {qwenTestBusy.value=false;}
    }
    function fieldClassificationTarget(item) {
      if (!item || (item.member_ids || []).length > 1) return null;
      const id = item.member_ids?.[0] || item.id;
      return [...(projectData.value?.itemized_bom || []), ...(projectData.value?.filtered_items || [])].find(row => row.id === id) || null;
    }
    function formatFieldRecordTime(value) {
      const date=new Date(value);
      return Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleString('zh-CN');
    }
    async function loadFieldHistory(item) {
      const target=fieldClassificationTarget(item);
      if (!target || !displayedProjectIsCurrent()) return;
      const projectId=currentProjectId.value;
      const version=++fieldHistoryVersion;
      fieldHistoryBusy.value=true;fieldHistoryError.value='';
      try {
        const response=await fetch(`/api/projects/${projectId}/bom/items/${encodeURIComponent(target.id)}/classify-text/history`);
        const data=await response.json();
        if (!response.ok) throw new Error(data.detail || '判读记录加载失败');
        if (version!==fieldHistoryVersion || projectId!==currentProjectId.value ||
            fieldClassificationTarget(drawerItem.value)?.id!==target.id) return;
        fieldHistory.value=data;
        if (fieldClassification.value?.item_id!==target.id) {
          const latest=data.records?.[0];
          fieldClassification.value=latest ? {...latest,historical:true,item_id:target.id} : null;
        }
      } catch (error) {
        if (version===fieldHistoryVersion) fieldHistoryError.value=error.message || '判读记录加载失败';
      } finally {
        if (version===fieldHistoryVersion) fieldHistoryBusy.value=false;
      }
    }
    function showFieldRecord(record) {
      fieldClassification.value={...record,historical:true,item_id:record.item_id};
    }
    async function previewDrawingText(item) {
      const target=fieldClassificationTarget(item);
      if (!target || !displayedProjectIsCurrent() || fieldPreviewBusy.value) return;
      const projectId=currentProjectId.value;
      fieldClassificationError.value='';fieldPreviewBusy.value=true;
      try {
        const response=await fetch(`/api/projects/${projectId}/bom/items/${encodeURIComponent(target.id)}/classify-text/preview`);
        const data=await response.json();
        if (!response.ok) throw new Error(data.detail || '裁图预览失败');
        if (projectId===currentProjectId.value) fieldPreview.value={...data,item_id:target.id};
      } catch (error) {fieldClassificationError.value=error.message || '裁图预览失败';}
      finally {fieldPreviewBusy.value=false;}
    }
    async function classifyDrawingText(item) {
      const target=fieldClassificationTarget(item);
      if (!target || !displayedProjectIsCurrent() || fieldClassificationBusy.value) return;
      const projectId=currentProjectId.value;
      fieldClassification.value=null;
      fieldClassificationError.value='';
      fieldClassificationBusy.value=true;
      try {
        const response=await fetch(`/api/projects/${projectId}/bom/items/${encodeURIComponent(target.id)}/classify-text`,{method:'POST'});
        const data=await response.json();
        if (!response.ok) throw new Error(data.detail || '千问判读失败');
        if (projectId!==currentProjectId.value) return;
        fieldClassification.value={...data,item_id:target.id};
        const updated=await fetch(`/api/projects/${projectId}`);
        if (updated.ok) applyProjectResponse(await updated.json());
        await loadFieldHistory(target);
        showToast(data.cache_hit ? '已读取该位号的千问判读缓存' : '千问已完成文字归属判读，请核对原文后采纳');
      } catch (error) {fieldClassificationError.value=error.message || '千问判读失败';}
      finally {fieldClassificationBusy.value=false;}
    }
    async function acceptFieldClassification() {
      if (editDirty.value || savingEdit.value) return;
      const result=fieldClassification.value;
      if (!result?.review_id || result.historical || result.review_status!=='PENDING' ||
          !(result.analysis?.can_propose_update || result.analysis?.can_propose_brand || result.analysis?.can_propose_components) || !displayedProjectIsCurrent()) return;
      const response=await fetch(`/api/projects/${currentProjectId.value}/reviews/${result.review_id}/action`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'ACCEPT'})});
      const data=await response.json();
      if (!response.ok) {fieldClassificationError.value=data.detail || '采纳失败，请重新核对';return;}
      if (!applyProjectResponse(data)) return;
      refreshPanelAfterAction();
      fieldClassification.value={...result,review_status:'ACCEPTED'};
      fieldHistory.value={...fieldHistory.value,records:(fieldHistory.value.records || []).map(record =>
        record.review_id===result.review_id ? {...record,review_status:'ACCEPTED'} : record)};
      fieldClassificationError.value='';
      showToast(result.analysis.can_propose_update ? '已采纳千问判读，型号和原图文字证据已更新' : '已确认图形品牌并写入物料，型号仍待核实');
    }
    const editDirty = computed(() => editingId.value && JSON.stringify(editForm.value) !== editInitialValues.value);
    const editTargetChanged = computed(() => Boolean(editingId.value && ![...(projectData.value?.itemized_bom || []), ...(projectData.value?.aggregated_bom || [])].some(item => item.id === editingId.value)));
    function focusItemPanel() {
      if (!document.activeElement?.closest('.item-panel')) panelReturnFocus = document.activeElement;
      nextTick(() => document.querySelector('.item-panel input, .item-panel [data-panel-heading]')?.focus());
    }
    function openItemDetails(item) { startEdit(item); }
    function openItemIssue(item, issue) {
      const fields = {'待选型':'model', '型号关联待核':'model', '本体/附件待核':'components', '数量待核':'quantity', '范围待核':'procurement_scope', '组件待核':'parent_tag'};
      if (startEdit(item)) nextTick(() => document.getElementById('bom-editor-' + (fields[issue] || 'name'))?.focus());
    }
    function refreshPanelAfterAction() {
      const id = editingId.value;
      const item = [...(projectData.value?.itemized_bom || []), ...(projectData.value?.aggregated_bom || [])].find(row => row.id === id);
      if (item && !editDirty.value) startEdit(item);
    }
    function closeItemPanel() {
      if (savingEdit.value) return false;
      if (editDirty.value && !confirm('还有未保存的修改，确定放弃？')) return false;
      editingId.value = ''; detailItemId.value = ''; editForm.value = {}; editItemSnapshot.value = null;
      nextTick(() => { if (panelReturnFocus?.isConnected) panelReturnFocus.focus(); });
      return true;
    }
    async function locateItem(item) {
      if (workspaceMode.value === 'bom') { setWorkspaceMode('compare'); await nextTick(); }
      await zoomToBomRow(item);
    }
    async function locateItemEvidence(item, evidence) {
      const tag = item.tag;
      if (!closeItemPanel()) return;
      if (workspaceMode.value === 'bom') { setWorkspaceMode('compare'); await nextTick(); }
      jumpToBomEvidence(item, evidence, tag);
    }
    async function deletePanelItem(item) {
      if (editDirty.value || savingEdit.value) return;
      if (await deleteBomRow(item)) closeItemPanel();
    }
    function closeToolsMenu(event) { event.currentTarget.closest('details')?.removeAttribute('open'); }

    const currentTableItems = computed(() => {
      const base = tableScopeItems.value;
      const q = searchQuery.value.trim().toLowerCase();
      return base.filter((it) => {
        if (!matchesIssue(it, issueFilter.value)) return false;
        if (categoryFilter.value === "PRIMARY" && isAuxItem(it)) return false;
        if (categoryFilter.value === "AUX" && !isAuxItem(it)) return false;
        if (sourceFilter.value !== "ALL" && it.source_type !== sourceFilter.value) {
          return false;
        }
        if (!q) return true;
        return (
          (it.tag || "").toLowerCase().includes(q) ||
          (it.name || "").toLowerCase().includes(q) ||
          (it.model || "").toLowerCase().includes(q) ||
          (it.brand || "").toLowerCase().includes(q) ||
          (it.remark || "").toLowerCase().includes(q) ||
          (it.specification || '').toLowerCase().includes(q) ||
          drawingAnnotations(it).some(ev=>ev.raw_text.toLowerCase().includes(q)) ||
          (it.parent_tag || '').toLowerCase().includes(q) ||
          String(pageOrderActive.value ? pagePositions.value.get(it.id).number : it.seq_no).includes(q)
        );
      });
    });

    const pendingReviewCount = computed(() => {
      if (!projectData.value || !projectData.value.review_items) return 0;
      return reviewGroups.value.length;
    });

    function sourceLabel(src) {
      if (isTableExtraction.value && src === 'EXPLICIT') return '部件表原文';
      return SOURCE_LABELS[src] || src;
    }

    // Deduplicate evidences by tag (prefer coil/main_circuit over contact_or_symbol so 30 -KA relays show 30 primary items instead of 85 contacts)
    const uniqueTagEvidences = computed(() => {
      const list = selectedEvidences.value || [];
      if (list.length <= 1) return list;
      if (selectedItemObj.value?.terminal_strip_tag) {
        const symbols = list.filter(ev => ev.role === 'terminal_symbol');
        const positions = symbols.length ? symbols : list.filter(ev => ev.role === 'terminal_position');
        return symbols.length ? orderedTerminalLocations(symbols) : [...positions].sort((a, b) => a.page - b.page || a.norm_bbox[1] - b.norm_bbox[1] || a.norm_bbox[0] - b.norm_bbox[0]);
      }
      const byTag = new Map();
      for (const ev of list) {
        const key = (ev.device_tag || ev.raw_text || "").trim().toUpperCase() || `${ev.page}_${(ev.norm_bbox || []).join(",")}`;
        if (!byTag.has(key)) {
          byTag.set(key, ev);
        } else {
          const prev = byTag.get(key);
          // Prefer coil or main_circuit or instrument_module over auxiliary contact
          if (ev.role === "coil" || ev.role === "main_circuit" || ev.role === "instrument_module") {
            byTag.set(key, ev);
          } else if (prev.role === "contact_or_symbol" && ev.page > prev.page) {
            byTag.set(key, ev);
          }
        }
      }
      return Array.from(byTag.values());
    });

    const displayedEvidences = computed(() => {
      if (!selectedEvidences.value || selectedEvidences.value.length === 0) return [];
      return evDedupeByTag.value && !pageOrderActive.value ? uniqueTagEvidences.value : selectedEvidences.value;
    });
    const evidenceChipLabels = computed(() => {
      const list = displayedEvidences.value;
      const strip = somTag(selectedItemObj.value?.terminal_strip_tag);
      const labels = list.map(ev => `P${ev.page}:${String(ev.raw_text || '').slice(0, 12)}`);
      if (!strip) return labels;
      const byNumber = new Map();
      list.forEach((ev, index) => {
        const number = String(ev.raw_text || '').trim();
        if (!['terminal_symbol', 'terminal_position'].includes(ev.role) || !/^\d+$/.test(number)) return;
        labels[index] = `${strip}:${number}`;
        const key = `${ev.page}:${number}`;
        if (!byNumber.has(key)) byNumber.set(key, []);
        byNumber.get(key).push(index);
      });
      for (const indexes of byNumber.values()) {
        const positions = [];
        for (const index of indexes) {
          const box = list[index].norm_bbox;
          if (!hasDrawingBox(box)) continue;
          const x = (box[0] + box[2]) / 2, y = (box[1] + box[3]) / 2;
          let position = positions.find(p => Math.abs(p.x - x) < .01 && Math.abs(p.y - y) < .01);
          if (!position) { position = {x, y, indexes:[]}; positions.push(position); }
          position.indexes.push(index);
        }
        if (positions.length < 2) continue;
        const xRange = Math.max(...positions.map(p => p.x)) - Math.min(...positions.map(p => p.x));
        const yRange = Math.max(...positions.map(p => p.y)) - Math.min(...positions.map(p => p.y));
        const vertical = yRange >= xRange;
        positions.sort((a, b) => vertical ? a.y - b.y || a.x - b.x : a.x - b.x || a.y - b.y);
        positions.forEach((position, order) => {
          const suffix = positions.length === 2 ? (vertical ? ['上', '下'][order] : ['左', '右'][order]) : `第${order + 1}处`;
          position.indexes.forEach(index => { labels[index] += `（${suffix}）`; });
        });
      }
      return labels;
    });

    async function jumpToLocation(page, normBbox, label, itemIds = []) {
      selectedSomId.value='';
      selectedDrawingHit.value=null;
      activeGridColumn.value=null;
      const pageChanged = currentPage.value !== page;
      currentPage.value = page;
      activeBox.value = hasDrawingBox(normBbox) ? {
        project_id:currentProjectId.value,view_mode:currentImageView.value,
        page: page,
        norm_bbox: normBbox,
        itemIds,
        label: label || "",
      } : null;
      if (pageChanged && showSomLayer.value) {
        await loadPageTopology();
      }
      panCenterToActiveBox();
    }

    function focusDrawingColumn(number) {
      const band=pageGridBands.value.find(entry => entry.number === number);
      if (!band) return;
      activeGridColumn.value={page:currentPage.value,number};
      nextTick(() => {
        const viewport=pdfViewportRef.value;
        const stage=pdfStageRef.value;
        if (!viewport || !stage) return;
        viewport.scrollTo({left:Math.max(0,stage.offsetLeft + band.center*stage.clientWidth - viewport.clientWidth/2),
                           top:viewport.scrollTop});
      });
    }

    async function jumpToCrossReference(link) {
      showDrawingGrid.value=true;
      selectedItemId.value=link.item?.id || '';
      selectedEvidences.value=link.item?.evidences || [];
      const x=Number.isFinite(link.gridX) ? link.gridX : Math.max(.02,Math.min(.98,(link.column+.5)/10));
      const box=link.targetBox || [Math.max(0,x-.008),.08,Math.min(1,x+.008),.12];
      await jumpToLocation(link.targetPage,box,`${link.item?.tag || '页栏引用'} · 图纸 ${link.sheet}.${link.column}`,
        link.item ? [link.item.id] : []);
      activeGridColumn.value={page:link.targetPage,number:link.column};
    }

    function jumpToBomEvidence(item, evidence, label) {
      // Model and specification text can sit next to a symbol but have their own exact positions.
      const separateText = ['model_text','model_inference','model_candidate','device_description','device_annotation','specification','schematic_grid'].includes(evidence.role);
      const ids = item && !separateText ? [item.id, ...(item.member_ids || [])] : [];
      return jumpToLocation(evidence.page, evidence.norm_bbox, label, ids);
    }

    function selectBomRow(item) {
      if (selectedItemId.value === item.id) {
        clearActiveBox();
        return;
      }
      zoomToBomRow(item);
    }

    async function zoomToBomRow(item) {
      const reference=itemDrawingReference(item);
      selectedItemId.value = item.id;
      const evidences=pageOrderActive.value && pagePositions.value.has(item.id)
        ? pagePositions.value.get(item.id).locations : item.evidences || [];
      selectedEvidences.value=reference.anchor && evidences.includes(reference.anchor)
        ? [reference.anchor,...evidences.filter(ev=>ev!==reference.anchor)] : evidences;
      currentEvIndex.value = 0;
      evExpanded.value = false;
      const firstEv = reference.anchor || displayedEvidences.value[0];
      const label = `${firstEv ? firstEv.raw_text : (item.tag || "#" + item.seq_no)} | ${item.name} (${item.model || "无型号"})`;
      await jumpToBomEvidence(item, firstEv || {page:reference.page, norm_bbox:item.primary_bbox}, label);
      if (reference.column!==null && selectedItemId.value===item.id && currentPage.value===reference.page) {
        showDrawingGrid.value=true;
        activeGridColumn.value={page:reference.page,number:reference.column};
      }
    }

    function selectEvidence(ev, itemLabel) {
      jumpToBomEvidence(selectedItemObj.value, ev, `${itemLabel || ev.raw_text} [P${ev.page}]`);
    }

    function selectEvidenceByIndex(idx) {
      const list = displayedEvidences.value;
      if (!list || list.length === 0) return;
      const clamped = ((idx % list.length) + list.length) % list.length;
      currentEvIndex.value = clamped;
      const ev = list[clamped];
      jumpToBomEvidence(selectedItemObj.value, ev, `${evidenceChipLabels.value[clamped]} [第${ev.page}页 · ${clamped + 1}/${list.length}]`);
      nextTick(() => {
        const strip = document.querySelector(".ev-scroll-strip");
        const activeChip = strip ? strip.querySelector(".ev-chip.active") : null;
        if (activeChip && strip) {
          activeChip.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
        }
      });
    }

    function stepEvidence(delta) {
      selectEvidenceByIndex(currentEvIndex.value + delta);
    }

    function selectReviewCard(rev) {
      if (isProjectReview(rev)) { showCadImportInfo();return; }
      if (selectedItemId.value === rev.id) {
        clearActiveBox();
        return;
      }
      selectedItemId.value = rev.id;
      selectedEvidences.value = [];
      jumpToLocation(rev.page, rev.norm_bbox, `${rev.tag} - ${rev.issue_type}`);
    }

    function isProjectReview(rev) { return rev.scope==='project' || rev.issue_type==='CAD_IMPORT_REVIEW'; }
    function canLocateReview(rev) { return !isProjectReview(rev) && hasDrawingBox(rev.norm_bbox); }
    function showCadImportInfo() {
      clearActiveBox();overviewExpanded.value=true;
      nextTick(()=>document.querySelector('.cad-import-notice')?.setAttribute('open',''));
    }

    function startEdit(item) {
      if (savingEdit.value) return false;
      if (editDirty.value && !confirm('还有未保存的修改，确定放弃？')) return false;
      if (item.is_quantity_adjustment || activeTab.value === 'filtered') {
        editingId.value = ''; editForm.value = {}; editItemSnapshot.value = null;
        detailItemId.value = item.id;
        focusItemPanel(); loadFieldHistory(item);
        if (item.is_quantity_adjustment) showToast('数量调整记录用于追溯，请在采购汇总中修改总数量。');
        return false;
      }
      selectedItemId.value = item.id;
      editingId.value = item.id;
      editItemSnapshot.value = JSON.parse(JSON.stringify(item));
      detailItemId.value = '';
      editForm.value = {
        id: item.id,
        seq_no: item.seq_no,
        tag: item.tag || "",
        brand: item.brand || "",
        name: item.name || "",
        model: item.model || "",
        unit: item.unit || "个",
        quantity: item.quantity,
        remark: item.remark || "",
        placement: item.placement || "",
        pdf_page_label: item.pdf_page_label || "",
        specification: item.specification || "",
        procurement_scope: item.procurement_scope || "unknown",
        parent_tag: item.parent_tag || "",
        entity_role: item.entity_role || "device",
      };
      editInitialValues.value = JSON.stringify(editForm.value);
      focusItemPanel();
      loadFieldHistory(item);
      return true;
    }

    function cancelEdit() { closeItemPanel(); }

    function displayedProjectIsCurrent() {
      return Boolean(currentProjectId.value && projectData.value?.project_id === currentProjectId.value);
    }

    function applyProjectResponse(data) {
      if (!displayedProjectIsCurrent() || data.project_id !== currentProjectId.value) return false;
      projectData.value = data;
      return true;
    }

    async function applySelection(item, suggestion, action) {
      if (!displayedProjectIsCurrent() || editDirty.value || savingEdit.value) return;
      const id = item.member_ids?.length === 1 ? item.member_ids[0] : item.id;
      const response = await fetch(`/api/projects/${currentProjectId.value}/bom/item/${id}/selection`, {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({model:suggestion.model, action}),
      });
      const data = await response.json();
      if (response.ok) { if (!applyProjectResponse(data)) return; refreshPanelAfterAction(); showToast(action === "ACCEPT" ? "已记录人工选型，原图型号和证据已保留。" : "已忽略该选型建议。"); }
      else { showToast(data.detail || "选型处理失败"); }
    }

    async function saveEdit(item) {
      if (!displayedProjectIsCurrent()) return;
      if (!item || savingEdit.value || !editDirty.value) return;
      if (editTargetChanged.value) { showToast('后台结果已更新，修改尚未保存。请关闭侧栏后重新选择物料。'); return; }
      const viewMode =
        activeTab.value === "secondary"
          ? isPartListMode.value
            ? "aggregated"
            : "itemized"
          : "primary";

      const payload = {
        ...editForm.value,
        view_mode: viewMode,
        quantity: editForm.value.quantity === "" || editForm.value.quantity == null ? null : Number(editForm.value.quantity),
      };
      delete payload.seq_no; // Never fall back to a reused row number after a background merge.

      if (payload.quantity !== null && (!Number.isFinite(payload.quantity) || payload.quantity < 0)) {
        showToast('数量请填写非负数字；尚未核定时留空。'); return;
      }
      const initial = JSON.parse(editInitialValues.value);
      const fields = Object.keys(initial).filter(key => key !== 'id' && key !== 'seq_no');
      const snapshot = editItemSnapshot.value;
      payload.expected_values = Object.fromEntries(
        [...fields, 'tags', 'member_ids', 'quantity_status', 'quantity_basis'].filter(key => key in snapshot).map(key => [key, snapshot[key]])
      );
      if (editDirty.value) {
        for (const key of fields) {
          if (editForm.value[key] === initial[key]) delete payload[key];
        }
      }
      savingEdit.value = true;
      try {
        const res = await fetch(`/api/projects/${currentProjectId.value}/bom/item`, {
          method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) { showToast(res.status === 404 ? '物料已发生变化，修改尚未保存，请重新选择。' : data.detail || '修改未保存，请重试'); return; }
        if (!applyProjectResponse(data)) return;
        editingId.value = ''; detailItemId.value = ''; editForm.value = {}; editItemSnapshot.value = null;
        showToast(data.summary.export_warning || `已保存第 ${item.seq_no} 项，同步更新采购汇总。`);
        nextTick(() => { if (panelReturnFocus?.isConnected) panelReturnFocus.focus(); });
      } catch { showToast('连接失败，修改内容已保留，请重试。'); }
      finally { savingEdit.value = false; }
    }

    async function deleteBomRow(item) {
      if (!displayedProjectIsCurrent()) return;
      if (!confirm(`确定从 BOM 中删除序号 ${item.seq_no} [${item.name} ${item.model}] 吗？`)) {
        return;
      }
      const res = await fetch(`/api/projects/${currentProjectId.value}/bom/item/${item.id}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (!applyProjectResponse(await res.json())) return;
        showToast(`🗑️ 已删除物料 [${item.name}]`);
        return true;
      }
      return false;
    }

    const selectedItemObj = computed(() => {
      if (!selectedItemId.value || !projectData.value) return null;
      return (
        baseBomList.value.find((it) => it.id === selectedItemId.value) ||
        (projectData.value.aggregated_bom || []).find((it) => it.id === selectedItemId.value) ||
        (projectData.value.itemized_bom || []).find((it) => it.id === selectedItemId.value) ||
        (projectData.value.filtered_items || []).find((it) => it.id === selectedItemId.value) ||
        null
      );
    });
    const terminalHighlights = computed(() => {
      if (!selectedItemObj.value?.terminal_strip_tag || !activeBox.value || activeBox.value.page !== currentPage.value) return [];
      const seen = new Set();
      return (selectedItemObj.value.evidences || []).filter(ev => {
        if (ev.page !== currentPage.value || ev.role !== 'terminal_symbol' || !hasDrawingBox(ev.norm_bbox)) return false;
        const key = ev.norm_bbox.join(',');
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    });

    function applyRecognizedToken(tok) {
      const target = tokenFillTarget.value || "model";
      addForm.value[target] = tok;
      if (target === "model") {
        const up = tok.toUpperCase();
        if (["LC1", "EZD", "NSX", "CVS", "GV2", "LRD", "IC65", "C65", "XB2", "RXM", "REXL"].some((k) => up.includes(k))) {
          addForm.value.brand = "施耐德";
        } else if (["6ES7", "6EP", "3RT", "3RV", "5SU"].some((k) => up.includes(k))) {
          addForm.value.brand = "西门子";
        } else if (["HDR-", "NDR-", "EDR-", "LRS-"].some((k) => up.includes(k))) {
          addForm.value.brand = "明纬";
        } else if (up.includes("NHR-")) {
          addForm.value.brand = "虹润";
        }
      }
    }

    async function bindDrawnBoxToSelectedRow() {
      if (!displayedProjectIsCurrent()) return;
      if (!selectedItemObj.value || !currentProjectId.value) return;
      const bbox = addForm.value.primary_bbox || [0.15, 0.2, 0.35, 0.3];
      const res = await fetch(
        `/api/projects/${currentProjectId.value}/bom/item/${selectedItemObj.value.id}/bind-box`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            page: Number(addForm.value.primary_page) || currentPage.value,
            norm_bbox: bbox,
            raw_text: `${addForm.value.tag || selectedItemObj.value.tag} (人工框选绑定)`,
            replace_primary: true,
            target_tag: addForm.value.tag || null,
            symbol_class: addForm.value.symbol_class || null,
          }),
        }
      );
      if (res.ok) {
        if (!applyProjectResponse(await res.json())) return;
        showAddModal.value = false;
        showToast(`🔗 已保存坐标绑定；确认符号类型且审核通过的框可用于训练。`);
      } else {
        const error = await res.json();
        alert(error.detail || "绑定未保存");
      }
    }

    function openAddModal() {
      boxCropBase64.value = "";
      boxRecognizedTokens.value = [];
      addForm.value = {
        tag: "-NEW1",
        brand: "施耐德",
        name: "",
        model: "",
        unit: "个",
        quantity: 1,
        remark: "人工网页新增",
        primary_page: currentPage.value,
        primary_bbox: [0.15, 0.2, 0.35, 0.3],
      };
      showAddModal.value = true;
    }

    async function submitAddItem() {
      if (!displayedProjectIsCurrent()) return;
      const payload = {
        ...addForm.value,
        symbol_class: addForm.value.symbol_class || null,
        quantity: Number(addForm.value.quantity),
        primary_page: Number(addForm.value.primary_page) || currentPage.value,
        primary_bbox: addForm.value.primary_bbox || [0.15, 0.2, 0.35, 0.3],
      };
      const res = await fetch(`/api/projects/${currentProjectId.value}/bom/item`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        if (!applyProjectResponse(await res.json())) return;
        showAddModal.value = false;
        showToast(`✅ 已新增物料 [${payload.name} ${payload.model}]，坐标与审核结果已保存。`);
      }
    }

    async function restoreFiltered(item) {
      if (!displayedProjectIsCurrent()) return;
      const res = await fetch(
        `/api/projects/${currentProjectId.value}/filtered/${item.id}/restore`,
        { method: "POST" }
      );
      if (res.ok) {
        if (!applyProjectResponse(await res.json())) return;
        showToast(`↩️ 已将 [${item.tag} ${item.name}] 恢复至正式 BOM！`);
      }
    }

    async function handleReviewAction(rev, action) {
      if (!displayedProjectIsCurrent()) return;
      const res = await fetch(
        `/api/projects/${currentProjectId.value}/reviews/${rev.id}/action`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        }
      );
      if (res.ok) {
        if (!applyProjectResponse(await res.json())) return;
        showToast(
          action === "ACCEPT"
            ? `✅ 已采纳审核项 [${rev.tag}] 并同步更新 BOM 清单！`
            : `✔️ 已标记审核项 [${rev.tag}] 状态`
        );
      }
    }

    function handleFileUpload(ev) {
      const file = ev.target.files[0];
      if (!file || loading.value) return;
      pendingUpload.value=file;uploadWorkflow.value='SCHEMATIC_CHECK';showUploadDialog.value=true;
      ev.target.value='';
    }
    function cancelUpload() { if (loading.value) return;showUploadDialog.value=false;pendingUpload.value=null; }
    async function submitUpload() {
      const file=pendingUpload.value;
      if (!file || loading.value) return;
      loading.value = true;
      showUploadDialog.value=false;
      const isCad = /\.(dwg|dxf)$/i.test(file.name);
      loadingMessage.value = isCad ? '正在读取 CAD、按图框分页并生成审核视图，请稍候…' : '正在读取 PDF 并提取物料…';
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append('workflow_mode',uploadWorkflow.value);
        const res = await fetch("/api/projects/upload", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const err = await res.json();
          showToast(err.detail || "上传解析失败");showUploadDialog.value=true;
          return;
        }
        const data = await res.json();
        await fetchProjects();
        await loadProject(data.project_id);
        pendingUpload.value=null;
        setWorkspaceMode('compare');
        showToast(data.summary?.cad_import ? `已导入 ${data.page_count} 张 CAD 图纸，请先查看导入提醒并核对有效图框。` : data.summary?.extraction_quality?.status === "LOW_COVERAGE" ? "提取覆盖不足，请先查看质量提示并核实原图。" : `已完成 [${file.name}] 初步提取，请核对型号和采购范围。`);
      } catch {
        showToast('图纸导入连接失败，请检查服务和文件，已有项目仍可查看。');
      } finally {
        loading.value = false;
      }
    }

    async function createCadSelection() {
      if (loading.value || !cadSelectedPages.value.length) return;
      loading.value = true; loadingMessage.value = '正在按所选图框建立新项目，原项目审核结果已保留…';
      try {
        const response = await fetch(`/api/projects/${encodeURIComponent(currentProjectId.value)}/cad-selection`, {
          method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({pages:cadSelectedPages.value}),
        });
        const data = await response.json();
        if (!response.ok) { showToast(data.detail || '选图失败'); return; }
        await fetchProjects(); await loadProject(data.project_id); setWorkspaceMode('compare');
        showToast(`已建立 ${data.page_count} 张图纸的选图项目；原项目审核结果保留。`);
      } catch { showToast('CAD 选图连接失败，请重试。'); }
      finally { loading.value = false; }
    }

    async function openRulesModal() {
      const res = await fetch("/api/rules");
      const data = await res.json();
      rulesJsonText.value = JSON.stringify(data, null, 2);
      showRulesModal.value = true;
    }

    async function saveRulesAndReextract() {
      let parsed;
      try {
        parsed = JSON.parse(rulesJsonText.value);
      } catch (e) {
        alert("JSON 格式有误，请检查语法：" + e.message);
        return;
      }
      await fetch("/api/rules", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      showRulesModal.value = false;
      if (currentProjectId.value) {
        loading.value = true;
        const res = await fetch(`/api/projects/${currentProjectId.value}/reextract`, {
          method: "POST",
        });
        if (res.ok) {
          projectData.value = await res.json();
          showToast("✅ 企业规则库已更新并重新完成图纸解析！");
        } else {
          const error = await res.json();
          showToast(error.detail || '重新解析未完成，已有审核结果保留。');
        }
        loading.value = false;
      }
    }

    async function openVlmModal(tab='page') {
      aiConfigTab.value=tab==='item' ? 'item' : 'page';
      aiConfigError.value='';
      qwenKeyValue.value='';
      qwenTestResult.value=null;
      const [pageResponse,itemResponse]=await Promise.all([fetch('/api/vlm-config'),fetch('/api/qwen-config')]);
      if (pageResponse.ok) vlmForm.value=await pageResponse.json();
      if (itemResponse.ok) {const itemConfig=await itemResponse.json();qwenHasKey.value=itemConfig.has_api_key;qwenKeySource.value=itemConfig.key_source || 'none';}
      showVlmModal.value = true;
    }

    async function saveVlmConfig() {
      const response = await fetch("/api/vlm-config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(vlmForm.value),
      });
      if (!response.ok) {
        const error = await response.json();
        aiConfigError.value=error.detail || "保存整页识别配置失败";
        return;
      }
      showVlmModal.value = false;
      showToast("已保存整份 / 本页 AI 识别配置");
    }

    async function triggerAiPageRecognize() {
      await startAiExtraction([currentPage.value]);
    }

    function downloadExcel() {
      if (!currentProjectId.value) return;
      window.location.href = `/api/projects/${currentProjectId.value}/export/excel`;
    }

    function downloadExcelDiagnostics() {
      if (!currentProjectId.value) return;
      window.location.href=`/api/projects/${currentProjectId.value}/export/excel/diagnostics`;
    }

    function downloadJson() {
      if (!currentProjectId.value) return;
      window.location.href = `/api/projects/${currentProjectId.value}/export/json`;
    }

    onMounted(() => {
      fetch('/api/auth/me').then(r=>r.json()).then(u=>{currentRole.value=u.role||'engineer'}).catch(()=>{});
      fetchProjects();

      const handleGlobalEsc = (e) => {
        if (e.defaultPrevented) return;
        if (e.key === "Escape" || e.key === "Esc" || e.keyCode === 27) {
          if (editingId.value || detailItemId.value) {
            e.preventDefault(); e.stopPropagation(); closeItemPanel(); return;
          }
          document.querySelectorAll('.toolbar-menu[open]').forEach(menu => menu.removeAttribute('open'));
          if (e.target?.classList?.contains('search-input') && searchQuery.value) {
            searchQuery.value = ''; e.preventDefault(); return;
          }
          clearActiveBox();
          if (isDrawBoxMode.value) {
            isDrawBoxMode.value = false;
            drawingBoxRect.value = null;
          }
          if (e.target && typeof e.target.blur === "function") {
            e.target.blur();
          }
        }
      };
      window.addEventListener("keydown", handleGlobalEsc, true);
      document.addEventListener('pointerdown', e => {
        if (!e.target.closest('.toolbar-menu')) document.querySelectorAll('.toolbar-menu[open]').forEach(menu => menu.removeAttribute('open'));
      });

      window.addEventListener("keydown", (e) => {
        if (editingId.value || detailItemId.value) {
          if (e.key === 'Tab') {
            const elements = [...document.querySelectorAll('.item-panel button:not([disabled]), .item-panel input:not([disabled]), .item-panel select:not([disabled]), .item-panel textarea:not([disabled])')].filter(el => el.offsetParent !== null);
            const first = elements[0], last = elements[elements.length - 1];
            if (first && e.shiftKey && (document.activeElement === first || !elements.includes(document.activeElement))) { e.preventDefault(); last.focus(); }
            else if (last && !e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
          }
          return;
        }
        if (showAddModal.value || showRulesModal.value || showVlmModal.value || showYoloModal.value || showUploadDialog.value || showTableDialog.value) return;
        const tag = (e.target && e.target.tagName) || "";
        if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
        if (!projectData.value) return;
        if (e.key === '/') { e.preventDefault(); document.querySelector('.search-input')?.focus(); return; }
        if (e.key === "ArrowLeft" && currentPage.value > 1) {
          changePage(currentPage.value - 1);
        } else if (e.key === "ArrowRight" && currentPage.value < projectData.value.page_count) {
          changePage(currentPage.value + 1);
        }
      });
    });

    return {
      showUploadDialog,pendingUpload,uploadWorkflow,submitUpload,cancelUpload,tableInfo,isTableExtraction,
      showTableDialog,tablePageChoice,tableSelectedPages,openTableExtraction,createTableProject,
      showAnalysisView, independentInfo, legacyTableProject, currentPageEligible, reconciliationFilter, reconciliationRows, pendingReconciliation, reconciliationLabels, decisionLabel, reconciliationBusy,
      canAdoptReference, locateComparison, handleReconciliation, createIndependentProject,
      clearActiveBox,
      handleCanvasClick,
      projects,
      currentRole,
      fetchProjects,
      currentProjectId,
      projectData,
      loading,
      loadingMessage,
      showCadSheets, cadSelectedPages, createCadSelection,
      leftPaneWidth,
      workspaceMode, setWorkspaceMode, overviewExpanded, toggleOverview, issueFilter, issueLabels, issueCounts, focusIssue, resetFilters, filtersActive,
      needsModel, modelPendingLabel, hasPendingComponents, componentLabel, componentEvidence, materializeComponent, itemIssues, scopeLabel, roleLabel, evidencePages, itemDrawingReference, matchesIssue, hasDistinctSpecification, drawingAnnotations, drawingAnnotationSummary,
      reviewStatusFilter, visibleReviews, awaitingFirstBom, pagesToCheck, expandedReview, unboundLocations, detailItemId, drawerItem, savingEdit, editDirty, editTargetChanged,
      fieldClassification, fieldClassificationBusy, fieldClassificationError, fieldClassificationTarget, classifyDrawingText, acceptFieldClassification,
      fieldHistory, fieldHistoryBusy, fieldHistoryError, formatFieldRecordTime, showFieldRecord,
      fieldPreview, fieldPreviewBusy, previewDrawingText,
      qwenKeyValue, qwenHasKey, qwenKeySource, qwenKeySaving, qwenTestBusy, qwenTestResult,
      openQwenKeyEditor, saveQwenKey, testQwenConnection,
      openItemDetails, openItemIssue, tableCountLabel, closeItemPanel, locateItem, locateItemEvidence, deletePanelItem, closeToolsMenu,
      isResizingSplit,
      pdfViewportRef,
      pdfStageRef,
      currentPage,
      showDrawingGrid, activeGridColumn, pageGridColumns, pageGridBands, crossPageLinks, visibleCrossPageLinks,
      crossPageLinkStyle, pageReferenceIndex,
      focusDrawingColumn, jumpToCrossReference,
      pageImageReady, pageImageError, onPageImageLoad, onPageImageError, retryPageImage, currentImageView, toggleAnalysisView,
      zoomPercent,
      autoFocusZoom,
      isPanning,
      isDrawBoxMode,
      drawingBoxRect,
      boxCropBase64,
      boxRecognizedTokens,
      tokenFillTarget,
      activeBox,
      selectedItemId,
      selectedDrawingHit,
      selectedItemObj,
      terminalHighlights,
      drawingBoxStyle,
      selectedEvidences,
      currentEvIndex,
      evDedupeByTag,
      evExpanded,
      uniqueTagEvidences,
      displayedEvidences,
      evidenceChipLabels,
      activeTab,
      deviceInventory, inventoryLoading, inventoryFamily, inventoryFamilies, inventoryItems, openInventory, locateInventoryItem,
      followDrawingPage, pageOrderMode, circuitPageOrder, useCircuitOrder, detailTab, pageOrderActive, pagePositions, tableScopeItems,
      searchQuery,
      sourceFilter,
      categoryFilter,
      editingId,
      editForm,
      showAddModal,
      addForm,
      showRulesModal,
      rulesJsonText,
      showVlmModal,
      aiConfigTab, aiConfigError,
      vlmForm,
      showSomLayer,
      somFilter, somShowExcluded, somKinds, somCounts, somFilterLabel, selectedSomId, focusedCandidate, highlightedCandidateKey,
      topologyLoading, topologyError, loadPageTopology, visibleSomBranches,
      filteredSomCandidates,
      overlaySomCandidates,
      pageTopology,
      showYoloModal,
      yoloStatus,
      aiLoading,
      aiJob,
      aiBusy,
      aiStatusLabel,
      startAiExtraction,
      cancelAiJob,
      toastMessage,
      currentPageInfo,
      pageImageUrl,
      displayedPageImageUrl,
      isPartListMode,
      primaryBomList,
      secondaryBomList,
      baseBomList,
      primaryCatCount,
      auxCatCount,
      currentTableItems,
      pendingReviewCount,
      loadProject,
      clearAllProjects,
      changePage,
      onPageChanged,
      sourceLabel,
      jumpToLocation,
      selectBomRow,
      zoomToBomRow,
      selectEvidence,
      selectEvidenceByIndex,
      stepEvidence,
      handleEvScrollWheel,
      selectReviewCard,
      isProjectReview, canLocateReview, showCadImportInfo,
      selectSomCandidate,
      handleCanvasWheel,
      startPan,
      movePan,
      endPan,
      handleCanvasDblClick,
      stepZoom,
      fitDrawingPage,
      resetZoom,
      setZoomAndCenter,
      focusOnActiveBox,
      panCenterToActiveBox,
      startSplitResize,
      startEdit,
      cancelEdit,
      saveEdit,
      applySelection,
      deleteBomRow,
      openAddModal,
      submitAddItem,
      toggleDrawBoxMode,
      applyRecognizedToken,
      bindDrawnBoxToSelectedRow,
      restoreFiltered,
      handleReviewAction,
      handleFileUpload,
      openRulesModal,
      saveRulesAndReextract,
      openVlmModal,
      saveVlmConfig,
      toggleSomLayer,
      openYoloModal,
      archiveCurrentProjectToYolo,
      triggerAiPageRecognize,
      downloadExcel,
      downloadExcelDiagnostics,
      downloadJson,
    };
  },
}).mount("#app");
