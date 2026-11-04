const { createApp, ref, computed, onMounted, nextTick } = Vue;

const SOURCE_LABELS = {
  EXPLICIT: "原图明文",
  TOPOLOGY_INFERRED: "拓扑推导",
  RULE_COMPLETED: "规则补全",
  ESTIMATED_AUX: "工程估算",
  MANUAL_EDITED: "人工已改",
};

createApp({
  setup() {
    const projects = ref([]);
    const currentProjectId = ref("");
    const projectData = ref(null);
    const loading = ref(false);

    // PDF viewer state
    const currentPage = ref(1);
    const zoomPercent = ref(100);
    const activeBox = ref(null); // { page, norm_bbox: [x0,y0,x1,y1], label }
    const selectedItemId = ref("");
    const selectedEvidences = ref([]);

    // Right pane tabs & filters
    const activeTab = ref("primary"); // "primary" | "secondary" | "reviews" | "filtered"
    const searchQuery = ref("");
    const sourceFilter = ref("ALL");

    // Inline editing state
    const editingId = ref("");
    const editForm = ref({});

    // Modals
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
    const vlmForm = ref({
      provider: "阿里云百炼 (Qwen-VL) / OpenAI 兼容接口",
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
      model: "qwen-vl-max",
      critic_model: "deepseek-r1",
      api_key: "",
    });
    const showSomLayer = ref(false);
    const pageTopology = ref(null);
    const showYoloModal = ref(false);
    const yoloStatus = ref(null);
    const aiLoading = ref(false);
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
      if (projects.value.length > 0 && !currentProjectId.value) {
        await loadProject(projects.value[0].project_id);
      }
    }

    async function loadProject(projId) {
      if (!projId) return;
      loading.value = true;
      try {
        currentProjectId.value = projId;
        const res = await fetch(`/api/projects/${projId}`);
        projectData.value = await res.json();
        currentPage.value = 1;
        activeBox.value = null;
        selectedItemId.value = "";
        selectedEvidences.value = [];
        editingId.value = "";
        activeTab.value = "primary";
      } finally {
        loading.value = false;
      }
    }

    const currentPageInfo = computed(() => {
      if (!projectData.value || !projectData.value.pages) return null;
      return projectData.value.pages.find((p) => p.page_num === currentPage.value) || null;
    });

    const pageImageUrl = computed(() => {
      if (!currentProjectId.value || !currentPage.value) return "";
      if (showSomLayer.value) {
        return `/api/projects/${currentProjectId.value}/pages/${currentPage.value}/som-image?dpi=180`;
      }
      return `/api/projects/${currentProjectId.value}/pages/${currentPage.value}/image?dpi=150`;
    });

    async function loadPageTopology() {
      if (!currentProjectId.value || !currentPage.value || !showSomLayer.value) return;
      const res = await fetch(
        `/api/projects/${currentProjectId.value}/pages/${currentPage.value}/topology`
      );
      if (res.ok) {
        pageTopology.value = await res.json();
      }
    }

    async function toggleSomLayer() {
      showSomLayer.value = !showSomLayer.value;
      if (showSomLayer.value) {
        await loadPageTopology();
        showToast("🎯 已开启微软 SoM [#编号] 候选框与 NetworkX 纵向导线连通层！");
      } else {
        pageTopology.value = null;
      }
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

    const currentTableItems = computed(() => {
      const base = activeTab.value === "secondary" ? secondaryBomList.value : primaryBomList.value;
      const q = searchQuery.value.trim().toLowerCase();
      return base.filter((it) => {
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
          String(it.seq_no).includes(q)
        );
      });
    });

    const pendingReviewCount = computed(() => {
      if (!projectData.value || !projectData.value.review_items) return 0;
      return projectData.value.review_items.filter((r) => r.status === "PENDING").length;
    });

    function sourceLabel(src) {
      return SOURCE_LABELS[src] || src;
    }

    function jumpToLocation(page, normBbox, label) {
      currentPage.value = page;
      activeBox.value = {
        page: page,
        norm_bbox: normBbox || [0.1, 0.1, 0.3, 0.2],
        label: label || "",
      };
      nextTick(() => {
        const viewport = document.querySelector(".pdf-canvas-viewport");
        const boxEl = document.querySelector(".bbox-highlight");
        if (viewport && boxEl) {
          boxEl.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
        }
      });
    }

    function selectBomRow(item) {
      selectedItemId.value = item.id;
      selectedEvidences.value = item.evidences || [];
      const label = `${item.tag || "#" + item.seq_no} | ${item.name} (${item.model || "无型号"})`;
      jumpToLocation(item.primary_page || 1, item.primary_bbox, label);
    }

    function selectEvidence(ev, itemLabel) {
      jumpToLocation(ev.page, ev.norm_bbox, `${itemLabel || ev.raw_text} [P${ev.page}]`);
    }

    function selectReviewCard(rev) {
      selectedItemId.value = rev.id;
      selectedEvidences.value = [];
      jumpToLocation(rev.page, rev.norm_bbox, `${rev.tag} - ${rev.issue_type}`);
    }

    function startEdit(item) {
      editingId.value = item.id;
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
      };
    }

    function cancelEdit() {
      editingId.value = "";
      editForm.value = {};
    }

    async function saveEdit(item) {
      const viewMode =
        activeTab.value === "secondary"
          ? isPartListMode.value
            ? "aggregated"
            : "itemized"
          : "primary";

      const payload = {
        ...editForm.value,
        view_mode: viewMode,
        quantity: Number(editForm.value.quantity) || 0,
      };

      const res = await fetch(`/api/projects/${currentProjectId.value}/bom/item`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        projectData.value = await res.json();
        editingId.value = "";
        showToast(`✅ 已保存序号 ${item.seq_no} 修改并实时同步汇总表与 Excel！`);
      }
    }

    async function deleteBomRow(item) {
      if (!confirm(`确定从 BOM 中删除序号 ${item.seq_no} [${item.name} ${item.model}] 吗？`)) {
        return;
      }
      const res = await fetch(`/api/projects/${currentProjectId.value}/bom/item/${item.id}`, {
        method: "DELETE",
      });
      if (res.ok) {
        projectData.value = await res.json();
        showToast(`🗑️ 已删除物料 [${item.name}]`);
      }
    }

    function openAddModal() {
      addForm.value = {
        tag: "-NEW1",
        brand: "施耐德",
        name: "",
        model: "",
        unit: "个",
        quantity: 1,
        remark: "人工网页新增",
        primary_page: currentPage.value,
      };
      showAddModal.value = true;
    }

    async function submitAddItem() {
      const payload = {
        ...addForm.value,
        quantity: Number(addForm.value.quantity) || 1,
        primary_page: Number(addForm.value.primary_page) || currentPage.value,
        primary_bbox: [0.15, 0.2, 0.35, 0.3],
      };
      const res = await fetch(`/api/projects/${currentProjectId.value}/bom/item`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        projectData.value = await res.json();
        showAddModal.value = false;
        showToast(`✅ 已新增物料 [${payload.name} ${payload.model}]`);
      }
    }

    async function restoreFiltered(item) {
      const res = await fetch(
        `/api/projects/${currentProjectId.value}/filtered/${item.id}/restore`,
        { method: "POST" }
      );
      if (res.ok) {
        projectData.value = await res.json();
        showToast(`↩️ 已将 [${item.tag} ${item.name}] 恢复至正式 BOM！`);
      }
    }

    async function handleReviewAction(rev, action) {
      const res = await fetch(
        `/api/projects/${currentProjectId.value}/reviews/${rev.id}/action`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        }
      );
      if (res.ok) {
        projectData.value = await res.json();
        showToast(
          action === "ACCEPT"
            ? `✅ 已采纳审核项 [${rev.tag}] 并同步更新 BOM 清单！`
            : `✔️ 已标记审核项 [${rev.tag}] 状态`
        );
      }
    }

    async function handleFileUpload(ev) {
      const file = ev.target.files[0];
      if (!file) return;
      loading.value = true;
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/api/projects/upload", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const err = await res.json();
          alert(err.detail || "上传解析失败");
          return;
        }
        const data = await res.json();
        await fetchProjects();
        currentProjectId.value = data.project_id;
        projectData.value = data;
        currentPage.value = 1;
        showToast(`🎉 图纸 [${file.name}] 解析完成！`);
      } finally {
        loading.value = false;
        ev.target.value = "";
      }
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
        }
        loading.value = false;
      }
    }

    async function openVlmModal() {
      const res = await fetch("/api/vlm-config");
      vlmForm.value = await res.json();
      showVlmModal.value = true;
    }

    async function saveVlmConfig() {
      await fetch("/api/vlm-config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(vlmForm.value),
      });
      showVlmModal.value = false;
      showToast("🤖 已保存 AI 视觉大模型配置！");
    }

    async function triggerAiPageRecognize() {
      if (!currentProjectId.value) return;
      aiLoading.value = true;
      try {
        const res = await fetch(
          `/api/projects/${currentProjectId.value}/pages/${currentPage.value}/ai-recognize`,
          { method: "POST" }
        );
        const data = await res.json();
        if (!res.ok) {
          alert(data.detail || "AI 视觉识别调用失败");
          if ((data.detail || "").includes("API Key")) {
            await openVlmModal();
          }
          return;
        }
        projectData.value = data;
        showToast(`✨ AI 视觉模型成功从第 ${currentPage.value} 页识别并追加 ${data.ai_added_count || 0} 项元器件！`);
      } finally {
        aiLoading.value = false;
      }
    }

    function downloadExcel() {
      if (!currentProjectId.value) return;
      window.location.href = `/api/projects/${currentProjectId.value}/export/excel`;
    }

    function downloadJson() {
      if (!currentProjectId.value) return;
      window.location.href = `/api/projects/${currentProjectId.value}/export/json`;
    }

    onMounted(() => {
      fetchProjects();
    });

    return {
      projects,
      currentProjectId,
      projectData,
      loading,
      currentPage,
      zoomPercent,
      activeBox,
      selectedItemId,
      selectedEvidences,
      activeTab,
      searchQuery,
      sourceFilter,
      editingId,
      editForm,
      showAddModal,
      addForm,
      showRulesModal,
      rulesJsonText,
      showVlmModal,
      vlmForm,
      showSomLayer,
      pageTopology,
      showYoloModal,
      yoloStatus,
      aiLoading,
      toastMessage,
      currentPageInfo,
      pageImageUrl,
      isPartListMode,
      primaryBomList,
      secondaryBomList,
      currentTableItems,
      pendingReviewCount,
      loadProject,
      sourceLabel,
      jumpToLocation,
      selectBomRow,
      selectEvidence,
      selectReviewCard,
      startEdit,
      cancelEdit,
      saveEdit,
      deleteBomRow,
      openAddModal,
      submitAddItem,
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
      downloadJson,
    };
  },
}).mount("#app");
