(() => {
  const get = id => document.getElementById(id);
  const message = get('message');
  const statusLabels = {
    queued: '排队中',
    running: '解析中',
    reviewing: 'AI 疑点复核中',
    merging: 'AI 结果合并中',
    completed: '已解析',
    partial: '部分完成',
    failed: '解析失败',
    cancelled: '已取消',
    interrupted: '已中断',
  };
  const activeStatuses = new Set(['queued', 'running', 'reviewing', 'merging']);
  let isAdmin = false;
  let users = [];
  let ownership = new Map();
  let projectRows = new Map();
  let importSignature = '';
  let refreshing = false;
  let pollTimer;

  async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '操作失败，请重试。');
    return data;
  }

  function post(url, data) {
    return api(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  function projectUrl(id) {
    return '/api/projects/' + encodeURIComponent(id);
  }

  function cell(text = '') {
    const td = document.createElement('td');
    td.textContent = text;
    return td;
  }

  function button(text, handler, className = '') {
    const element = document.createElement('button');
    element.type = 'button';
    element.textContent = text;
    element.className = className;
    element.onclick = handler;
    return element;
  }

  function detail(parent, text) {
    const label = document.createElement('span');
    label.className = 'detail';
    label.textContent = text;
    parent.append(label);
  }

  function showStatus(element, text, description, tone) {
    const signature = JSON.stringify([text, description, tone]);
    if (element.dataset.signature === signature) return;
    element.dataset.signature = signature;
    element.dataset.tone = tone;
    const label = document.createElement('strong');
    label.textContent = text;
    element.replaceChildren(label);
    if (description) detail(element, description);
  }

  function createProjectRow(project) {
    const row = document.createElement('tr');
    const name = cell(project.source_filename || project.pdf_filename);
    name.className = 'drawing-name';
    const status = cell();
    status.className = 'status';
    const owner = cell();
    const actionCell = cell();
    const actions = document.createElement('div');
    actions.className = 'actions';
    actionCell.append(actions);
    const link = document.createElement('a');
    link.href = '/?project=' + encodeURIComponent(project.project_id);
    link.textContent = '打开';
    actions.append(
      link,
      button('历史', () => showHistory(project)),
      button('移入回收站', () => {
        if (confirm('将该项目移入回收站？')) mutate(projectUrl(project.project_id) + '/trash');
      }, 'danger'),
    );
    let select;
    if (isAdmin) {
      select = document.createElement('select');
      select.setAttribute('aria-label', '选择项目所有者');
      for (const user of users.filter(user => user.active)) {
        select.add(new Option(user.username, user.id));
      }
      actions.append(select, button('转交', () => {
        if (select.value) mutate(projectUrl(project.project_id) + '/transfer', {owner_id: select.value});
      }));
    }
    row.append(name, status, owner, actionCell);
    return {row, status, owner, link, select};
  }

  function updateProjectRow(project, controls) {
    const job = project.ai_job;
    const active = activeStatuses.has(job?.status);
    const description = [
      project.page_count + ' 页',
      project.workflow_mode === 'TABLE_EXTRACT' ? '仅提取部件表' : '',
    ].filter(Boolean).join(' · ');
    const aiProgress = active && job.total_pages ? ' · ' + job.completed_pages + '/' + job.total_pages + ' 页' : '';
    const failed = job && !active && job.status !== 'completed';
    if (active) {
      const status = {queued: 'AI 排队中', running: 'AI 识别中'}[job.status] || statusLabels[job.status];
      showStatus(controls.status, status, (job.message || '') + aiProgress, 'active');
    } else if (failed || project.ai_start_warning) {
      showStatus(controls.status, '初步结果可查看', failed ? 'AI ' + (statusLabels[job.status] || '待确认') + ' · ' + (job.message || '') : project.ai_start_warning, 'alert');
    } else {
      showStatus(controls.status, '已解析', description, 'ready');
    }
    controls.link.hidden = project.initial_ai_pending === true;
    const record = ownership.get(project.project_id);
    controls.owner.textContent = record?.username || '本人';
    if (controls.select && controls.select.dataset.owner !== (record?.owner_id || '')) {
      controls.select.dataset.owner = record?.owner_id || '';
      controls.select.value = record?.owner_id || '';
    }
  }

  function renderImports(imports) {
    const signature = JSON.stringify(imports);
    if (signature === importSignature) return;
    importSignature = signature;
    get('imports').replaceChildren(...imports.map(job => {
      const row = document.createElement('tr');
      const name = cell(job.filename);
      name.className = 'drawing-name';
      if (job.workflow_mode === 'TABLE_EXTRACT') detail(name, '仅提取部件表');
      const status = cell();
      status.className = 'status';
      showStatus(status, statusLabels[job.status] || '待确认', job.message, activeStatuses.has(job.status) ? 'active' : 'alert');
      row.append(name, status, cell('—'), cell());
      return row;
    }));
  }

  async function refreshDrawings() {
    if (refreshing) return;
    refreshing = true;
    clearTimeout(pollTimer);
    let delay = 5000;
    try {
      const {projects, imports} = await api('/api/drawings');
      const ids = projects.map(project => project.project_id);
      const changed = ids.length !== projectRows.size || ids.some(id => !projectRows.has(id));
      if (isAdmin && changed) await refreshOwnership();
      const nextRows = new Map();
      for (const project of projects.slice().reverse()) {
        const controls = projectRows.get(project.project_id) || createProjectRow(project);
        updateProjectRow(project, controls);
        nextRows.set(project.project_id, controls);
      }
      if (changed) get('projects').replaceChildren(...Array.from(nextRows.values(), controls => controls.row));
      projectRows = nextRows;
      renderImports(imports);
      const empty = get('empty');
      empty.hidden = Boolean(projects.length || imports.length);
      empty.textContent = '还没有图纸，点击「上传图纸」开始。';
      if (imports.some(job => activeStatuses.has(job.status)) || projects.some(project => activeStatuses.has(project.ai_job?.status))) {
        delay = 2000;
      }
    } catch (error) {
      message.textContent = '图纸列表暂时无法更新。' + error.message;
    } finally {
      refreshing = false;
      pollTimer = setTimeout(refreshDrawings, delay);
    }
  }

  async function refreshOwnership() {
    if (!isAdmin) return;
    const records = await api('/api/projects/ownership');
    ownership = new Map(records.map(row => [row.id, row]));
  }

  async function refreshTrash() {
    const trash = await api('/api/projects/trash');
    get('trash-empty').hidden = Boolean(trash.length);
    get('trash').replaceChildren(...trash.map(project => {
      const row = document.createElement('tr');
      const actions = cell();
      actions.append(button('恢复', () => mutate(projectUrl(project.id) + '/restore')));
      row.append(cell(project.name), actions);
      return row;
    }));
  }

  async function mutate(url, data) {
    try {
      await post(url, data);
      await Promise.all([refreshTrash(), refreshOwnership()]);
      await refreshDrawings();
      message.textContent = '已更新。';
    } catch (error) {
      message.textContent = error.message;
    }
  }

  async function showHistory(project) {
    try {
      const url = projectUrl(project.project_id);
      const revisions = await api(url + '/revisions');
      get('history-title').textContent = '历史版本 · ' + (project.project_name || project.pdf_filename);
      get('history').replaceChildren(...revisions.map(revision => {
        const row = document.createElement('div');
        row.className = 'history';
        row.textContent = '版本 ' + revision.version + ' · ' + revision.summary + ' · ' + new Date(revision.created_at).toLocaleString();
        row.append(button('恢复此版本', async () => {
          if (!confirm('恢复此版本并生成新版本？')) return;
          try {
            const current = await api(url);
            await post(url + '/revisions/' + revision.version + '/restore', {expected_version: current.data_version});
            message.textContent = '已恢复。';
            await showHistory(project);
            refreshDrawings();
          } catch (error) {
            message.textContent = error.message;
          }
        }));
        return row;
      }));
      const section = get('history-section');
      section.hidden = false;
      section.scrollIntoView({behavior: 'smooth'});
    } catch (error) {
      message.textContent = error.message;
    }
  }

  const dialog = get('upload-dialog');
  const form = get('upload-form');
  const submit = get('submit-upload');
  const cancel = get('cancel-upload');
  const uploadError = get('upload-error');
  get('show-upload').onclick = () => {
    form.reset();
    uploadError.textContent = '';
    dialog.showModal();
  };
  cancel.onclick = () => dialog.close();
  dialog.addEventListener('cancel', event => {
    if (submit.disabled) event.preventDefault();
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (submit.disabled) return;
    uploadError.textContent = '';
    const data = new FormData(form);
    const file = data.get('file');
    if (file.size > 100 * 1024 * 1024) {
      uploadError.textContent = '文件超过 100 MB，请拆分后上传。';
      return;
    }
    submit.disabled = cancel.disabled = true;
    submit.textContent = '正在上传…';
    try {
      await api('/api/drawing-imports', {method: 'POST', body: data});
      dialog.close();
      message.textContent = '上传完成，服务器已接手解析。关闭网页后可以再回来查看进度。';
      refreshDrawings();
    } catch (error) {
      uploadError.textContent = error.message;
      refreshDrawings();
    } finally {
      submit.disabled = cancel.disabled = false;
      submit.textContent = '上传并排队';
    }
  });

  get('logout').onclick = async () => {
    try {
      await post('/api/auth/logout');
      location.href = '/login';
    } catch (error) {
      message.textContent = error.message;
    }
  };
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) refreshDrawings();
  });

  async function start() {
    try {
      const me = await api('/api/auth/me');
      isAdmin = me.role === 'admin';
      get('admin-link').hidden = !isAdmin;
      get('logout').hidden = false;
      if (isAdmin) users = await api('/api/auth/users');
    } catch (error) {
      if (error.message === '请登录' || error.message === '登录已失效') {
        location.href = '/login';
        return;
      }
    }
    refreshDrawings();
    refreshTrash().catch(error => { message.textContent = error.message; });
  }
  start();
})();
