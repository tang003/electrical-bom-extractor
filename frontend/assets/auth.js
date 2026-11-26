(() => {
  const original = window.fetch.bind(window);
  window.bomFetch = original;
  const versions = new Map();
  window.fetch = (input, options = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    const method = (options.method || (typeof input !== 'string' && input.method) || 'GET').toUpperCase();
    const headers = new Headers(options.headers || (typeof input !== 'string' ? input.headers : undefined));
    const projectMatch = url.match(/^\/api\/projects\/([^/?#]+)/);
    const projectId = projectMatch ? decodeURIComponent(projectMatch[1]) : null;
    if (projectId && versions.has(projectId) && !['GET', 'HEAD', 'OPTIONS'].includes(method)) {
      headers.set('If-Match', String(versions.get(projectId)));
    }
    if (url.startsWith('/api/') && !['GET', 'HEAD', 'OPTIONS'].includes(method)) {
      const csrf = document.cookie.split('; ').find(x => x.startsWith('bom_csrf='));
      if (csrf) headers.set('X-BOM-CSRF', decodeURIComponent(csrf.split('=').slice(1).join('=')));
    }
    return original(input, { ...options, headers }).then(response => {
      if (projectId && response.ok) {
        const version = response.headers.get('X-BOM-Version');
        if (version) versions.set(projectId, Number(version));
        else if (method === 'GET' && /^\/api\/projects\/[^/?#]+$/.test(url)) {
          response.clone().json().then(data => { if (data.data_version) versions.set(projectId, data.data_version); }).catch(() => {});
        }
      }
      if (response.status === 401 && !url.startsWith('/api/auth/')) location.href = '/login';
      return response;
    });
  };
})();
