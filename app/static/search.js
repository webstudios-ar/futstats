// Buscador inteligente con autocomplete + búsqueda de partidos
(function() {
    const input = document.getElementById('search-input');
    const results = document.getElementById('search-results');
    if (!input || !results) return;
    
    let debounceTimer = null;
    let currentRequest = null;
    let focusedIdx = -1;
    let currentItems = [];
    
    function escapeHtml(s) {
        return String(s || '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    
    function fmtDate(iso) {
        const d = new Date(iso);
        const now = new Date();
        const opts = { hour: '2-digit', minute: '2-digit' };
        const sameDay = d.toDateString() === now.toDateString();
        if (sameDay) return 'HOY · ' + d.toLocaleTimeString([], opts);
        const tomorrow = new Date(now); tomorrow.setDate(now.getDate() + 1);
        if (d.toDateString() === tomorrow.toDateString()) {
            return 'MAÑ · ' + d.toLocaleTimeString([], opts);
        }
        return d.toLocaleDateString([], {day: '2-digit', month: 'short'}).toUpperCase() +
            ' · ' + d.toLocaleTimeString([], opts);
    }
    
    function renderResults(matches, teams, query) {
        currentItems = [];
        let html = '';
        
        if (matches && matches.length > 0) {
            html += '<div class="search-result-section">';
            html += '<div class="search-section-title">Partidos</div>';
            matches.forEach((m, i) => {
                currentItems.push({type: 'match', id: m.id});
                const homeLogo = m.home.logo ? `<img src="${escapeHtml(m.home.logo)}" alt="">` : '';
                const awayLogo = m.away.logo ? `<img src="${escapeHtml(m.away.logo)}" alt="">` : '';
                html += `
                    <button class="search-result-item" data-idx="${currentItems.length - 1}">
                        <div class="team-logos">${homeLogo}${awayLogo}</div>
                        <div class="teams">
                            <div class="teams-line">
                                <span>${escapeHtml(m.home.name)}</span>
                                <span class="vs-sep">vs</span>
                                <span>${escapeHtml(m.away.name)}</span>
                            </div>
                            <div class="meta">${escapeHtml(m.league || '')}</div>
                        </div>
                        <div class="date">${fmtDate(m.date)}</div>
                    </button>
                `;
            });
            html += '</div>';
        }
        
        if (teams && teams.length > 0 && (!matches || matches.length === 0)) {
            html += '<div class="search-result-section">';
            html += '<div class="search-section-title">Equipos</div>';
            teams.forEach((t) => {
                currentItems.push({type: 'team', name: t.name});
                const logo = t.logo ? `<img src="${escapeHtml(t.logo)}" alt="">` : '';
                html += `
                    <button class="search-result-item" data-idx="${currentItems.length - 1}">
                        <div class="team-logos">${logo}</div>
                        <div class="teams">
                            <div class="teams-line"><span>${escapeHtml(t.name)}</span></div>
                            <div class="meta">${escapeHtml(t.league || '')}</div>
                        </div>
                    </button>
                `;
            });
            html += '</div>';
        }
        
        if (currentItems.length === 0) {
            html = `<div class="search-empty">Sin resultados para "${escapeHtml(query)}"</div>`;
        }
        
        results.innerHTML = html;
        results.classList.add('show');
        focusedIdx = -1;
        
        // Bind clicks
        results.querySelectorAll('.search-result-item').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.dataset.idx);
                selectItem(currentItems[idx]);
            });
        });
    }
    
    function selectItem(item) {
        if (item.type === 'match') {
            window.location.href = `/match/${item.id}`;
        } else if (item.type === 'team') {
            // Si selecciona un equipo, ponemos su nombre en el input y disparamos otra búsqueda
            input.value = item.name;
            input.focus();
            doSearch(item.name);
        }
    }
    
    async function doSearch(q) {
        q = q.trim();
        if (q.length < 2) {
            results.classList.remove('show');
            return;
        }
        
        if (currentRequest) currentRequest.abort();
        const controller = new AbortController();
        currentRequest = controller;
        
        try {
            // Si contiene "vs" / "v" / "-", buscar partidos
            const hasVs = /\s(vs|v|-|contra)\s/i.test(q);
            
            const [matchesResp, teamsResp] = await Promise.all([
                fetch(`/api/search?q=${encodeURIComponent(q)}&limit=8`, {signal: controller.signal}),
                hasVs ? Promise.resolve(null) : fetch(`/api/search/teams?q=${encodeURIComponent(q)}&limit=6`, {signal: controller.signal}),
            ]);
            
            const matches = matchesResp.ok ? await matchesResp.json() : [];
            const teams = teamsResp && teamsResp.ok ? await teamsResp.json() : [];
            
            renderResults(matches, teams, q);
        } catch (err) {
            if (err.name !== 'AbortError') console.error(err);
        }
    }
    
    input.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const q = input.value;
        if (q.trim().length < 2) {
            results.classList.remove('show');
            return;
        }
        debounceTimer = setTimeout(() => doSearch(q), 180);
    });
    
    input.addEventListener('keydown', (e) => {
        const items = results.querySelectorAll('.search-result-item');
        if (items.length === 0) return;
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            focusedIdx = Math.min(focusedIdx + 1, items.length - 1);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            focusedIdx = Math.max(focusedIdx - 1, 0);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (focusedIdx >= 0 && currentItems[focusedIdx]) {
                selectItem(currentItems[focusedIdx]);
                return;
            } else if (items.length > 0) {
                items[0].click();
                return;
            }
        } else if (e.key === 'Escape') {
            results.classList.remove('show');
            input.blur();
            return;
        } else {
            return;
        }
        
        items.forEach((el, i) => {
            if (i === focusedIdx) el.classList.add('focused');
            else el.classList.remove('focused');
        });
        if (focusedIdx >= 0) items[focusedIdx].scrollIntoView({block: 'nearest'});
    });
    
    // Cerrar al hacer click fuera
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !results.contains(e.target)) {
            results.classList.remove('show');
        }
    });
    
    input.addEventListener('focus', () => {
        if (input.value.trim().length >= 2 && results.innerHTML) {
            results.classList.add('show');
        }
    });
})();
