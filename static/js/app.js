/**
 * MLBites - PyTorch Interview Questions
 * Main Application JavaScript
 */

// ============================================
// State Management
// ============================================

const state = {
    currentQuestion: null,
    showingSolution: false,
    startingCode: '',
    solutionCode: '',
    solutionHtml: '',
    editor: null,
    editorChangeDisposable: null,
    isResizing: false,
    isTerminalResizing: false,
    sidebarCollapsed: false,
    editorFullscreen: false,
    questionFullscreen: false,
    questionActiveTab: 'description',
    isProgrammaticEdit: false,
    drafts: {},
    terminalVisible: false,
    terminalHeightPx: 180,
    activeFramework: 'pytorch', // 'pytorch' | 'numpy'
};

// ============================================
// DOM Elements
// ============================================

const elements = {
    welcomeScreen: document.getElementById('welcome-screen'),
    splitContainer: document.getElementById('split-container'),
    questionPanel: document.getElementById('question-panel'),
    editorPanel: document.getElementById('editor-panel'),
    questionMeta: document.getElementById('question-meta'),
    questionTags: document.getElementById('question-tags'),
    questionContent: document.getElementById('question-content'),
    tabDescription: document.getElementById('tab-description'),
    tabSolution: document.getElementById('tab-solution'),
    questionFullscreenBtn: document.getElementById('question-fullscreen-btn'),
    editorContainer: document.getElementById('editor-container'),
    toggleSolutionBtn: document.getElementById('toggle-solution-btn'),
    resetCodeBtn: document.getElementById('reset-code-btn'),
    formatCodeBtn: document.getElementById('format-code-btn'),
    fullscreenBtn: document.getElementById('fullscreen-btn'),
    runCodeBtn: document.getElementById('run-code-btn'),
    toggleTerminalBtn: document.getElementById('toggle-terminal-btn'),
    terminalPanel: document.getElementById('terminal-panel'),
    terminalResizer: document.getElementById('terminal-resizer'),
    terminalOutput: document.getElementById('terminal-output'),
    terminalStatus: document.getElementById('terminal-status'),
    clearTerminalBtn: document.getElementById('clear-terminal-btn'),
    searchInput: document.getElementById('search-input'),
    questionList: document.getElementById('question-list'),
    frameworkBtnPytorch: document.getElementById('framework-btn-pytorch'),
    frameworkBtnNumpy: document.getElementById('framework-btn-numpy'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    sidebarToggleFloating: document.getElementById('sidebar-toggle-floating'),
    resizer: document.getElementById('resizer'),
    toast: document.getElementById('toast'),
    themeToggle: document.getElementById('theme-toggle'),
    hljsTheme: document.getElementById('hljs-theme'),
};

// ============================================
// Framework Filter (PyTorch / NumPy)
// ============================================

const FRAMEWORK_STORAGE_KEY = 'mlbites:framework:v1';

function normalizeFramework(v) {
    const s = String(v || '').toLowerCase().trim();
    return s === 'numpy' ? 'numpy' : 'pytorch';
}

function loadFrameworkFromStorage() {
    try {
        const raw = localStorage.getItem(FRAMEWORK_STORAGE_KEY);
        if (!raw) return;
        state.activeFramework = normalizeFramework(raw);
    } catch {
        // ignore
    }
}

function saveFrameworkToStorage(framework) {
    try {
        localStorage.setItem(FRAMEWORK_STORAGE_KEY, normalizeFramework(framework));
    } catch {
        // ignore
    }
}

function setFrameworkFilter(framework) {
    const next = normalizeFramework(framework);
    state.activeFramework = next;
    saveFrameworkToStorage(next);

    const isPytorch = next === 'pytorch';
    if (elements.frameworkBtnPytorch) {
        elements.frameworkBtnPytorch.classList.toggle('active', isPytorch);
        elements.frameworkBtnPytorch.setAttribute('aria-pressed', isPytorch ? 'true' : 'false');
    }
    if (elements.frameworkBtnNumpy) {
        elements.frameworkBtnNumpy.classList.toggle('active', !isPytorch);
        elements.frameworkBtnNumpy.setAttribute('aria-pressed', !isPytorch ? 'true' : 'false');
    }

    applySidebarFilters(elements.searchInput?.value || '');
}

// ============================================
// Terminal / Output Panel
// ============================================

const TERMINAL_HEIGHT_STORAGE_KEY = 'mlbites:terminalHeight:v1';

function loadTerminalPrefs() {
    try {
        const raw = localStorage.getItem(TERMINAL_HEIGHT_STORAGE_KEY);
        const v = raw ? Number(raw) : NaN;
        if (Number.isFinite(v) && v >= 80 && v <= 800) {
            state.terminalHeightPx = v;
        }
    } catch {
        // ignore
    }
}

function saveTerminalHeight(heightPx) {
    try {
        localStorage.setItem(TERMINAL_HEIGHT_STORAGE_KEY, String(heightPx));
    } catch {
        // ignore
    }
}

function setTerminalStatus(status, label) {
    if (!elements.terminalStatus) return;
    const normalized = status || 'idle';
    const text = label || 'Idle';
    elements.terminalStatus.textContent = text;
    elements.terminalStatus.classList.remove(
        'terminal-status--idle',
        'terminal-status--running',
        'terminal-status--pass',
        'terminal-status--fail',
        'terminal-status--error',
        'terminal-status--timeout'
    );
    elements.terminalStatus.classList.add(`terminal-status--${normalized}`);
}

function setTerminalOutput(text) {
    if (!elements.terminalOutput) return;
    elements.terminalOutput.textContent = text || '';
}

function setTerminalVisible(visible, { force = false } = {}) {
    const next = force ? !!visible : !!visible;
    state.terminalVisible = next;

    if (elements.toggleTerminalBtn) {
        elements.toggleTerminalBtn.setAttribute('aria-pressed', next ? 'true' : 'false');
        elements.toggleTerminalBtn.title = next ? 'Hide output' : 'Show output';
        elements.toggleTerminalBtn.setAttribute('aria-label', elements.toggleTerminalBtn.title);
    }

    if (!elements.terminalPanel || !elements.terminalResizer) return;

    elements.terminalPanel.classList.toggle('hidden', !next);
    elements.terminalResizer.classList.toggle('hidden', !next);
    elements.terminalPanel.style.height = next ? `${state.terminalHeightPx}px` : '';

    // Re-layout Monaco after the DOM changes
    setTimeout(() => {
        if (state.editor) state.editor.layout();
    }, 0);
}

function clearTerminal() {
    setTerminalStatus('idle', 'Idle');
    setTerminalOutput('');
}

function formatRunResult({ status, duration_ms, stdout, stderr }) {
    const dur = Number.isFinite(duration_ms) ? `${Math.round(duration_ms)}ms` : '';
    const upper = String(status || 'error').toUpperCase();
    const lines = [];
    lines.push(`=== RESULT: ${upper}${dur ? ` (${dur})` : ''} ===`);
    if (stdout) {
        lines.push('');
        lines.push('--- stdout ---');
        lines.push(stdout.trimEnd());
    }
    if (stderr) {
        lines.push('');
        lines.push('--- stderr ---');
        lines.push(stderr.trimEnd());
    }
    return lines.join('\n') + '\n';
}

async function runCurrentCode() {
    if (!state.currentQuestion) {
        showToast('Select a question first.', 'warning');
        return;
    }
    if (!state.editor) {
        showToast('Editor not ready yet.', 'warning');
        return;
    }

    const slug = state.currentQuestion.slug;
    const code = state.editor.getValue();

    // Show output panel immediately so the user sees progress
    setTerminalVisible(true, { force: true });
    setTerminalStatus('running', 'Running…');
    setTerminalOutput('Running tests...\n');

    if (elements.runCodeBtn) {
        elements.runCodeBtn.disabled = true;
        elements.runCodeBtn.setAttribute('aria-busy', 'true');
    }

    try {
        const resp = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question_slug: slug, code }),
        });

        const text = await resp.text();
        let data = null;
        try {
            data = JSON.parse(text);
        } catch {
            data = null;
        }

        if (!resp.ok || !data) {
            setTerminalStatus('error', 'Error');
            setTerminalOutput(`Failed to run.\n\nHTTP ${resp.status}\n${text}\n`);
            showToast('Run failed.', 'error', 2600);
            return;
        }

        const status = data.status || 'error';
        const durationMs = data.duration_ms;
        const stdout = typeof data.stdout === 'string' ? data.stdout : '';
        const stderr = typeof data.stderr === 'string' ? data.stderr : '';

        if (status === 'pass') {
            setTerminalStatus('pass', 'PASS');
            showToast('All tests passed.', 'success');
        } else if (status === 'fail') {
            setTerminalStatus('fail', 'FAIL');
            showToast('Tests failed.', 'warning', 2400);
        } else if (status === 'timeout') {
            setTerminalStatus('timeout', 'TIMEOUT');
            showToast('Run timed out.', 'warning', 2600);
        } else {
            setTerminalStatus('error', 'ERROR');
            showToast('Run error.', 'error', 2600);
        }

        setTerminalOutput(
            formatRunResult({
                status,
                duration_ms: durationMs,
                stdout,
                stderr,
            })
        );
    } catch (e) {
        console.error('[run] request failed', e);
        setTerminalStatus('error', 'Error');
        setTerminalOutput(`Request failed.\n\n${String(e)}\n`);
        showToast('Run request failed.', 'error', 2600);
    } finally {
        if (elements.runCodeBtn) {
            elements.runCodeBtn.disabled = false;
            elements.runCodeBtn.removeAttribute('aria-busy');
        }
    }
}

// ============================================
// Theme (Light/Dark)
// ============================================

const THEME_STORAGE_KEY = 'mlbites:theme:v1';

function getStoredTheme() {
    try {
        const v = localStorage.getItem(THEME_STORAGE_KEY);
        return v === 'light' || v === 'dark' ? v : null;
    } catch {
        return null;
    }
}

function getSystemTheme() {
    try {
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    } catch {
        return 'dark';
    }
}

function setStoredTheme(theme) {
    try {
        localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
        // ignore
    }
}

function setHljsTheme(theme) {
    if (!elements.hljsTheme) return;
    // Keep versions consistent with index.html imports
    const base = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/';
    elements.hljsTheme.href = theme === 'light' ? `${base}github.min.css` : `${base}github-dark.min.css`;
}

function updateThemeToggleUi(theme) {
    if (!elements.themeToggle) return;
    const isLight = theme === 'light';
    elements.themeToggle.textContent = isLight ? '☀️' : '🌙';
    elements.themeToggle.setAttribute('aria-pressed', isLight ? 'true' : 'false');
    elements.themeToggle.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
    elements.themeToggle.setAttribute('aria-label', elements.themeToggle.title);
}

function applyTheme(theme) {
    const next = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    setHljsTheme(next);
    updateThemeToggleUi(next);

    // Monaco theme (if available)
    try {
        if (typeof monaco !== 'undefined' && monaco?.editor?.setTheme) {
            monaco.editor.setTheme(next === 'light' ? 'mlbites-light' : 'mlbites-dark');
        }
    } catch {
        // ignore
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const next = current === 'light' ? 'dark' : 'light';
    setStoredTheme(next);
    applyTheme(next);
}

function initTheme() {
    const stored = getStoredTheme();
    const initial = stored ?? getSystemTheme();
    applyTheme(initial);
}

// ============================================
// Editor Toolbar Helpers
// ============================================

let toastTimer = null;
function showToast(message, variant = 'success', timeoutMs = 1800) {
    if (!elements.toast) return;
    if (toastTimer) clearTimeout(toastTimer);

    elements.toast.textContent = message;
    elements.toast.classList.remove('hidden', 'toast--success', 'toast--warning', 'toast--error');
    elements.toast.classList.add(`toast--${variant}`);

    toastTimer = setTimeout(() => {
        elements.toast.classList.add('hidden');
    }, timeoutMs);
}

const DRAFTS_STORAGE_KEY = 'mlbites:drafts:v1';
let draftsSaveTimer = null;
function loadDraftsFromStorage() {
    try {
        const raw = localStorage.getItem(DRAFTS_STORAGE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
            state.drafts = parsed;
        }
    } catch {
        // ignore
    }
}

function saveDraftsToStorageDebounced() {
    if (draftsSaveTimer) clearTimeout(draftsSaveTimer);
    draftsSaveTimer = setTimeout(() => {
        try {
            localStorage.setItem(DRAFTS_STORAGE_KEY, JSON.stringify(state.drafts));
        } catch {
            // ignore (e.g., storage full)
        }
    }, 250);
}

function getDraft(slug) {
    const v = state.drafts?.[slug];
    return typeof v === 'string' ? v : null;
}

function setDraft(slug, code) {
    if (!slug) return;
    state.drafts[slug] = code;
    saveDraftsToStorageDebounced();
}

function normalizeInvisibleWhitespace(text) {
    return text
        .replace(/\r\n/g, '\n')
        .replace(/[ \t]+$/gm, '')
        .replace(/\s+$/g, '');
}

function basicWhitespaceCleanup(code) {
    // Client-side fallback cleanup (not a full Python formatter)
    let out = code.replace(/\r\n/g, '\n').replace(/\t/g, '    ');
    out = out.replace(/[ \t]+$/gm, '');
    out = out.replace(/\n{4,}/g, '\n\n\n');
    out = out.replace(/\s*$/g, '\n');
    return out;
}

function setEditorReadOnly(readOnly) {
    if (!state.editor) return;
    state.editor.updateOptions({ readOnly });
}

function updateSolutionToggleButton(showingSolution) {
    if (!elements.toggleSolutionBtn) return;
    elements.toggleSolutionBtn.classList.toggle('active', showingSolution);
    elements.toggleSolutionBtn.setAttribute('aria-pressed', showingSolution ? 'true' : 'false');
    elements.toggleSolutionBtn.title = showingSolution ? 'Show starter code' : 'Show solution';
    elements.toggleSolutionBtn.setAttribute('aria-label', showingSolution ? 'Show starter code' : 'Show solution');
}

function setFullscreen(enabled) {
    state.editorFullscreen = enabled;
    if (enabled) {
        // Can't fullscreen both at once
        setQuestionFullscreen(false);
    }
    document.body.classList.toggle('editor-fullscreen', enabled);

    if (elements.fullscreenBtn) {
        elements.fullscreenBtn.classList.toggle('active', enabled);
        elements.fullscreenBtn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
        elements.fullscreenBtn.title = enabled ? 'Exit fullscreen' : 'Fullscreen';
        elements.fullscreenBtn.setAttribute('aria-label', enabled ? 'Exit fullscreen' : 'Fullscreen');
    }

    // Let layout settle, then re-layout Monaco
    setTimeout(() => {
        if (state.editor) state.editor.layout();
    }, 0);
}

function setQuestionFullscreen(enabled) {
    state.questionFullscreen = enabled;
    if (enabled) {
        // Can't fullscreen both at once
        setFullscreen(false);
    }
    document.body.classList.toggle('question-fullscreen', enabled);

    if (elements.questionFullscreenBtn) {
        elements.questionFullscreenBtn.classList.toggle('active', enabled);
        elements.questionFullscreenBtn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
        elements.questionFullscreenBtn.title = enabled ? 'Exit fullscreen question' : 'Fullscreen question';
        elements.questionFullscreenBtn.setAttribute(
            'aria-label',
            enabled ? 'Exit fullscreen question' : 'Fullscreen question'
        );
    }

    // Let layout settle, then re-layout Monaco (if visible)
    setTimeout(() => {
        if (state.editor) state.editor.layout();
    }, 0);
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderQuestionDescription() {
    if (!state.currentQuestion) return;
    elements.questionContent.innerHTML = state.currentQuestion.description_html;
    elements.questionContent.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

function renderQuestionSolution() {
    const html = state.solutionHtml;
    if (!html) {
        elements.questionContent.innerHTML =
            '<p>Written solution not available yet. Use the <strong>Show solution</strong> button in the editor panel for code.</p>';
        return;
    }

    elements.questionContent.innerHTML = html;
    elements.questionContent.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

async function setQuestionTab(tab) {
    const next = tab === 'solution' ? 'solution' : 'description';
    state.questionActiveTab = next;

    if (elements.tabDescription) {
        const isActive = next === 'description';
        elements.tabDescription.classList.toggle('active', isActive);
        elements.tabDescription.setAttribute('aria-selected', isActive ? 'true' : 'false');
    }
    if (elements.tabSolution) {
        const isActive = next === 'solution';
        elements.tabSolution.classList.toggle('active', isActive);
        elements.tabSolution.setAttribute('aria-selected', isActive ? 'true' : 'false');
    }

    if (next === 'description') {
        renderQuestionDescription();
        return;
    }

    if (!state.solutionCode) {
        await loadSolution();
    }
    renderQuestionSolution();
}

async function runEditorFormat() {
    if (!state.editor) return;
    if (state.showingSolution) {
        showToast('Switch to starter code to format.', 'warning');
        return;
    }

    const model = state.editor.getModel && state.editor.getModel();
    if (!model) return;

    const before = model.getValue();
    const slug = state.currentQuestion?.slug || 'unknown';
    console.log('[format] start', { slug, chars: before.length });

    // Preferred: server-side Ruff formatter (no persistence)
    try {
        const resp = await fetch('/api/format/python', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: before,
                filename: 'starting_point.py',
                question_slug: slug,
            }),
        });

        const text = await resp.text();
        let data = null;
        try {
            data = JSON.parse(text);
        } catch {
            // leave as null
        }

        console.log('[format] response', {
            ok: resp.ok,
            status: resp.status,
            used_ruff: data?.used_ruff,
            changed: data?.changed,
            error: data?.error,
            formatted_len: typeof data?.formatted_code === 'string' ? data.formatted_code.length : null,
        });

        if (resp.ok && data && typeof data.formatted_code === 'string') {
            const formatted = data.formatted_code;
            if (formatted !== before) {
                const whitespaceOnly =
                    normalizeInvisibleWhitespace(before) === normalizeInvisibleWhitespace(formatted);

                state.editor.pushUndoStop();
                state.editor.executeEdits('formatRuff', [
                    { range: model.getFullModelRange(), text: formatted },
                ]);
                state.editor.pushUndoStop();

                showToast(whitespaceOnly ? 'Formatted (whitespace only).' : 'Formatted (Ruff).', 'success');
                console.log('[format] applied', { slug, whitespaceOnly });
                return;
            }

            // Ruff ran but found nothing to change
            if (data.used_ruff) {
                showToast('No changes.', 'warning');
                console.log('[format] no changes (ruff)', { slug });
                return;
            }
        }

        // Ruff unavailable or error
        showToast('Formatter unavailable (ruff).', 'warning', 2600);
        console.warn('[format] ruff unavailable', { slug, body: data });
    } catch (e) {
        console.warn('[format] request failed', e);
        showToast('Formatter request failed.', 'warning', 2600);
    }

    // Fallback: basic cleanup so the button still has an effect
    const cleaned = basicWhitespaceCleanup(before);
    if (cleaned !== before) {
        const whitespaceOnly =
            normalizeInvisibleWhitespace(before) === normalizeInvisibleWhitespace(cleaned);
        state.editor.pushUndoStop();
        state.editor.executeEdits('formatCleanup', [
            { range: model.getFullModelRange(), text: cleaned },
        ]);
        state.editor.pushUndoStop();
        showToast(whitespaceOnly ? 'Formatted (whitespace only).' : 'Formatted (cleanup).', 'success');
        console.log('[format] applied fallback cleanup', { slug, whitespaceOnly });
        return;
    }

    showToast('No changes.', 'warning');
    console.log('[format] no changes (fallback)', { slug });
}

// ============================================
// Monaco Editor Setup
// ============================================

let monacoReady = false;

function loadMonaco() {
    return new Promise((resolve, reject) => {
        if (monacoReady) {
            resolve();
            return;
        }

        const waitForRequire = () => {
            if (typeof require !== 'undefined') {
                require.config({
                    paths: {
                        vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs'
                    }
                });

                require(['vs/editor/editor.main'], function () {
                    monaco.editor.defineTheme('mlbites-dark', {
                        base: 'vs-dark',
                        inherit: true,
                        rules: [
                            { token: 'comment', foreground: '6e7681', fontStyle: 'italic' },
                            { token: 'keyword', foreground: 'ff7b72' },
                            { token: 'string', foreground: 'a5d6ff' },
                            { token: 'number', foreground: '79c0ff' },
                            { token: 'type', foreground: 'ffa657' },
                            { token: 'function', foreground: 'd2a8ff' },
                        ],
                        colors: {
                            'editor.background': '#0d1117',
                            'editor.foreground': '#e6edf3',
                            'editor.lineHighlightBackground': '#161b22',
                            'editorLineNumber.foreground': '#6e7681',
                            'editorLineNumber.activeForeground': '#e6edf3',
                            'editor.selectionBackground': '#264f78',
                            'editor.inactiveSelectionBackground': '#264f7855',
                            'editorCursor.foreground': '#58a6ff',
                            'editorIndentGuide.background': '#21262d',
                            'editorIndentGuide.activeBackground': '#30363d',
                        }
                    });

                    monaco.editor.defineTheme('mlbites-light', {
                        base: 'vs',
                        inherit: true,
                        rules: [
                            { token: 'comment', foreground: '6e7781', fontStyle: 'italic' },
                            { token: 'keyword', foreground: 'cf222e' },
                            { token: 'string', foreground: '0a3069' },
                            { token: 'number', foreground: '0550ae' },
                            { token: 'type', foreground: '8250df' },
                            { token: 'function', foreground: '0969da' },
                        ],
                        colors: {
                            'editor.background': '#ffffff',
                            'editor.foreground': '#24292f',
                            'editor.lineHighlightBackground': '#f6f8fa',
                            'editorLineNumber.foreground': '#8c959f',
                            'editorLineNumber.activeForeground': '#24292f',
                            'editor.selectionBackground': '#add6ff',
                            'editor.inactiveSelectionBackground': '#add6ff66',
                            'editorCursor.foreground': '#0969da',
                            'editorIndentGuide.background': '#d0d7de',
                            'editorIndentGuide.activeBackground': '#8c959f',
                        }
                    });

                    monacoReady = true;
                    console.log('Monaco Editor loaded successfully');
                    resolve();
                });
            } else {
                setTimeout(waitForRequire, 100);
            }
        };
        waitForRequire();
    });
}

async function createEditor(code = '') {
    await loadMonaco();

    if (state.editor) {
        state.editor.dispose();
    }

    state.editor = monaco.editor.create(elements.editorContainer, {
        value: code,
        language: 'python',
        theme: document.documentElement.getAttribute('data-theme') === 'light' ? 'mlbites-light' : 'mlbites-dark',
        fontSize: 14,
        fontFamily: "'JetBrains Mono', monospace",
        lineNumbers: 'on',
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'on',
        padding: { top: 16, bottom: 16 },
        renderLineHighlight: 'line',
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        smoothScrolling: true,
        readOnly: false,
    });

    if (state.editorChangeDisposable) {
        try { state.editorChangeDisposable.dispose(); } catch { }
        state.editorChangeDisposable = null;
    }

    state.editorChangeDisposable = state.editor.onDidChangeModelContent(() => {
        if (!state.currentQuestion) return;
        if (state.showingSolution) return;
        if (state.isProgrammaticEdit) return;
        setDraft(state.currentQuestion.slug, state.editor.getValue());
    });

    return state.editor;
}

function updateEditorContent(code) {
    if (state.editor) {
        state.isProgrammaticEdit = true;
        state.editor.setValue(code);
        state.isProgrammaticEdit = false;
    }
}

// ============================================
// Sidebar Toggle
// ============================================

function toggleSidebar() {
    state.sidebarCollapsed = !state.sidebarCollapsed;

    if (state.sidebarCollapsed) {
        elements.sidebar.classList.add('collapsed');
        elements.sidebarToggleFloating.classList.add('visible');
    } else {
        elements.sidebar.classList.remove('collapsed');
        elements.sidebarToggleFloating.classList.remove('visible');
    }

    // Trigger Monaco Editor resize after transition
    setTimeout(() => {
        if (state.editor) {
            state.editor.layout();
        }
    }, 300);
}

// ============================================
// Resizable Panels
// ============================================

function initResizer() {
    if (!elements.resizer) return;

    let startX, startWidth;

    const onMouseDown = (e) => {
        e.preventDefault();
        state.isResizing = true;
        startX = e.clientX;
        startWidth = elements.questionPanel.offsetWidth;

        document.body.classList.add('no-select');
        elements.resizer.classList.add('resizing');

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    const onMouseMove = (e) => {
        if (!state.isResizing) return;

        const dx = e.clientX - startX;
        const newWidth = startWidth + dx;
        const containerWidth = elements.splitContainer.offsetWidth;

        // Constrain to min/max widths (20% - 70% of container)
        const minWidth = containerWidth * 0.2;
        const maxWidth = containerWidth * 0.7;

        if (newWidth >= minWidth && newWidth <= maxWidth) {
            elements.questionPanel.style.flex = `0 0 ${newWidth}px`;
        }
    };

    const onMouseUp = () => {
        state.isResizing = false;
        document.body.classList.remove('no-select');
        elements.resizer.classList.remove('resizing');

        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);

        // Trigger Monaco Editor resize
        if (state.editor) {
            state.editor.layout();
        }
    };

    elements.resizer.addEventListener('mousedown', onMouseDown);

    // Touch support
    elements.resizer.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        state.isResizing = true;
        startX = touch.clientX;
        startWidth = elements.questionPanel.offsetWidth;

        document.body.classList.add('no-select');
        elements.resizer.classList.add('resizing');
    });

    document.addEventListener('touchmove', (e) => {
        if (!state.isResizing) return;
        const touch = e.touches[0];

        const dx = touch.clientX - startX;
        const newWidth = startWidth + dx;
        const containerWidth = elements.splitContainer.offsetWidth;

        const minWidth = containerWidth * 0.2;
        const maxWidth = containerWidth * 0.7;

        if (newWidth >= minWidth && newWidth <= maxWidth) {
            elements.questionPanel.style.flex = `0 0 ${newWidth}px`;
        }
    });

    document.addEventListener('touchend', () => {
        if (state.isResizing) {
            state.isResizing = false;
            document.body.classList.remove('no-select');
            elements.resizer.classList.remove('resizing');

            if (state.editor) {
                state.editor.layout();
            }
        }
    });
}

function initTerminalResizer() {
    if (!elements.terminalResizer) return;
    if (!elements.terminalPanel) return;
    if (!elements.editorPanel) return;

    let startY = 0;
    let startHeight = 0;

    const minHeight = 80;
    const maxHeight = 800;

    const onMouseDown = (e) => {
        if (!state.terminalVisible) return;
        e.preventDefault();
        state.isTerminalResizing = true;
        startY = e.clientY;
        startHeight = elements.terminalPanel.offsetHeight || state.terminalHeightPx;

        document.body.classList.add('no-select');
        elements.terminalResizer.classList.add('resizing');

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    const onMouseMove = (e) => {
        if (!state.isTerminalResizing) return;
        const dy = e.clientY - startY;
        // Resizer is ABOVE the terminal panel:
        // - drag UP => increase terminal height
        // - drag DOWN => decrease terminal height
        const nextHeight = Math.max(minHeight, Math.min(maxHeight, startHeight - dy));
        state.terminalHeightPx = nextHeight;
        elements.terminalPanel.style.height = `${nextHeight}px`;
        if (state.editor) state.editor.layout();
    };

    const onMouseUp = () => {
        if (!state.isTerminalResizing) return;
        state.isTerminalResizing = false;
        document.body.classList.remove('no-select');
        elements.terminalResizer.classList.remove('resizing');

        saveTerminalHeight(state.terminalHeightPx);

        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    };

    elements.terminalResizer.addEventListener('mousedown', onMouseDown);

    // Touch support
    elements.terminalResizer.addEventListener('touchstart', (e) => {
        if (!state.terminalVisible) return;
        e.preventDefault();
        const touch = e.touches[0];
        state.isTerminalResizing = true;
        startY = touch.clientY;
        startHeight = elements.terminalPanel.offsetHeight || state.terminalHeightPx;
        document.body.classList.add('no-select');
        elements.terminalResizer.classList.add('resizing');
    });

    document.addEventListener('touchmove', (e) => {
        if (!state.isTerminalResizing) return;
        const touch = e.touches[0];
        const dy = touch.clientY - startY;
        const nextHeight = Math.max(minHeight, Math.min(maxHeight, startHeight - dy));
        state.terminalHeightPx = nextHeight;
        elements.terminalPanel.style.height = `${nextHeight}px`;
        if (state.editor) state.editor.layout();
    });

    document.addEventListener('touchend', () => {
        if (!state.isTerminalResizing) return;
        state.isTerminalResizing = false;
        document.body.classList.remove('no-select');
        elements.terminalResizer.classList.remove('resizing');
        saveTerminalHeight(state.terminalHeightPx);
    });
}

// ============================================
// Question Loading
// ============================================

async function loadQuestion(slug) {
    try {
        // Persist current draft before switching questions
        if (state.currentQuestion && state.editor && !state.showingSolution) {
            setDraft(state.currentQuestion.slug, state.editor.getValue());
        }

        const response = await fetch(`/api/questions/${slug}`);
        if (!response.ok) throw new Error('Failed to load question');

        const question = await response.json();
        state.currentQuestion = question;
        state.startingCode = question.starting_code;
        state.solutionCode = '';
        state.solutionHtml = '';
        state.showingSolution = false;

        displayQuestion(question);
        await setQuestionTab('description');

        elements.welcomeScreen.style.display = 'none';
        elements.splitContainer.style.display = 'flex';

        const draft = getDraft(question.slug);
        await createEditor(draft ?? question.starting_code);

        setEditorReadOnly(false);
        updateSolutionToggleButton(false);

        updateActiveQuestion(slug);

    } catch (error) {
        console.error('Error loading question:', error);
    }
}

function displayQuestion(question) {
    const difficultyClass = `difficulty-${question.difficulty.toLowerCase()}`;
    elements.questionMeta.innerHTML = `<span class="difficulty ${difficultyClass}">${question.difficulty}</span>`;

    // Render tags as accessible, clickable chips (button elements)
    elements.questionTags.innerHTML = '';
    const tagContainer = elements.questionTags.closest('.panel-header-tags');
    if (tagContainer) {
        tagContainer.style.display = question.tags?.length ? '' : 'none';
    }
    question.tags.forEach((tag) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'tag';
        btn.dataset.tag = tag;
        btn.setAttribute('aria-label', `Filter questions by tag: ${tag}`);
        btn.textContent = tag;
        elements.questionTags.appendChild(btn);
    });

    // Content is rendered by the active tab.
}

async function loadSolution() {
    if (!state.currentQuestion) return;

    try {
        const response = await fetch(`/api/questions/${state.currentQuestion.slug}/solution`);
        if (!response.ok) throw new Error('Failed to load solution');

        const data = await response.json();
        state.solutionCode = data.solution_code;
        state.solutionHtml = typeof data.solution_html === 'string' ? data.solution_html : '';

        return data.solution_code;
    } catch (error) {
        console.error('Error loading solution:', error);
        return null;
    }
}

function updateActiveQuestion(slug) {
    document.querySelectorAll('.question-item').forEach(item => {
        item.classList.remove('active');
    });

    const activeItem = document.querySelector(`.question-item[data-slug="${slug}"]`);
    if (activeItem) {
        activeItem.classList.add('active');
    }
}

// ============================================
// Search / Filter
// ============================================

function applySidebarFilters(searchTerm) {
    const searchLower = String(searchTerm || '').toLowerCase().trim();
    const questionItems = document.querySelectorAll('.question-item');
    // Sidebar groups are rendered server-side (currently grouped by difficulty).
    // We keep the existing `.category-group` class for styling and collapse behavior.
    const groups = document.querySelectorAll('.category-group');
    const activeFramework = normalizeFramework(state.activeFramework);

    questionItems.forEach(item => {
        const itemFramework = normalizeFramework(item.dataset.framework);
        const matchesFramework = itemFramework === activeFramework;

        // Search matches tags or title (same behavior as before)
        let matchesSearch = true;
        if (searchLower) {
            const tags = String(item.dataset.tags || '').toLowerCase();
            const title = String(item.querySelector('.question-title')?.textContent || '').toLowerCase();
            matchesSearch = tags.includes(searchLower) || title.includes(searchLower);
        }

        item.style.display = matchesFramework && matchesSearch ? '' : 'none';
    });

    groups.forEach(group => {
        const visibleItems = group.querySelectorAll('.question-item:not([style*="display: none"])');
        group.style.display = visibleItems.length > 0 ? '' : 'none';

        // Update the group count to reflect visible items under current filters
        const countEl = group.querySelector('.category-count');
        if (countEl) countEl.textContent = String(visibleItems.length);
    });
}

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    // Theme toggle (logo icon)
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', () => {
            toggleTheme();
        });
        elements.themeToggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleTheme();
            }
        });
    }

    // Group header click handlers (collapsible)
    document.querySelectorAll('.category-header').forEach(header => {
        header.addEventListener('click', (e) => {
            e.preventDefault();
            const categoryGroup = header.closest('.category-group');
            if (categoryGroup) {
                categoryGroup.classList.toggle('collapsed');
                const isExpanded = !categoryGroup.classList.contains('collapsed');
                header.setAttribute('aria-expanded', isExpanded.toString());
            }
        });
    });

    // Question click handlers
    document.querySelectorAll('.question-item').forEach(item => {
        item.addEventListener('click', () => {
            const slug = item.dataset.slug;
            loadQuestion(slug);
        });
    });

    // Editor toolbar: toggle solution
    if (elements.toggleSolutionBtn) {
        elements.toggleSolutionBtn.addEventListener('click', async () => {
            if (!state.currentQuestion) return;
            const slug = state.currentQuestion.slug;

            if (state.showingSolution) {
                const draft = getDraft(slug);
                updateEditorContent(draft ?? state.startingCode);
                state.showingSolution = false;
                setEditorReadOnly(false);
                updateSolutionToggleButton(false);
                return;
            }

            // Save current in-progress code before showing solution
            if (state.editor) {
                setDraft(slug, state.editor.getValue());
            }

            if (!state.solutionCode) {
                await loadSolution();
            }
            if (state.solutionCode) {
                updateEditorContent(state.solutionCode);
                state.showingSolution = true;
                setEditorReadOnly(true);
                updateSolutionToggleButton(true);
            }
        });
    }

    // Editor toolbar: reset
    if (elements.resetCodeBtn) {
        elements.resetCodeBtn.addEventListener('click', () => {
            if (!state.currentQuestion) return;
            const slug = state.currentQuestion.slug;
            const ok = window.confirm(
                'Reset your code to the default starter code?\n\nThis will discard your current edits for this question.'
            );
            if (!ok) {
                console.log('[reset] cancelled', { slug });
                return;
            }
            setDraft(slug, state.startingCode);
            updateEditorContent(state.startingCode);
            state.showingSolution = false;
            setEditorReadOnly(false);
            updateSolutionToggleButton(false);
            showToast('Reset to default.', 'success');
            console.log('[reset] applied', { slug });
        });
    }

    // Editor toolbar: format
    if (elements.formatCodeBtn) {
        elements.formatCodeBtn.addEventListener('click', () => {
            runEditorFormat();
        });
    }

    // Editor toolbar: fullscreen
    if (elements.fullscreenBtn) {
        elements.fullscreenBtn.addEventListener('click', () => {
            setFullscreen(!state.editorFullscreen);
        });
    }

    // Editor toolbar: run tests
    if (elements.runCodeBtn) {
        elements.runCodeBtn.addEventListener('click', () => {
            runCurrentCode();
        });
    }

    // Editor toolbar: toggle output panel
    if (elements.toggleTerminalBtn) {
        elements.toggleTerminalBtn.addEventListener('click', () => {
            setTerminalVisible(!state.terminalVisible);
        });
    }

    if (elements.clearTerminalBtn) {
        elements.clearTerminalBtn.addEventListener('click', () => {
            clearTerminal();
        });
    }

    // Question panel: tabs
    if (elements.tabDescription) {
        elements.tabDescription.addEventListener('click', () => {
            setQuestionTab('description');
        });
    }
    if (elements.tabSolution) {
        elements.tabSolution.addEventListener('click', () => {
            setQuestionTab('solution');
        });
    }

    // Question panel: fullscreen
    if (elements.questionFullscreenBtn) {
        elements.questionFullscreenBtn.addEventListener('click', () => {
            setQuestionFullscreen(!state.questionFullscreen);
        });
    }

    // Escape exits editor fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (state.editorFullscreen) setFullscreen(false);
        if (state.questionFullscreen) setQuestionFullscreen(false);
    });

    // Search input
    elements.searchInput.addEventListener('input', (e) => {
        applySidebarFilters(e.target.value);
    });

    elements.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            elements.searchInput.value = '';
            applySidebarFilters('');
        }
    });

    // Tag chip click -> filter sidebar search
    if (elements.questionTags) {
        elements.questionTags.addEventListener('click', (e) => {
            const btn = e.target?.closest?.('button.tag');
            const tag = btn?.dataset?.tag;
            if (!tag) return;
            if (!elements.searchInput) return;
            elements.searchInput.value = tag;
            applySidebarFilters(tag);
            showToast(`Filtered by tag: ${tag}`, 'success');
        });
    }

    // Framework toggle (PyTorch / NumPy)
    if (elements.frameworkBtnPytorch) {
        elements.frameworkBtnPytorch.addEventListener('click', () => {
            setFrameworkFilter('pytorch');
        });
    }
    if (elements.frameworkBtnNumpy) {
        elements.frameworkBtnNumpy.addEventListener('click', () => {
            setFrameworkFilter('numpy');
        });
    }

    // Sidebar toggle
    if (elements.sidebarToggle) {
        elements.sidebarToggle.addEventListener('click', toggleSidebar);
    }
    if (elements.sidebarToggleFloating) {
        elements.sidebarToggleFloating.addEventListener('click', toggleSidebar);
    }
}

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    loadDraftsFromStorage();
    loadTerminalPrefs();
    loadFrameworkFromStorage();
    initTheme();
    setupEventListeners();
    initResizer();
    initTerminalResizer();

    // Initialize toolbar state
    updateSolutionToggleButton(false);
    setFullscreen(false);
    setTerminalVisible(false, { force: true });
    clearTerminal();

    // Apply initial sidebar filters (framework + any prefilled search)
    setFrameworkFilter(state.activeFramework);

    // Pre-load Monaco Editor
    loadMonaco().then(() => {
        console.log('Monaco pre-loaded');
        // Ensure Monaco theme matches current UI theme
        const theme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
        applyTheme(theme);
    });
});
