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
    editor: null,
    editorChangeDisposable: null,
    isResizing: false,
    sidebarCollapsed: false,
    editorFullscreen: false,
    isProgrammaticEdit: false,
    drafts: {},
};

// ============================================
// DOM Elements
// ============================================

const elements = {
    welcomeScreen: document.getElementById('welcome-screen'),
    splitContainer: document.getElementById('split-container'),
    questionPanel: document.getElementById('question-panel'),
    editorPanel: document.getElementById('editor-panel'),
    questionTitle: document.getElementById('question-title'),
    questionMeta: document.getElementById('question-meta'),
    questionTags: document.getElementById('question-tags'),
    questionContent: document.getElementById('question-content'),
    editorContainer: document.getElementById('editor-container'),
    toggleSolutionBtn: document.getElementById('toggle-solution-btn'),
    resetCodeBtn: document.getElementById('reset-code-btn'),
    formatCodeBtn: document.getElementById('format-code-btn'),
    fullscreenBtn: document.getElementById('fullscreen-btn'),
    searchInput: document.getElementById('search-input'),
    questionList: document.getElementById('question-list'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    sidebarToggleFloating: document.getElementById('sidebar-toggle-floating'),
    resizer: document.getElementById('resizer'),
    toast: document.getElementById('toast'),
};

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
        theme: 'mlbites-dark',
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
        state.showingSolution = false;

        displayQuestion(question);

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
    elements.questionTitle.textContent = question.title;

    const difficultyClass = `difficulty-${question.difficulty.toLowerCase()}`;
    const tagsHtml = question.tags.map(tag => `<span class="tag">${tag}</span>`).join('');

    elements.questionMeta.innerHTML = `<span class="difficulty ${difficultyClass}">${question.difficulty}</span>`;
    elements.questionTags.innerHTML = tagsHtml;

    elements.questionContent.innerHTML = question.description_html;

    elements.questionContent.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

async function loadSolution() {
    if (!state.currentQuestion) return;

    try {
        const response = await fetch(`/api/questions/${state.currentQuestion.slug}/solution`);
        if (!response.ok) throw new Error('Failed to load solution');

        const data = await response.json();
        state.solutionCode = data.solution_code;

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

function filterQuestions(searchTerm) {
    const searchLower = searchTerm.toLowerCase().trim();
    const questionItems = document.querySelectorAll('.question-item');
    const categoryGroups = document.querySelectorAll('.category-group');

    if (!searchLower) {
        questionItems.forEach(item => item.style.display = '');
        categoryGroups.forEach(group => group.style.display = '');
        return;
    }

    questionItems.forEach(item => {
        const tags = item.dataset.tags.toLowerCase();
        const title = item.querySelector('.question-title').textContent.toLowerCase();

        if (tags.includes(searchLower) || title.includes(searchLower)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });

    categoryGroups.forEach(group => {
        const visibleItems = group.querySelectorAll('.question-item:not([style*="display: none"])');
        group.style.display = visibleItems.length > 0 ? '' : 'none';
    });
}

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    // Category header click handlers (collapsible)
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

    // Escape exits editor fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.editorFullscreen) {
            setFullscreen(false);
        }
    });

    // Search input
    elements.searchInput.addEventListener('input', (e) => {
        filterQuestions(e.target.value);
    });

    elements.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            elements.searchInput.value = '';
            filterQuestions('');
        }
    });

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
    setupEventListeners();
    initResizer();

    // Initialize toolbar state
    updateSolutionToggleButton(false);
    setFullscreen(false);

    // Pre-load Monaco Editor
    loadMonaco().then(() => {
        console.log('Monaco pre-loaded');
    });
});
