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
    isResizing: false,
    sidebarCollapsed: false,
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
    questionContent: document.getElementById('question-content'),
    editorContainer: document.getElementById('editor-container'),
    fileIndicator: document.getElementById('file-indicator'),
    revealBtn: document.getElementById('reveal-btn'),
    revealBtnText: document.getElementById('reveal-btn-text'),
    searchInput: document.getElementById('search-input'),
    questionList: document.getElementById('question-list'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    sidebarToggleFloating: document.getElementById('sidebar-toggle-floating'),
    resizer: document.getElementById('resizer'),
};

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

    return state.editor;
}

function updateEditorContent(code) {
    if (state.editor) {
        state.editor.setValue(code);
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

        await createEditor(question.starting_code);

        updateRevealButton(false);
        elements.fileIndicator.textContent = 'starting_point.py';

        updateActiveQuestion(slug);

    } catch (error) {
        console.error('Error loading question:', error);
    }
}

function displayQuestion(question) {
    elements.questionTitle.textContent = question.title;

    const difficultyClass = `difficulty-${question.difficulty.toLowerCase()}`;
    const tagsHtml = question.tags.map(tag => `<span class="tag">${tag}</span>`).join('');

    elements.questionMeta.innerHTML = `
        <span class="difficulty ${difficultyClass}">${question.difficulty}</span>
        <div class="tags">${tagsHtml}</div>
    `;

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

// ============================================
// UI Updates
// ============================================

function updateActiveQuestion(slug) {
    document.querySelectorAll('.question-item').forEach(item => {
        item.classList.remove('active');
    });

    const activeItem = document.querySelector(`.question-item[data-slug="${slug}"]`);
    if (activeItem) {
        activeItem.classList.add('active');
    }
}

function updateRevealButton(showingSolution) {
    if (showingSolution) {
        elements.revealBtn.classList.add('active');
        elements.revealBtnText.textContent = 'Show Starter Code';
        elements.fileIndicator.textContent = 'solution.py';
    } else {
        elements.revealBtn.classList.remove('active');
        elements.revealBtnText.textContent = 'Reveal Solution';
        elements.fileIndicator.textContent = 'starting_point.py';
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

    // Reveal solution button
    elements.revealBtn.addEventListener('click', async () => {
        if (state.showingSolution) {
            updateEditorContent(state.startingCode);
            state.showingSolution = false;
            updateRevealButton(false);
        } else {
            if (!state.solutionCode) {
                await loadSolution();
            }
            if (state.solutionCode) {
                updateEditorContent(state.solutionCode);
                state.showingSolution = true;
                updateRevealButton(true);
            }
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
    setupEventListeners();
    initResizer();

    // Pre-load Monaco Editor
    loadMonaco().then(() => {
        console.log('Monaco pre-loaded');
    });
});
