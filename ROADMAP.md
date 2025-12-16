# Dialeng Roadmap 🗺️

This document outlines the planned features and development direction for Dialeng.

## Current Status: v0.3.1

✅ **Completed in v0.1.0 (MVP)**
- Three cell types: Code, Note, Prompt
- Editable AI responses in prompt cells
- Python kernel with persistent namespace
- Solveit-compatible .ipynb serialization
- HTMX reactive UI
- Basic keyboard shortcuts
- Mock LLM for testing
- File-based notebook storage

✅ **Completed in v0.2.0**
- Ace Editor with syntax highlighting (Monokai theme)
- Jupyter-style output for last expression in code cells
- Auto-select next cell after execution (all cell types)
- Shift+Enter on note cells moves to next cell
- Markdown preview for user prompts (after running)
- Cell selection on click (anywhere on cell)
- Efficient partial re-rendering (only affected cells)
- Fixed Ace Editor positioning and syntax highlighting bugs

✅ **Completed in v0.3.0**
- Dark/light theme toggle with localStorage persistence
- Mobile-responsive layout (tablet 768px, mobile 480px breakpoints)
- Cell folding/collapsing with visual feedback
- WebSocket real-time token streaming for AI responses
- Cancel generation mid-stream (⏹ button)
- 🧠 Thinking mode indicator during AI processing

✅ **Completed in v0.3.1**
- Multi-level collapsing for input/output sections independently
- Three collapse levels: Expanded (▼), Scrollable (◐), Summary (▬)
- Keyboard shortcuts for collapsing: Z (input), Shift+Z (output), Alt+Z (both)
- Code cells default to scrollable output mode
- Persistent collapse state saved to notebook metadata

---

## Phase 1: Core Improvements (v0.2.0)

### 🔌 Real LLM Integration
- [ ] **Anthropic Claude** - Full integration with streaming
- [ ] **OpenAI GPT** - ChatGPT and GPT-4 support
- [ ] **Ollama** - Local model support
- [ ] **Provider selector** - Switch providers per-notebook or per-cell
- [ ] **API key management** - Secure storage, environment variables

### 📡 WebSocket Streaming
- [x] Real-time token streaming for AI responses ✅ v0.3.0
- [x] Progress indicators during generation ✅ v0.3.0
- [x] Cancel generation mid-stream ✅ v0.3.0
- [ ] Typing indicators

### 💾 Enhanced Persistence
- [ ] Auto-save with configurable interval
- [ ] Undo/redo history
- [ ] Version history for notebooks
- [ ] SQLite backend option (FastLite)
- [ ] Export to multiple formats (.py, .md, .html)

### 🎨 UI Polish
- [x] Syntax highlighting (Ace Editor with Monokai theme) ✅ v0.2.0
- [x] Better markdown preview (full CommonMark support) ✅ v0.2.0
- [x] Cell folding/collapsing ✅ v0.3.0
- [x] Dark/light theme toggle ✅ v0.3.0
- [x] Mobile-responsive layout ✅ v0.3.0
- [x] Multi-level section collapse (input/output) ✅ v0.3.1

---

## Phase 2: Advanced Features (v0.3.0)

### 🤖 AI Enhancements
- [ ] **Context management** - Pin/hide cells from AI context
- [ ] **System prompts** - Custom instructions per notebook
- [ ] **Dialog modes** - Learning, Concise, Code-focused
- [ ] **Tool use** - Let AI execute code, search web, etc.
- [ ] **Multi-turn context** - Conversation history within prompt cells
- [x] **Thinking mode** - Extended thinking for complex problems ✅ v0.3.0

### 📊 Rich Outputs
- [ ] **Matplotlib/Plotly** - Inline chart rendering
- [ ] **DataFrames** - Pretty table display for pandas
- [ ] **Images** - Display PIL/base64 images
- [ ] **HTML widgets** - Interactive outputs
- [ ] **LaTeX** - Math equation rendering

### 🔧 Kernel Improvements
- [ ] **Multiple kernels** - Python, JavaScript, SQL, Shell
- [ ] **Kernel status** - Busy/idle indicator
- [ ] **Interrupt execution** - Stop long-running code
- [ ] **Environment management** - Virtual env support
- [ ] **Package installation** - pip install from notebook

### 📁 File Management
- [ ] **File browser** - Navigate and open notebooks
- [ ] **Drag-and-drop** - Upload files to notebook
- [ ] **File attachments** - Attach files to cells
- [ ] **Import from URL** - Load .ipynb from URL

---

## Phase 3: Collaboration (v0.4.0)

### 👥 Real-time Collaboration
- [ ] **Live cursors** - See collaborators editing
- [ ] **Presence indicators** - Who's viewing the notebook
- [ ] **Conflict resolution** - Handle simultaneous edits
- [ ] **Comments** - Inline comments on cells
- [ ] **Sharing** - Generate shareable links

### 🔐 Authentication
- [ ] **User accounts** - Sign up / sign in
- [ ] **OAuth** - Google, GitHub, etc.
- [ ] **Permissions** - Owner, editor, viewer roles
- [ ] **Private notebooks** - User-specific storage

### ☁️ Cloud Features
- [ ] **Cloud storage** - S3, GCS, Azure Blob
- [ ] **Cloud compute** - Remote kernel execution
- [ ] **Scheduled runs** - Cron-like notebook execution
- [ ] **Webhooks** - Trigger notebooks from events

---

## Phase 4: Platform Features (v1.0.0)

### 📚 Module System (Solveit parity)
- [ ] **Export to .py** - Convert notebook to Python module
- [ ] **Import notebooks** - Use other notebooks as modules
- [ ] **Dependency management** - Track notebook dependencies
- [ ] **Package publishing** - Publish to PyPI from notebook

### 🔍 Search & Discovery
- [ ] **Full-text search** - Search across all notebooks
- [ ] **Tags** - Organize notebooks with tags
- [ ] **Templates** - Start from predefined templates
- [ ] **Gallery** - Share and discover public notebooks

### 📈 Analytics & Monitoring
- [ ] **Usage tracking** - Token usage, execution time
- [ ] **Cost estimation** - LLM API cost tracking
- [ ] **Error reporting** - Aggregated error logs
- [ ] **Performance metrics** - Cell execution times

### 🔌 Integrations
- [ ] **VS Code extension** - Edit notebooks in VS Code
- [ ] **JupyterLab** - Use alongside traditional notebooks
- [ ] **GitHub integration** - Sync with repos
- [ ] **Slack/Discord** - Notifications and sharing

---

## Experimental Ideas 🧪

These are speculative features that may or may not be implemented:

### 🧠 Advanced AI Features
- **Agent mode** - AI can autonomously run multiple cells
- **Code review** - AI reviews and suggests improvements
- **Test generation** - Auto-generate tests for code
- **Documentation** - Auto-generate docstrings and README
- **Refactoring** - AI-assisted code refactoring

### 🎮 Interactive Features
- **Voice input** - Dictate prompts
- **Voice output** - AI reads responses
- **Drawing canvas** - Sketch diagrams
- **Whiteboard mode** - Freeform notes + cells

### 🔬 Data Science
- **Dataset browser** - Explore data visually
- **Auto-EDA** - Automatic exploratory data analysis
- **Model training UI** - Visual ML pipeline builder
- **Experiment tracking** - MLflow-like tracking

### 🌐 Deployment
- **One-click deploy** - Deploy notebook as API/app
- **Gradio/Streamlit export** - Convert to interactive app
- **Docker export** - Package notebook as container
- **Serverless functions** - Deploy cells as functions

---

## Contributing

We welcome contributions! Here's how you can help:

### Good First Issues
- [x] Add syntax highlighting ✅ v0.2.0
- [x] Improve mobile responsiveness ✅ v0.3.0
- [x] Add more keyboard shortcuts ✅ v0.2.0, v0.3.1
- [ ] Write documentation
- [ ] Add unit tests

### Medium Complexity
- [ ] Implement a new LLM provider
- [ ] Add a new cell type
- [ ] Implement undo/redo
- [x] Add theme support ✅ v0.3.0

### Advanced
- [x] WebSocket streaming implementation ✅ v0.3.0
- [ ] Real-time collaboration
- [ ] Cloud storage backend
- [ ] Multi-kernel support

### How to Contribute

1. Check the [Issues](https://github.com/your-repo/llm-notebook/issues) for tasks
2. Comment on an issue to claim it
3. Fork and create a feature branch
4. Submit a PR with your changes
5. Respond to review feedback

See [DEVELOPERS.md](DEVELOPERS.md) for technical guidance.

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.1.0 | 2024-XX | Initial MVP with core features |
| v0.2.0 | 2024-12-09 | Ace Editor, Jupyter-style output, cell navigation, UI fixes |
| v0.3.0 | 2024-12-09 | Theme toggle, mobile responsive, cell folding, WebSocket streaming, cancel generation |
| v0.3.1 | 2024-12-09 | Multi-level input/output collapsing, keyboard shortcuts for collapse |
| v0.4.0 | TBD | Real LLM integration, context management, rich outputs |
| v0.5.0 | TBD | Collaboration features |
| v1.0.0 | TBD | Full platform with module system |

---

## Comparison: Solveit Features

Tracking parity with [Solveit](https://solve.it.com/):

| Feature | Solveit | Dialeng | Status |
|---------|---------|--------------|--------|
| Code cells | ✅ | ✅ | Done |
| Note cells | ✅ | ✅ | Done |
| Prompt cells | ✅ | ✅ | Done |
| Editable AI responses | ✅ | ✅ | Done |
| .ipynb format | ✅ | ✅ | Done |
| Context management | ✅ | ⬜ | Planned v0.4 |
| WebSocket streaming | ✅ | ✅ | Done v0.3.0 |
| Cancel generation | ✅ | ✅ | Done v0.3.0 |
| Dark/light theme | ✅ | ✅ | Done v0.3.0 |
| Cell collapsing | ✅ | ✅ | Done v0.3.1 |
| Tool use | ✅ | ⬜ | Planned v0.4 |
| Real-time collaboration | ✅ | ⬜ | Planned v0.5 |
| Module export | ✅ | ⬜ | Planned v1.0 |
| Cloud sync | ✅ | ⬜ | Planned v0.5 |
| FastHTML support | ✅ | ✅ | Built with it! |
| Open source | ❌ | ✅ | **Advantage!** |

---

## Feedback

Have ideas for features? Found a bug?

- Open an [Issue](https://github.com/your-repo/llm-notebook/issues)
- Start a [Discussion](https://github.com/your-repo/llm-notebook/discussions)
- Submit a PR with your implementation

This roadmap is a living document and will evolve based on community feedback!
