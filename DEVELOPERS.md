# LLM Notebook Developer Guide 🛠️

A comprehensive guide for developers who want to extend, customize, or contribute to LLM Notebook.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Core Concepts](#core-concepts)
4. [Extension Points](#extension-points)
5. [Adding New Cell Types](#adding-new-cell-types)
6. [Integrating Real LLM Providers](#integrating-real-llm-providers)
7. [Adding Persistence Backends](#adding-persistence-backends)
8. [UI Customization](#ui-customization)
9. [Testing](#testing)
10. [Common Patterns](#common-patterns)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastHTML App                              │
├─────────────────────────────────────────────────────────────────┤
│  Routes          │  Components       │  Services                 │
│  ───────         │  ──────────       │  ────────                 │
│  /notebook/{id}  │  NotebookPage()   │  PythonKernel            │
│  /cell/{id}/run  │  CellView()       │  LLM Provider            │
│  /cell/add       │  AddButtons()     │  Storage (ipynb)         │
│  ...             │  TypeSelect()     │  Context Builder         │
├─────────────────────────────────────────────────────────────────┤
│                     Data Models                                  │
│  ─────────────────────────────────────────────────────────────  │
│  Cell (code/note/prompt) ←→ Jupyter Cell (ipynb format)         │
│  Notebook ←→ ipynb file                                         │
├─────────────────────────────────────────────────────────────────┤
│                     Frontend (HTMX)                              │
│  ─────────────────────────────────────────────────────────────  │
│  hx-post, hx-target, hx-swap for reactive updates               │
│  Minimal JavaScript for keyboard shortcuts & markdown preview   │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
User Action → HTMX Request → FastHTML Route → Update Model → Return HTML Fragment → HTMX Swaps DOM
```

Example: Running a code cell
```
Click ▶ → POST /cell/{id}/run → kernel.execute() → Update cell.output → Return CellView() → Replace #cell-{id}
```

---

## Project Structure

```
llm_notebook/
├── app.py              # Main application (all-in-one)
├── requirements.txt    # Dependencies
├── README.md          # User documentation
├── DEVELOPERS.md      # This file
├── ROADMAP.md         # Future plans
└── notebooks/         # Saved notebooks (created at runtime)
    └── *.ipynb
```

### Suggested Structure for Larger Projects

```
llm_notebook/
├── app.py              # FastHTML app setup & routes
├── models.py           # Cell, Notebook dataclasses
├── kernel.py           # PythonKernel class
├── llm/
│   ├── __init__.py
│   ├── base.py         # Abstract LLM interface
│   ├── mock.py         # Mock provider
│   ├── anthropic.py    # Claude integration
│   └── openai.py       # OpenAI integration
├── storage/
│   ├── __init__.py
│   ├── ipynb.py        # ipynb serialization
│   └── database.py     # Optional DB backend
├── components.py       # UI components
├── static/
│   ├── style.css
│   └── script.js
└── tests/
    ├── test_models.py
    ├── test_kernel.py
    └── test_serialization.py
```

---

## Core Concepts

### 1. The Cell Model

```python
@dataclass
class Cell:
    id: str                    # Unique identifier (used in ipynb)
    cell_type: str             # "code" | "note" | "prompt"
    source: str                # Main content (code/markdown/user prompt)
    output: str                # Execution result / AI response
    execution_count: int       # For code cells
    
    # Solveit metadata
    time_run: str              # When cell was last run
    skipped: bool              # Cell marked as skipped
    use_thinking: bool         # For prompt cells: use thinking mode
    collapsed: bool            # UI state
    pinned: bool               # Keep in context even if hidden
```

### 2. The Solveit Separator

Prompt cells store both user input and AI response in a single markdown cell:

```python
SEPARATOR_PATTERN = re.compile(r'##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_([a-f0-9]+) -->')

def split_prompt_content(content: str) -> tuple[str, str]:
    """Split combined content into (user_prompt, ai_response)"""
    match = SEPARATOR_PATTERN.search(content)
    if match:
        user_prompt = content[:match.start()].strip()
        ai_response = content[match.end():].strip()
        return user_prompt, ai_response
    return content.strip(), ""

def join_prompt_content(user_prompt: str, ai_response: str) -> str:
    """Combine user prompt and AI response with separator"""
    sep_id = uuid.uuid4().hex[:8]
    separator = f"##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_{sep_id} -->"
    return f"{user_prompt}\n\n{separator}\n\n{ai_response}"
```

### 3. HTMX Patterns

The app uses HTMX for reactive updates without a JavaScript framework:

```python
# Button that updates a specific cell
Button("▶", 
    hx_post=f"/cell/{cell.id}/run",      # POST to this endpoint
    hx_target=f"#cell-{cell.id}",         # Replace this element
    hx_swap="outerHTML"                   # Replace entire element
)

# Input that saves on blur
Textarea(cell.source,
    hx_post=f"/cell/{cell.id}/source",
    hx_trigger="blur changed",            # Only POST if changed
    hx_swap="none"                        # Don't update DOM
)
```

---

## Extension Points

### 1. LLM Provider Interface

Create an abstract base class for LLM providers:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def stream(
        self, 
        prompt: str, 
        context: str,
        system_prompt: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier"""
        pass
```

### 2. Storage Backend Interface

```python
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """Base class for notebook storage"""
    
    @abstractmethod
    def save(self, notebook: Notebook) -> None:
        pass
    
    @abstractmethod
    def load(self, notebook_id: str) -> Notebook:
        pass
    
    @abstractmethod
    def list(self) -> list[str]:
        pass
    
    @abstractmethod
    def delete(self, notebook_id: str) -> None:
        pass
```

### 3. Cell Type Registry

```python
# Registry for cell types
CELL_TYPES: dict[str, type] = {}

def register_cell_type(name: str):
    """Decorator to register new cell types"""
    def decorator(cls):
        CELL_TYPES[name] = cls
        return cls
    return decorator

@register_cell_type("code")
class CodeCell(Cell):
    def render(self) -> FT:
        # Return FastHTML components
        pass
    
    def execute(self, kernel: PythonKernel) -> str:
        # Execute and return output
        pass
```

---

## Adding New Cell Types

### Example: SQL Cell

```python
# 1. Add to CellType enum
class CellType(str, Enum):
    CODE = "code"
    NOTE = "note"
    PROMPT = "prompt"
    SQL = "sql"  # New!

# 2. Add rendering in CellView()
def CellView(cell: Cell, notebook_id: str = "default"):
    # ... existing code ...
    
    if cell.cell_type == "sql":
        body = Div(
            Textarea(cell.source, cls="source sql-source",
                    placeholder="-- Enter SQL query...",
                    hx_post=f"/cell/{cell.id}/source",
                    hx_trigger="blur changed",
                    hx_swap="none"),
            # Results table
            Div(render_sql_results(cell.output), cls="sql-results")
                if cell.output else None,
            cls="cell-body"
        )

# 3. Add execution logic in /cell/{id}/run
@rt("/cell/{cid}/run")
async def post(cid: str):
    # ... existing code ...
    
    if c.cell_type == "sql":
        try:
            import sqlite3
            conn = sqlite3.connect(":memory:")  # Or real DB
            cursor = conn.execute(c.source)
            results = cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            c.output = json.dumps({"columns": columns, "rows": results})
        except Exception as e:
            c.output = f"Error: {e}"

# 4. Add ipynb serialization in Cell.to_jupyter_cell()
def to_jupyter_cell(self) -> Dict[str, Any]:
    if self.cell_type == "sql":
        return {
            "cell_type": "code",  # Store as code cell
            "id": self.id,
            "metadata": {
                "language": "sql",  # Mark as SQL
                "llm_notebook_type": "sql"
            },
            "source": self._to_source_lines(self.source),
            "outputs": self._format_outputs(self.output)
        }
```

### Example: Image Generation Cell

```python
class CellType(str, Enum):
    # ... existing ...
    IMAGE = "image"

# In CellView
if cell.cell_type == "image":
    body = Div(
        Textarea(cell.source, cls="source",
                placeholder="Describe the image to generate...",
                hx_post=f"/cell/{cell.id}/source",
                hx_trigger="blur changed"),
        Img(src=cell.output, cls="generated-image") if cell.output else None,
        cls="cell-body"
    )

# In /cell/{id}/run
if c.cell_type == "image":
    # Call image generation API
    image_url = await generate_image(c.source)
    c.output = image_url
```

---

## Integrating Real LLM Providers

### Anthropic (Claude)

```python
import anthropic

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    async def stream(
        self, 
        prompt: str, 
        context: str,
        system_prompt: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        
        if context:
            messages.insert(0, {
                "role": "user", 
                "content": f"Context from notebook:\n\n{context}"
            })
            messages.insert(1, {
                "role": "assistant",
                "content": "I understand. I can see the notebook context."
            })
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "You are a helpful coding assistant.",
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def get_model_name(self) -> str:
        return self.model
```

### OpenAI

```python
from openai import AsyncOpenAI

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def stream(
        self, 
        prompt: str, 
        context: str,
        system_prompt: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({
                "role": "system", 
                "content": f"Notebook context:\n{context}"
            })
        
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def get_model_name(self) -> str:
        return self.model
```

### Ollama (Local)

```python
import httpx

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
    
    async def stream(
        self, 
        prompt: str, 
        context: str,
        **kwargs
    ) -> AsyncIterator[str]:
        full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": full_prompt},
                timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
    
    def get_model_name(self) -> str:
        return f"ollama/{self.model}"
```

### Provider Factory

```python
def get_llm_provider(provider: str = "mock", **kwargs) -> LLMProvider:
    """Factory function to get LLM provider"""
    providers = {
        "mock": MockProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    
    return providers[provider](**kwargs)

# Usage in app
import os
llm = get_llm_provider(
    os.getenv("LLM_PROVIDER", "mock"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL")
)
```

---

## Adding Persistence Backends

### SQLite with FastLite

```python
from fastlite import database

class SQLiteStorage(StorageBackend):
    def __init__(self, db_path: str = "notebooks.db"):
        self.db = database(db_path)
        self.notebooks = self.db.t.notebooks
        if not self.notebooks:
            self.notebooks.create(
                id=str, 
                title=str, 
                content=str,  # JSON
                updated_at=str,
                pk='id'
            )
    
    def save(self, notebook: Notebook) -> None:
        data = {
            "id": notebook.id,
            "title": notebook.title,
            "content": json.dumps(notebook.to_ipynb()),
            "updated_at": datetime.now().isoformat()
        }
        self.notebooks.upsert(data)
    
    def load(self, notebook_id: str) -> Notebook:
        row = self.notebooks[notebook_id]
        if not row:
            raise KeyError(f"Notebook not found: {notebook_id}")
        data = json.loads(row.content)
        return Notebook.from_ipynb(data, notebook_id)
    
    def list(self) -> list[str]:
        return [row.id for row in self.notebooks()]
    
    def delete(self, notebook_id: str) -> None:
        self.notebooks.delete(notebook_id)
```

### S3 Storage

```python
import boto3

class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = "notebooks/"):
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix
    
    def save(self, notebook: Notebook) -> None:
        key = f"{self.prefix}{notebook.id}.ipynb"
        content = json.dumps(notebook.to_ipynb())
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=content)
    
    def load(self, notebook_id: str) -> Notebook:
        key = f"{self.prefix}{notebook_id}.ipynb"
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        data = json.loads(response["Body"].read())
        return Notebook.from_ipynb(data, notebook_id)
    
    def list(self) -> list[str]:
        response = self.s3.list_objects_v2(
            Bucket=self.bucket, 
            Prefix=self.prefix
        )
        return [
            obj["Key"].replace(self.prefix, "").replace(".ipynb", "")
            for obj in response.get("Contents", [])
        ]
```

---

## UI Customization

### Adding Themes

```python
# Define theme CSS variables
THEMES = {
    "dark": """
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --text-primary: #c9d1d9;
            --accent-blue: #58a6ff;
        }
    """,
    "light": """
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --text-primary: #24292f;
            --accent-blue: #0969da;
        }
    """,
    "solarized": """
        :root {
            --bg-primary: #002b36;
            --bg-secondary: #073642;
            --text-primary: #839496;
            --accent-blue: #268bd2;
        }
    """
}

# Add theme selector to toolbar
Select(
    *[Option(name.title(), value=name) for name in THEMES],
    hx_post="/settings/theme",
    hx_target="style#theme-vars"
)
```

### Custom Components

```python
# Create reusable components
def IconButton(icon: str, **kwargs):
    """Standardized icon button"""
    return Button(icon, cls="btn btn-icon", **kwargs)

def Badge(text: str, color: str = "blue"):
    """Colored badge component"""
    return Span(text, cls=f"badge badge-{color}")

def Card(title: str, *content, **kwargs):
    """Card with header"""
    return Div(
        Div(title, cls="card-header"),
        Div(*content, cls="card-body"),
        cls="card",
        **kwargs
    )
```

### Adding Keyboard Shortcuts

```javascript
// In the JS section
const shortcuts = {
    'ctrl+enter': () => runCurrentCell(),
    'ctrl+s': () => saveNotebook(),
    'ctrl+shift+enter': () => runAllCells(),
    'alt+enter': () => runAndInsertBelow(),
    'ctrl+m m': () => changeCellType('note'),
    'ctrl+m c': () => changeCellType('code'),
    'ctrl+m p': () => changeCellType('prompt'),
    'ctrl+up': () => moveCellUp(),
    'ctrl+down': () => moveCellDown(),
};

document.addEventListener('keydown', e => {
    const key = [
        e.ctrlKey ? 'ctrl' : '',
        e.shiftKey ? 'shift' : '',
        e.altKey ? 'alt' : '',
        e.key.toLowerCase()
    ].filter(Boolean).join('+');
    
    if (shortcuts[key]) {
        e.preventDefault();
        shortcuts[key]();
    }
});
```

---

## Testing

### Unit Tests

```python
# tests/test_models.py
import pytest
from app import Cell, Notebook, split_prompt_content, join_prompt_content

def test_split_prompt_content():
    combined = """What is Python?

##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_abc12345 -->

Python is a programming language."""
    
    user, ai = split_prompt_content(combined)
    assert user == "What is Python?"
    assert ai == "Python is a programming language."

def test_join_prompt_content():
    user = "Hello"
    ai = "Hi there!"
    combined = join_prompt_content(user, ai)
    
    assert "Hello" in combined
    assert "Hi there!" in combined
    assert "SOLVEIT_SEPARATOR" in combined

def test_cell_to_jupyter_roundtrip():
    cell = Cell(
        id="test",
        cell_type="prompt",
        source="Question?",
        output="Answer."
    )
    
    jupyter = cell.to_jupyter_cell()
    restored = Cell.from_jupyter_cell(jupyter)
    
    assert restored.source == cell.source
    assert restored.output == cell.output
    assert restored.cell_type == cell.cell_type

def test_notebook_ipynb_roundtrip():
    nb = Notebook(id="test", title="Test")
    nb.cells = [
        Cell(cell_type="note", source="# Hello"),
        Cell(cell_type="code", source="x = 1"),
        Cell(cell_type="prompt", source="What?", output="That."),
    ]
    
    ipynb = nb.to_ipynb()
    restored = Notebook.from_ipynb(ipynb)
    
    assert len(restored.cells) == 3
    assert restored.cells[0].cell_type == "note"
    assert restored.cells[2].output == "That."
```

### Integration Tests

```python
# tests/test_routes.py
from fasthtml.common import Client
from app import app

client = Client(app)

def test_home_redirects():
    response = client.get("/")
    assert response.status_code == 302
    assert "/notebook/" in response.headers["location"]

def test_create_notebook():
    response = client.get("/notebook/new", follow_redirects=False)
    assert response.status_code == 302

def test_add_cell():
    response = client.post("/cell/add?pos=0&type=code")
    assert response.status_code == 200
    assert "cell-" in response.text

def test_run_code_cell():
    # First add a cell
    client.post("/cell/add?pos=0&type=code")
    # Get cell id from response and run it
    # ...
```

---

## Common Patterns

### 1. Async Streaming to HTMX

For real-time streaming responses, use Server-Sent Events:

```python
from fasthtml.common import EventStream

@rt("/cell/{cid}/stream")
async def get(cid: str):
    async def generate():
        async for chunk in llm.stream(prompt, context):
            # Send as SSE
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return EventStream(generate())
```

```html
<!-- In frontend -->
<div hx-ext="sse" sse-connect="/cell/{id}/stream" sse-swap="message">
    <!-- Content updates here -->
</div>
```

### 2. Optimistic Updates

Update UI immediately, then sync with server:

```javascript
function runCell(cellId) {
    // Show loading state immediately
    document.querySelector(`#cell-${cellId} .output`).innerHTML = 
        '<span class="loading">Running...</span>';
    
    // Then make actual request
    htmx.ajax('POST', `/cell/${cellId}/run`, {target: `#cell-${cellId}`});
}
```

### 3. Debounced Saves

Auto-save with debouncing:

```javascript
let saveTimeout;
function debouncedSave(cellId, content) {
    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
        htmx.ajax('POST', `/cell/${cellId}/source`, {
            values: {source: content}
        });
    }, 500);  // Save after 500ms of no typing
}
```

### 4. Error Boundaries

Graceful error handling:

```python
@rt("/cell/{cid}/run")
async def post(cid: str):
    try:
        # ... execution logic ...
        return CellView(cell)
    except Exception as e:
        return Div(
            Div(f"Error: {e}", cls="error-message"),
            Button("Retry", hx_post=f"/cell/{cid}/run", hx_target=f"#cell-{cid}"),
            cls="error-container"
        )
```

---

## Getting Help

- **FastHTML Docs**: https://docs.fastht.ml/
- **HTMX Docs**: https://htmx.org/docs/
- **Solveit**: https://solve.it.com/ (reference implementation)
- **Answer.AI**: https://www.answer.ai/ (FastHTML creators)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [ROADMAP.md](ROADMAP.md) for planned features and areas where help is needed!
