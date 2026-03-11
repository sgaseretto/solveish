# Real-Time Collaboration - How It Works

This document explains the implementation of real-time collaboration in Dialeng, enabling multiple users to work on the same notebook simultaneously.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [WebSocket Connection Management](#websocket-connection-management)
3. [Message Types](#message-types)
4. [OOB Swap Mechanism](#oob-swap-mechanism)
5. [Cell-Specific Behavior](#cell-specific-behavior)
   - [Code Cells](#code-cells)
   - [Note Cells](#note-cells)
   - [Prompt Cells](#prompt-cells)
6. [Conflict Avoidance](#conflict-avoidance)
7. [Key Files and Functions](#key-files-and-functions)
8. [Improving the Implementation](#improving-the-implementation)

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph Client["Client Browser"]
        subgraph JS["JavaScript Layer"]
            connectWS["connectWebSocket()"]
            processOOB["processOOBSwap()<br/>Handles HTML updates from collaborators"]
            appendResp["appendToResponse()<br/>Handles streaming chunks"]
            finishStream["finishStreaming()<br/>Cleans up after streaming completes"]
        end
    end

    subgraph Server["Server (FastHTML)"]
        subgraph WSLayer["WebSocket Layer"]
            wsConn["ws_connections: Dict[notebook_id, List[send_functions]]"]
            wsDecorator["@app.ws('/ws/{nb_id}', conn=..., disconn=...)"]
        end

        subgraph BroadcastLayer["Broadcasting Layer"]
            broadcast["broadcast_to_notebook(nb_id, component)"]
            allCells["AllCellsOOB(nb)<br/>Full cells container with OOB attr"]
            cellView["CellViewOOB(cell)<br/>Single cell with OOB attr"]
            toXml["to_xml(component)<br/>Serializes FastHTML to HTML string"]
        end
    end

    Client <-->|"WebSocket"| Server
    WSLayer --> BroadcastLayer
```

### Key Concepts

1. **WebSocket per Notebook**: Each notebook has its own WebSocket endpoint (`/ws/{notebook_id}`). All clients viewing the same notebook share updates.

2. **OOB (Out-of-Band) Swaps**: Instead of sending JSON messages that need custom handlers, we send HTML with `hx-swap-oob="true"` attributes. This leverages HTMX's built-in DOM replacement.

3. **List-Based Connection Tracking**: Connections are stored as a simple list of `send` functions (following FastHTML's Game of Life pattern), not a dictionary with IDs.

4. **Two Message Categories**:
   - **HTML messages** (start with `<`): Processed via `processOOBSwap()` for DOM updates
   - **JSON messages**: Used for streaming chunks, thinking indicators, and cancellation

---

## WebSocket Connection Management

### Server-Side Connection Tracking

```python
# Global connection registry (app.py:353)
ws_connections: Dict[str, List[Any]] = {}
```

Each notebook ID maps to a list of WebSocket `send` functions.

### Connection Lifecycle

```python
# When a client connects (app.py:2490-2501)
async def ws_on_connect(send, scope):
    # Extract notebook ID from WebSocket path (/ws/notebook_id)
    path = scope.get('path', '')
    parts = path.strip('/').split('/')
    nb_id = parts[1] if len(parts) > 1 else 'default'

    # Add to connections list
    if nb_id not in ws_connections:
        ws_connections[nb_id] = []
    ws_connections[nb_id].append(send)

# When a client disconnects (app.py:2503-2511)
async def ws_on_disconnect(send, scope):
    # Extract notebook ID and remove from list
    if nb_id in ws_connections and send in ws_connections[nb_id]:
        ws_connections[nb_id].remove(send)
```

### Client-Side Connection

```javascript
// Establishes WebSocket on page load (app.py:1600-1606)
function connectWebSocket(notebookId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/${notebookId}`);

    ws.onmessage = function(event) {
        const msg = event.data;

        // HTML messages (collaboration updates)
        if (msg.startsWith('<')) {
            processOOBSwap(msg);
            return;
        }

        // JSON messages (streaming, thinking indicators)
        const data = JSON.parse(msg);
        // ... handle streaming
    };
}
```

---

## Message Types

### HTML Messages (OOB Swaps)

Used for structural updates that replace DOM elements. Targeted swaps (`CellOutputOOB`, `CellHeaderOOB`) replace only subsections to preserve Monaco editors.

| Trigger | Function Called | HTML Target | Editor Impact |
|---------|-----------------|-------------|---------------|
| Add cell | `AllCellsOOB(nb)` | `#cells` | Destroyed & recreated |
| Delete cell | `AllCellsOOB(nb)` | `#cells` | Destroyed & recreated |
| Move cell | `AllCellsOOB(nb)` | `#cells` | Destroyed & recreated |
| Run code cell | `CellOutputOOB(cell)` + `CellHeaderOOB(cell, nb_id)` | `#output-{id}` + `#header-{id}` | **Preserved** |
| State toggle | `CellHeaderOOB(cell, nb_id)` | `#header-{id}` | **Preserved** |
| Collapse cell | `CellViewOOB(cell, nb_id)` | `#cell-{id}` | Destroyed & recreated |
| Change cell type | `CellViewOOB(cell, nb_id)` | `#cell-{id}` | Destroyed & recreated |
| Prompt complete | `CellViewOOB(cell, nb_id)` | `#cell-{id}` | Destroyed & recreated |

### JSON Messages

Used for streaming, in-place editor updates, and state changes.

| Type | When Sent | Payload | Purpose |
|------|-----------|---------|---------|
| `stream_chunk` | Each LLM token | `{type, cell_id, chunk, thinking?}` | Prompt cell streaming |
| `stream_end` | Streaming complete | `{type, cell_id}` | Prompt cell done |
| `code_stream_start` | Code execution begins | `{type, cell_id}` | Show running indicator |
| `code_stream_chunk` | Code output chunk | `{type, cell_id, chunk, stream}` | Stream code output |
| `code_stream_end` | Code execution done | `{type, cell_id, has_error}` | Finalize code cell |
| `cell_source_update` | Dialoghelper source edit | `{type, cell_id, source}` | Update Monaco via `setValue()` — no FOUST |
| `cell_class_update` | State/class change | `{type, cell_id, cls}` | Update CSS classes in-place |
| `cell_state_change` | Cell state transition | `{type, cell_id, state}` | Queue state tracking |
| `queue_update` | Queue changes | `{type, running_cell_id, queued_cell_ids}` | Update queue UI |
| `thinking_start` | Thinking mode begins | `{type, cell_id}` | Show thinking indicator |
| `thinking_end` | Thinking mode ends | `{type, cell_id}` | Hide thinking indicator |
| `cancel` | User clicks cancel (client→server) | `{type, cell_id}` | Cancel streaming |

---

## OOB Swap Mechanism

### What is OOB (Out-of-Band)?

HTMX's OOB swap allows updating elements by ID without the typical request/response cycle. When an element has `hx-swap-oob="true"`, HTMX finds the element with matching ID and replaces it.

### Server-Side: Creating OOB Components (`dialeng/ui/oob.py`)

```python
# Full cells container replacement — destroys all editors (used for add/delete/move)
def AllCellsOOB(nb: Notebook):
    items = [AddButtons(0, nb.id)]
    for i, c in enumerate(nb.cells):
        items.extend([CellView(c, nb.id), AddButtons(i+1, nb.id)])
    return Div(*items, id="cells", hx_swap_oob="true")

# Single cell replacement — destroys editor (used for type change, collapse)
def CellViewOOB(cell: Cell, notebook_id: str):
    cell_div = CellView(cell, notebook_id)
    return Div(*cell_div.children, id=f"cell-{cell.id}",
               cls=cell_div.attrs.get('class', ''), hx_swap_oob="true", ...)

# Targeted output swap — preserves editor (used for execution)
def CellOutputOOB(cell):
    return Div(*output_content, id=f"output-{cell.id}", hx_swap_oob="true")

# Targeted header swap — preserves editor (used for execution, state toggle)
def CellHeaderOOB(cell, notebook_id):
    return Div(*CellHeader(cell, notebook_id).children,
               id=f"header-{cell.id}", hx_swap_oob="true")
```

### Server-Side: Broadcasting

```python
# Broadcast helper (app.py:386-428)
async def broadcast_to_notebook(nb_id: str, component, exclude_send: Any = None):
    """Broadcast an HTML component to all WebSocket connections."""
    if nb_id not in ws_connections or not ws_connections[nb_id]:
        return

    # CRITICAL: Use to_xml() not str() for HTML serialization
    # str(component) returns only the element ID, not HTML!
    html_str = to_xml(component)

    alive = []
    for send in ws_connections[nb_id]:
        if send is exclude_send:
            alive.append(send)
            continue
        try:
            await send(html_str)
            alive.append(send)
        except Exception:
            pass  # Dead connection, don't add to alive

    ws_connections[nb_id] = alive  # Clean up dead connections
```

### Client-Side: Processing OOB Swaps

```javascript
// Process incoming HTML from WebSocket (app.js)
function processOOBSwap(html) {
    const template = document.createElement('template');
    template.innerHTML = html.trim();

    // Collect OOB elements (may be nested in wrapper divs)
    const elements = [];
    for (const el of template.content.children) {
        if (el.getAttribute('hx-swap-oob')) elements.push(el);
        else el.querySelectorAll('[hx-swap-oob]').forEach(n => elements.push(n));
    }

    for (const element of elements) {
        const targetId = element.id;
        const target = document.getElementById(targetId);
        if (!target) continue;

        // Targeted swap: output or header only (preserves Monaco editor)
        if (targetId.startsWith('output-') || targetId.startsWith('header-')) {
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);
            htmx.process(document.getElementById(targetId));
        }
        // Full single cell swap (destroys + recreates Monaco editor)
        else if (targetId.startsWith('cell-')) {
            // Skip if user is editing or cell is streaming
            if (isEditing || isStreaming) continue;
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);
            // Reinitialize Monaco editor for code/shell cells
            if (newCell.dataset.type === 'code') initMonacoEditor(cellId);
            else if (newCell.dataset.type === 'shell') initMonacoEditor(cellId, 'sh');
        }
        // Full cells container update (destroys + recreates all editors)
        else if (targetId === 'cells') {
            element.removeAttribute('hx-swap-oob');
            target.replaceWith(element);
            reinitializeMonacoEditors();
            renderAllPreviews();
        }
    }
}
```

### Script Injection OOB

The `processOOBSwap()` function handles JavaScript injection from dialoghelper's `iife()`, `add_scr()`, and `fire_event()` functions. This requires special handling because scripts inserted via `innerHTML` don't execute automatically (browser security feature).

**Two script injection patterns are supported:**

#### 1. Swap Strategy Pattern (iife, add_scr)

Used by `iife()` and `add_scr()`:

```html
<div hx-swap-oob="beforeend:#js-script">
    <script>console.log('Hello from Python!');</script>
</div>
```

```javascript
// app.py:2684-2726
if (oobAttr && oobAttr.includes(':')) {
    const [swapStrategy, targetSelector] = oobAttr.split(':');
    const target = document.querySelector(targetSelector);

    if (target) {
        const scripts = element.querySelectorAll('script');
        scripts.forEach(script => {
            // Create new script element (triggers execution)
            const newScript = document.createElement('script');
            Array.from(script.attributes).forEach(attr => {
                newScript.setAttribute(attr.name, attr.value);
            });
            newScript.textContent = script.textContent;
            target.appendChild(newScript);  // Executes here
        });
    }
}
```

#### 2. Direct Script Pattern (fire_event)

Used by `fire_event()`:

```html
<script hx-swap-oob="true" id="js-event">htmx.trigger(document.body, 'my-event', {...})</script>
```

```javascript
// app.py:2731-2751
if (element.tagName === 'SCRIPT') {
    const newScript = document.createElement('script');
    Array.from(element.attributes).forEach(attr => {
        if (attr.name !== 'hx-swap-oob') {
            newScript.setAttribute(attr.name, attr.value);
        }
    });
    newScript.textContent = element.textContent;

    // Replace or append
    const existingScript = element.id ? document.getElementById(element.id) : null;
    if (existingScript) {
        existingScript.replaceWith(newScript);
    } else {
        document.body.appendChild(newScript);
    }
}
```

**Key insight**: `document.createElement('script')` followed by DOM insertion triggers execution, unlike `innerHTML` which doesn't.

---

## Cell-Specific Behavior

### Code Cells

**When Run:**
1. Server queues cell execution in the `ExecutionQueue`
2. Route returns `""` immediately (no HTMX swap — `hx_swap="none"`)
3. Output streams via JSON WebSocket messages (`code_stream_chunk`)
4. On completion, targeted OOB swaps update only output and header
5. Monaco editor DOM is **preserved** — no FOUST

**Route:** `POST /notebook/{nb_id}/cell/{cid}/run`

```python
# Code cell execution — returns immediately, runs in background
if c.cell_type == "code":
    queue.queue_cell(nb_id, c)
    return ""

# On completion (via ExecutionQueue callback):
async def finalize_cell_execution(nb_id, cell, has_error):
    cell.output = ''.join(output_parts)
    # Targeted OOB — only output and header, editor untouched
    await broadcast_to_notebook(nb_id, CellOutputOOB(cell))
    await broadcast_to_notebook(nb_id, CellHeaderOOB(cell, nb_id))
```

### Note Cells

**When Run:**
- Note cells don't execute; they just render markdown
- The "run" action simply moves focus to the next cell
- Collaborators see updates when markdown source changes (on blur)

**Route:** `POST /notebook/{nb_id}/cell/{cid}/source`
- Updates source on blur
- No broadcast needed (collaborators don't see typing in real-time)

### Prompt Cells

**Most Complex:** Prompt cells have streaming behavior that requires special handling.

**When Run:**
1. Client calls `startStreaming(cellId)` which:
   - Adds `streaming` class to cell
   - Shows thinking indicator
   - Swaps run button for cancel button

2. Server streams tokens via WebSocket JSON messages:
```python
# Streaming loop (app.py:2398-2425)
async for item in mock_llm_stream(c.source, context, c.use_thinking):
    if cid in cancelled_cells:
        break

    if item["type"] == "chunk":
        response_parts.append(item["content"])

    # Send via WebSocket to ALL connected clients
    if nb_id in ws_connections and ws_connections[nb_id]:
        msg = json.dumps({
            "type": "stream_chunk",
            "cell_id": cid,
            "chunk": item["content"]
        })
        for send in ws_connections[nb_id]:
            try:
                await send(msg)
            except:
                pass
```

3. Client handles streaming chunks:
```javascript
function appendToResponse(cellId, chunk, isThinking) {
    const textarea = document.getElementById(`output-${cellId}`);
    if (textarea) {
        textarea.value += chunk;
        // Also update preview
        const preview = document.querySelector(`[data-cell-id="${cellId}"][data-field="output"]`);
        if (preview) {
            preview.innerHTML = renderMarkdown(textarea.value);
        }
    }
}
```

4. When streaming completes:
```python
# Send end signal (app.py:2430-2437)
msg = json.dumps({"type": "stream_end", "cell_id": cid})
for send in ws_connections[nb_id]:
    await send(msg)

# Broadcast final cell state via OOB
await broadcast_to_notebook(nb_id, CellViewOOB(c, nb_id))
```

5. Client finishes streaming:
```javascript
function finishStreaming(cellId) {
    const cell = document.getElementById(`cell-${cellId}`);
    cell.classList.remove('streaming');
    // Show run button, hide cancel button
}
```

**Important Notes for Prompt Cells:**
- Streaming goes to ALL connected clients
- The initiating client has `streaming` class set, so OOB updates are skipped
- Collaborators receive stream chunks but may not have proper UI setup
- After streaming, the OOB broadcast ensures all clients have final state

---

## Conflict Avoidance

### Why Conflicts Happen

When multiple users edit the same notebook:
- User A might be typing in a cell while User B runs a cell
- User A might be editing while the server broadcasts an update
- A cell might be streaming while an update arrives

### How Conflicts Are Avoided

1. **Check Before Replacing (Client-Side)**
```javascript
function processOOBSwap(html) {
    // ...
    if (targetId.startsWith('cell-')) {
        const isEditing = target.contains(document.activeElement);
        const isStreaming = target.classList.contains('streaming');

        if (isEditing || isStreaming) {
            console.log('[WS] Skipping OOB swap');
            continue;  // Don't replace
        }
    }
}
```

2. **Skip Full Container Updates During Editing**
```javascript
if (targetId === 'cells') {
    const editingCell = focusedCell;
    const streamingCell = document.querySelector('.cell.streaming');

    if (editingCell || streamingCell) {
        continue;  // Don't replace entire container
    }
}
```

3. **No Real-Time Typing Sync**
- Individual keystrokes are NOT broadcast
- Only cell-level operations (run, add, delete, move) trigger updates
- This prevents constant interruptions during typing

### What's NOT Protected

- If User A is editing Cell 1 and User B deletes Cell 2, User A's container updates
- Race conditions when two users perform actions simultaneously
- No operational transform or CRDT (no merge of concurrent edits)

---

## Key Files and Functions

### Server-Side (app.py)

| Location | Function | Purpose |
|----------|----------|---------|
| Line 353 | `ws_connections` | Global dict tracking WebSocket connections per notebook |
| Line 356 | `cancelled_cells` | Set tracking cancelled prompt generations |
| Line 386-428 | `broadcast_to_notebook()` | Sends HTML to all WebSocket clients |
| Line 2124-2132 | `AllCellsOOB()` | Creates full cells container with OOB attribute |
| Line 2134-2150 | `CellViewOOB()` | Creates single cell with OOB attribute |
| Line 2490-2501 | `ws_on_connect()` | Handles new WebSocket connections |
| Line 2503-2511 | `ws_on_disconnect()` | Handles WebSocket disconnections |
| Line 2513-2529 | `@app.ws('/ws/{nb_id}')` | WebSocket endpoint handler |

### Client-Side (`dialeng/static/js/app.js`)

| Function | Purpose |
|----------|---------|
| `connectWebSocket()` | Establishes WebSocket, routes HTML to `processOOBSwap` and JSON to handlers |
| `processOOBSwap()` | Handles incoming HTML OOB updates (targeted, cell, or full container) |
| `initMonacoEditor()` | Creates Monaco editor with skip guard for existing editors |
| `reinitializeMonacoEditors()` | Recreates all Monaco editors after full `#cells` swap |
| `startCodeStreaming()` / `finishCodeStreaming()` | Manage code cell execution UI state |
| `appendCodeOutput()` | Debounced (RAF) streaming output for code cells |
| `focusNextCell()` / `moveToNextCell()` | Cell navigation with proper DOM focus for all cell types |

---

## Improving the Implementation

### Current Limitations

1. **No Typing Sync**: Collaborators don't see each other typing in real-time
2. **Basic Conflict Handling**: Edits during updates are simply skipped
3. **No Cursor Awareness**: Can't see where collaborators are editing
4. **No Undo/Redo Sync**: Each client has independent undo history
5. **Streaming Goes to All**: Prompt streaming reaches all clients, even those not ready

### Potential Improvements

#### 1. Add Cursor Presence
Show where each collaborator is editing:
```javascript
// Broadcast cursor position
function broadcastCursor(cellId, position) {
    ws.send(JSON.stringify({
        type: 'cursor',
        cell_id: cellId,
        position: position,
        user_id: userId
    }));
}

// Display remote cursors
function showRemoteCursor(cellId, position, userId) {
    // Add colored cursor indicator in the cell
}
```

#### 2. Stream Only to Initiating Client
Currently, prompt streaming goes to ALL clients. To stream only to the initiator:

```python
# Option 1: Track which connection initiated the request
# This requires correlating HTTP requests with WebSocket connections

# Option 2: Use a unique request ID
@rt("/notebook/{nb_id}/cell/{cid}/run")
async def post(nb_id: str, cid: str, request_id: str = None):
    # Mark which request ID should receive streaming
    streaming_requests[cid] = request_id

    # Only send to matching connection
    for send in ws_connections[nb_id]:
        if get_request_id(send) == request_id:
            await send(chunk)
```

#### 3. Add Operational Transform (OT)
For true real-time collaborative editing (like Google Docs):
- Track operations (insert, delete) rather than full state
- Transform concurrent operations to maintain consistency
- Requires significant architecture changes

Libraries to consider:
- `yjs` - CRDT-based collaborative editing
- `ShareDB` - OT-based real-time database

#### 4. Improve Conflict Resolution
Instead of skipping updates, merge changes:
```javascript
function processOOBSwap(html) {
    if (isEditing) {
        // Instead of skipping, queue the update
        pendingUpdates[cellId] = html;
        // Apply when user stops editing
    }
}

function onCellBlur(cellId) {
    if (pendingUpdates[cellId]) {
        applyPendingUpdate(cellId);
    }
}
```

#### 5. Add User Identification
Track who made each change:
```python
# Add user info to connections
ws_connections: Dict[str, List[Tuple[Any, str]]] = {}  # [(send, user_id), ...]

# Include user info in broadcasts
await broadcast_to_notebook(nb_id, CellViewOOB(cell, nb_id),
                           metadata={"user": current_user})
```

### Testing Collaboration

To test collaboration locally:

1. Start the server: `uv run python app.py`
2. Open http://localhost:8000 in one browser tab
3. Open the same URL in another tab (or incognito window)
4. Make changes in one tab and watch them appear in the other

For multi-machine testing:
1. Find your local IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Share URL like `http://192.168.1.100:8000/notebook/mynotebook`
3. Other devices on the same network can collaborate

### Debugging Tips

1. **Check Server Logs**: Look for `[BROADCAST]` and `[WS]` messages
2. **Browser Console**: Look for `[WS] Received` and `[WS] Skipping` messages
3. **Network Tab**: Monitor WebSocket frames in browser DevTools
4. **Test with Simple Changes**: Try adding/deleting cells before testing complex scenarios

---

## Summary

The real-time collaboration system uses:
- **WebSocket** for bidirectional communication
- **OOB swaps** for efficient DOM updates
- **List-based connection tracking** for simplicity
- **HTML + JSON hybrid** for different update types
- **Client-side conflict checking** to avoid interrupting users

The implementation prioritizes simplicity over features like cursor awareness or true concurrent editing, making it easier to understand and maintain while still providing useful real-time collaboration.
