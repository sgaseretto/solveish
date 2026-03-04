"""
Demo Controller - Dialeng Presentation Remote

A FastHTML app that acts as a phone-based remote control for stepping through
a scripted Dialeng demo. Uses httpx to call Dialeng's dialoghelper HTTP endpoints,
adding/modifying cells in real-time. All content (code, outputs, LLM responses)
is pre-scripted for a deterministic presentation.

Usage:
    uv run python demo_controller.py                        # defaults: dialeng=8000, controller=8001
    uv run python demo_controller.py --port 8002            # custom controller port
    uv run python demo_controller.py --dialeng-port 9000    # custom dialeng port
"""
import argparse, asyncio, json
from dataclasses import dataclass, field
from fasthtml.common import *
import httpx

# =============================================================================
# Demo State
# =============================================================================

@dataclass
class DemoState:
    notebook_id: str = ""
    current_step: int = 0
    cell_ids: dict = field(default_factory=dict)  # label -> cell_id
    dialeng_port: int = 8000

    def reset(self):
        self.current_step = 0
        self.cell_ids = {}

    @property
    def base_url(self): return f"http://localhost:{self.dialeng_port}"

demo = DemoState()

# =============================================================================
# Dialeng API Helpers (httpx -> dialoghelper endpoints)
# =============================================================================

async def add_cell(content, msg_type="code", placement="after", ref_id="", output="",
                   o_collapsed="0", i_collapsed="0"):
    """Add a cell to the notebook. Returns the new cell ID."""
    data = {
        "dlg_name": demo.notebook_id,
        "content": content,
        "msg_type": msg_type,
        "placement": placement,
        "output": output,
        "o_collapsed": o_collapsed,
        "i_collapsed": i_collapsed,
    }
    if ref_id: data["id_"] = ref_id
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{demo.base_url}/add_relative_", data=data, timeout=10)
        return resp.text.strip()


async def update_cell(cell_id, content=None, output=None, msg_type=None):
    """Update a cell's content, output, or type."""
    data = {"dlg_name": demo.notebook_id, "id_": cell_id}
    if content is not None: data["content"] = content
    if output is not None: data["output"] = output
    if msg_type is not None: data["msg_type"] = msg_type
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{demo.base_url}/update_msg_", data=data, timeout=10)
        return resp.text.strip()


async def delete_cell(cell_id):
    """Delete a cell from the notebook."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{demo.base_url}/rm_msg_", data={
            "dlg_name": demo.notebook_id, "msid": cell_id
        }, timeout=10)
        return resp.json()


async def move_cell(cell_id, direction="up"):
    """Move a cell up or down."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{demo.base_url}/notebook/{demo.notebook_id}/cell/{cell_id}/move/{direction}",
            timeout=10
        )
        return resp.text


async def get_all_cells():
    """Get all cells in the notebook."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{demo.base_url}/curr_dialog_", data={
            "dlg_name": demo.notebook_id, "with_messages": "true"
        }, timeout=10)
        return resp.json()

# =============================================================================
# Mocked Content
# =============================================================================

CODE_CREATE_DATA = '''\
import pandas as pd
import numpy as np

def create_sample_data():
    """Generate sample monthly sales data."""
    np.random.seed(42)
    months = pd.date_range('2024-01', periods=12, freq='M')
    return pd.DataFrame({
        'month': months,
        'units_sold': np.random.randint(100, 500, 12),
        'unit_price': np.round(np.random.uniform(10, 50, 12), 2),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 12)
    })'''

CODE_CLEAN_DATA = '''\
def clean_data(df):
    """Clean and add time features."""
    df = df.copy()
    df['month_name'] = df['month'].dt.strftime('%b')
    df['quarter'] = df['month'].dt.quarter
    return df'''

CODE_TRANSFORM_DATA = '''\
def transform_data(df):
    """Add revenue metrics."""
    df = df.copy()
    df['revenue'] = df['units_sold'] * df['unit_price']
    df['revenue_k'] = (df['revenue'] / 1000).round(1)
    return df'''

CODE_TEST_PIPELINE = '''\
# Run the full pipeline
raw = create_sample_data()
cleaned = clean_data(raw)
transformed = transform_data(cleaned)

print(f"Pipeline: {len(raw)} rows processed")
print(f"\\nRevenue by region:")
print(transformed.groupby('region')['revenue'].sum().round(0))
print(f"\\nTotal revenue: ${transformed['revenue'].sum():,.0f}")
transformed[['month_name', 'units_sold', 'unit_price', 'revenue', 'region']]'''

OUTPUT_TEST_PIPELINE = """\
Pipeline: 12 rows processed

Revenue by region:
region
East     11234.0
North    14567.0
South     8923.0
West     12345.0
Name: revenue, dtype: float64

Total revenue: $47,069.00

    month_name  units_sold  unit_price   revenue region
0          Jan         202       29.84   6027.68  South
1          Feb         435       15.60   6786.00   East
2          Mar         292       41.17  12021.64  North
3          Apr         177       27.93   4943.61   West
4          May         390       12.45   4855.50  North
5          Jun         268       38.72  10377.96  South
6          Jul         456       22.18  10114.08   East
7          Aug         123       44.91   5523.93   West
8          Sep         334       19.55   6529.70  North
9          Oct         401       33.27  13341.27  North
10         Nov         189       47.12   8903.68   West
11         Dec         312       25.63   7996.56   East"""

PROMPT_VISUALIZATION = """\
Given the monthly sales data above with revenue by month, create a matplotlib \
bar chart showing monthly revenue. Use the `transformed` DataFrame which has \
columns: month_name, revenue, units_sold, region."""

LLM_RESPONSE_CHART = """\
Here's a bar chart showing the monthly revenue distribution:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(transformed['month_name'], transformed['revenue'])
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.title('Monthly Revenue 2024')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

This creates a simple bar chart with each month on the x-axis and revenue on the \
y-axis. The bars show the revenue for each month of 2024."""

CODE_BASIC_CHART = '''\
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(transformed['month_name'], transformed['revenue'])
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.title('Monthly Revenue 2024')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

OUTPUT_BASIC_CHART = "[Chart displayed: Basic bar chart - Monthly Revenue 2024]"

PROMPT_IMPROVE_CHART = """\
The code cell above creates a basic bar chart of monthly revenue. I'd like you to \
improve it with:
1. A blue-to-green color gradient for the bars
2. Revenue dollar labels on top of each bar
3. A red dashed trend line showing overall direction
4. Cleaner styling (remove top/right spines)"""

LLM_RESPONSE_IMPROVED = """\
Here's the improved visualization with gradient colors, value labels, and a trend line:

```python
import matplotlib.pyplot as plt
import numpy as np

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(transformed)))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(transformed['month_name'], transformed['revenue'],
              color=colors, edgecolor='white', linewidth=0.5)

# Value labels on each bar
for bar, val in zip(bars, transformed['revenue']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Trend line
x_num = np.arange(len(transformed))
z = np.polyfit(x_num, transformed['revenue'].values, 1)
p = np.poly1d(z)
ax.plot(x_num, p(x_num), '--', color='#e74c3c', linewidth=2, label='Trend')

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.set_title('Monthly Revenue 2024', fontsize=14, fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Key improvements:
- **Color gradient** using viridis colormap for visual hierarchy
- **Dollar labels** on each bar for quick reading
- **Trend line** (red dashed) showing the overall revenue direction
- **Clean styling** with removed spines for a modern look"""

CODE_IMPROVED_CHART = '''\
import matplotlib.pyplot as plt
import numpy as np

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(transformed)))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(transformed['month_name'], transformed['revenue'],
              color=colors, edgecolor='white', linewidth=0.5)

# Value labels on each bar
for bar, val in zip(bars, transformed['revenue']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Trend line
x_num = np.arange(len(transformed))
z = np.polyfit(x_num, transformed['revenue'].values, 1)
p = np.poly1d(z)
ax.plot(x_num, p(x_num), '--', color='#e74c3c', linewidth=2, label='Trend')

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.set_title('Monthly Revenue 2024', fontsize=14, fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

OUTPUT_IMPROVED_CHART = "[Chart displayed: Improved bar chart with gradient colors, value labels, and trend line]"

PROMPT_CONSOLIDATE = """\
Now consolidate the entire pipeline into a single `analyze_sales()` function that:
1. Creates the sample data
2. Cleans and transforms it
3. Generates the improved chart with gradient, labels, and trend line
4. Returns the processed DataFrame"""

LLM_RESPONSE_CONSOLIDATED = """\
Here's the consolidated function that runs the complete analysis pipeline:

```python
def analyze_sales():
    \"\"\"Complete sales analysis: load, transform, and visualize.\"\"\"
    import pandas as pd, numpy as np, matplotlib.pyplot as plt

    # 1. Create sample data
    np.random.seed(42)
    months = pd.date_range('2024-01', periods=12, freq='M')
    df = pd.DataFrame({
        'month': months,
        'units_sold': np.random.randint(100, 500, 12),
        'unit_price': np.round(np.random.uniform(10, 50, 12), 2),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 12)
    })

    # 2. Clean & transform
    df['month_name'] = df['month'].dt.strftime('%b')
    df['revenue'] = df['units_sold'] * df['unit_price']

    # 3. Visualize
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df['month_name'], df['revenue'], color=colors,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, df['revenue']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    x_num = np.arange(len(df))
    z = np.polyfit(x_num, df['revenue'].values, 1)
    ax.plot(x_num, np.poly1d(z)(x_num), '--', color='#e74c3c',
            linewidth=2, label='Trend')
    ax.set_title('Monthly Revenue 2024', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Revenue ($)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    return df

result = analyze_sales()
print(f"Analysis complete: {len(result)} months")
print(f"Total revenue: ${result['revenue'].sum():,.0f}")
```

This single function encapsulates the entire workflow: data creation, cleaning, \
transformation, and the improved visualization with gradient bars, labels, and trend line."""

CODE_FINAL = '''\
def analyze_sales():
    """Complete sales analysis: load, transform, and visualize."""
    import pandas as pd, numpy as np, matplotlib.pyplot as plt

    # 1. Create sample data
    np.random.seed(42)
    months = pd.date_range('2024-01', periods=12, freq='M')
    df = pd.DataFrame({
        'month': months,
        'units_sold': np.random.randint(100, 500, 12),
        'unit_price': np.round(np.random.uniform(10, 50, 12), 2),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 12)
    })

    # 2. Clean & transform
    df['month_name'] = df['month'].dt.strftime('%b')
    df['revenue'] = df['units_sold'] * df['unit_price']

    # 3. Visualize
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df['month_name'], df['revenue'], color=colors,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, df['revenue']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    x_num = np.arange(len(df))
    z = np.polyfit(x_num, df['revenue'].values, 1)
    ax.plot(x_num, np.poly1d(z)(x_num), '--', color='#e74c3c',
            linewidth=2, label='Trend')
    ax.set_title('Monthly Revenue 2024', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Revenue ($)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    return df

result = analyze_sales()
print(f"Analysis complete: {len(result)} months")
print(f"Total revenue: ${result['revenue'].sum():,.0f}")'''

OUTPUT_FINAL = """\
[Chart displayed: Complete analysis chart with gradient, labels, and trend line]

Analysis complete: 12 months
Total revenue: $47,069.00"""

# =============================================================================
# Demo Steps
# =============================================================================

STEPS = [
    {"label": "Set up notebook",       "desc": "Add title note"},
    {"label": "Add data functions",    "desc": "Add create_sample_data, clean_data, transform_data"},
    {"label": "Test the pipeline",     "desc": "Add test cell with mocked output"},
    {"label": "Ask for visualization", "desc": "Add prompt cell asking for a chart"},
    {"label": "LLM generates chart",   "desc": "Show mocked LLM response with chart code"},
    {"label": "Test the chart",        "desc": "Add code cell with chart code and mocked output"},
    {"label": "Rearrange cells",       "desc": "Move chart cell above the prompt cell"},
    {"label": "Refine the prompt",     "desc": "Update prompt to ask for improvements"},
    {"label": "LLM improves chart",    "desc": "Show mocked improved LLM response"},
    {"label": "Test improved chart",   "desc": "Add code cell with improved chart and mocked output"},
    {"label": "Ask to consolidate",    "desc": "Add prompt asking to consolidate everything"},
    {"label": "LLM consolidates",      "desc": "Show mocked consolidated function"},
    {"label": "Test final function",   "desc": "Add final code cell with mocked output"},
]


async def run_step(step_num):
    """Execute a demo step. Returns True on success."""
    ids = demo.cell_ids

    if step_num == 0:
        # Step 1: Add title note
        ids['title'] = await add_cell(
            "# Sales Data Analysis Pipeline\n\nBuilding a data analysis workflow step by step.",
            msg_type="note", placement="after"
        )

    elif step_num == 1:
        # Step 2: Add three function cells
        ids['create_data'] = await add_cell(
            CODE_CREATE_DATA, msg_type="code", ref_id=ids['title']
        )
        ids['clean_data'] = await add_cell(
            CODE_CLEAN_DATA, msg_type="code", ref_id=ids['create_data']
        )
        ids['transform_data'] = await add_cell(
            CODE_TRANSFORM_DATA, msg_type="code", ref_id=ids['clean_data']
        )

    elif step_num == 2:
        # Step 3: Add test cell with mocked output
        ids['test_pipeline'] = await add_cell(
            CODE_TEST_PIPELINE, msg_type="code", ref_id=ids['transform_data'],
            output=OUTPUT_TEST_PIPELINE, o_collapsed="1"
        )

    elif step_num == 3:
        # Step 4: Add prompt cell
        ids['prompt_chart'] = await add_cell(
            PROMPT_VISUALIZATION, msg_type="prompt", ref_id=ids['test_pipeline']
        )

    elif step_num == 4:
        # Step 5: Mock LLM response
        await update_cell(ids['prompt_chart'], output=LLM_RESPONSE_CHART)

    elif step_num == 5:
        # Step 6: Add chart code cell with mocked output
        ids['basic_chart'] = await add_cell(
            CODE_BASIC_CHART, msg_type="code", ref_id=ids['prompt_chart'],
            output=OUTPUT_BASIC_CHART, o_collapsed="1"
        )

    elif step_num == 6:
        # Step 7: Move chart cell above the prompt cell
        # The chart cell is currently AFTER the prompt. We need to move it
        # above the prompt (which is 1 position up).
        await move_cell(ids['basic_chart'], "up")

    elif step_num == 7:
        # Step 8: Update the prompt to reference chart code above
        await update_cell(ids['prompt_chart'],
                          content=PROMPT_IMPROVE_CHART,
                          output="")  # Clear old response

    elif step_num == 8:
        # Step 9: Mock improved LLM response
        await update_cell(ids['prompt_chart'], output=LLM_RESPONSE_IMPROVED)

    elif step_num == 9:
        # Step 10: Add improved chart code cell
        ids['improved_chart'] = await add_cell(
            CODE_IMPROVED_CHART, msg_type="code", ref_id=ids['prompt_chart'],
            output=OUTPUT_IMPROVED_CHART, o_collapsed="1"
        )

    elif step_num == 10:
        # Step 11: Add consolidation prompt
        ids['prompt_consolidate'] = await add_cell(
            PROMPT_CONSOLIDATE, msg_type="prompt", ref_id=ids['improved_chart']
        )

    elif step_num == 11:
        # Step 12: Mock consolidated LLM response
        await update_cell(ids['prompt_consolidate'], output=LLM_RESPONSE_CONSOLIDATED)

    elif step_num == 12:
        # Step 13: Add final code cell
        ids['final'] = await add_cell(
            CODE_FINAL, msg_type="code", ref_id=ids['prompt_consolidate'],
            output=OUTPUT_FINAL, o_collapsed="1"
        )

    return True

# =============================================================================
# FastHTML App
# =============================================================================

app, rt = fast_app(
    pico=True,
    hdrs=(
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Style("""
            :root { --pico-font-size: 16px; }
            body { max-width: 500px; margin: 0 auto; padding: 1rem; }
            .step-list { list-style: none; padding: 0; margin: 1rem 0; }
            .step-item {
                padding: 0.5rem 0.75rem; margin: 0.25rem 0;
                border-radius: 8px; border-left: 4px solid transparent;
                font-size: 0.9rem; color: var(--pico-muted-color);
            }
            .step-item.done {
                border-left-color: #2ecc71; color: var(--pico-color);
            }
            .step-item.current {
                border-left-color: #3498db; background: rgba(52,152,219,0.1);
                color: var(--pico-color); font-weight: 600;
            }
            .step-item.pending { opacity: 0.5; }
            .next-btn {
                width: 100%; padding: 1.2rem; font-size: 1.3rem;
                font-weight: 700; border-radius: 12px; margin: 1rem 0;
            }
            .restart-btn {
                width: 100%; padding: 0.8rem; font-size: 1rem;
                background: #e74c3c; border-color: #e74c3c;
                border-radius: 12px; margin-top: 1rem;
            }
            .restart-btn:hover { background: #c0392b; border-color: #c0392b; }
            .config-section { margin-bottom: 1.5rem; }
            .config-section input { margin-bottom: 0.5rem; }
            .status { text-align: center; padding: 0.5rem; border-radius: 8px;
                      margin-bottom: 1rem; font-size: 0.9rem; }
            .status.connected { background: rgba(46,204,113,0.15); color: #27ae60; }
            .status.disconnected { background: rgba(231,76,60,0.15); color: #c0392b; }
            .step-counter { text-align: center; font-size: 1.1rem;
                            color: var(--pico-muted-color); margin: 0.5rem 0; }
            .done-msg { text-align: center; padding: 2rem; font-size: 1.2rem;
                        color: #27ae60; font-weight: 600; }
        """),
    ),
)


def step_list_html():
    """Render the step list with current/done/pending styling."""
    items = []
    for i, step in enumerate(STEPS):
        if i < demo.current_step:
            cls = "step-item done"
            prefix = "  "
        elif i == demo.current_step:
            cls = "step-item current"
            prefix = "-> "
        else:
            cls = "step-item pending"
            prefix = "   "
        items.append(Li(f"{prefix}{i+1}. {step['label']}", cls=cls))
    return Ul(*items, cls="step-list")


def main_ui():
    """Build the full controller UI."""
    connected = bool(demo.notebook_id)
    total = len(STEPS)
    done = demo.current_step >= total

    parts = [H2("Dialeng Demo Remote")]

    # Config section
    if not connected:
        parts.append(Div(
            Form(
                Label("Notebook ID (from URL)", _for="nb_id"),
                Input(id="nb_id", name="notebook_id", placeholder="e.g. a1b2c3d4",
                      required=True, autofocus=True),
                Label("Dialeng Port", _for="port_input"),
                Input(id="port_input", name="dialeng_port", type="number",
                      value=str(demo.dialeng_port), required=True),
                Button("Connect", type="submit", cls="next-btn"),
                hx_post="/connect", hx_target="#app-root", hx_swap="innerHTML",
            ),
            cls="config-section"
        ))
    else:
        # Connected status
        parts.append(Div(
            f"Connected to notebook: {demo.notebook_id}",
            cls="status connected"
        ))

        # Step counter
        if not done:
            parts.append(Div(
                f"Step {demo.current_step + 1} of {total}",
                cls="step-counter"
            ))

        # Step list
        parts.append(step_list_html())

        # Next button or done message
        if done:
            parts.append(Div("Demo complete!", cls="done-msg"))
        else:
            next_step = STEPS[demo.current_step]
            parts.append(
                Button(
                    f"Next: {next_step['label']}",
                    hx_post="/next", hx_target="#app-root", hx_swap="innerHTML",
                    cls="next-btn",
                )
            )
            parts.append(Small(next_step['desc'],
                               style="display:block; text-align:center; opacity:0.7;"))

        # Restart button
        parts.append(
            Button(
                "Restart Demo",
                hx_post="/restart", hx_target="#app-root", hx_swap="innerHTML",
                hx_confirm="This will delete all cells. Continue?",
                cls="restart-btn",
            )
        )

        # Disconnect link
        parts.append(
            A("Disconnect", href="/disconnect",
              style="display:block; text-align:center; margin-top:1rem; font-size:0.85rem;")
        )

    return Div(*parts, id="app-root")


@rt("/")
def get():
    return Titled("Demo Controller", main_ui())


@rt("/connect")
async def post(notebook_id: str, dialeng_port: int = 8000):
    demo.notebook_id = notebook_id.strip()
    demo.dialeng_port = dialeng_port
    demo.reset()
    # Verify connection by calling curr_dialog
    try:
        result = await get_all_cells()
        if "error" in str(result).lower() and "not found" in str(result).lower():
            demo.notebook_id = ""
            return Div(
                Div(f"Notebook '{notebook_id}' not found. Check the ID.",
                    cls="status disconnected"),
                main_ui(),
                id="app-root"
            )
    except Exception as e:
        demo.notebook_id = ""
        return Div(
            Div(f"Cannot reach Dialeng at port {dialeng_port}: {e}",
                cls="status disconnected"),
            main_ui(),
            id="app-root"
        )
    return main_ui()


@rt("/disconnect")
def get():
    demo.notebook_id = ""
    demo.reset()
    return RedirectResponse("/", status_code=303)


@rt("/next")
async def post():
    if demo.current_step < len(STEPS):
        try:
            await run_step(demo.current_step)
            demo.current_step += 1
        except Exception as e:
            return Div(
                Div(f"Step failed: {e}", cls="status disconnected"),
                main_ui(),
                id="app-root"
            )
    return main_ui()


@rt("/restart")
async def post():
    if demo.notebook_id:
        try:
            result = await get_all_cells()
            messages = result.get("messages", [])
            # Delete in reverse order to avoid index shifting
            for msg in reversed(messages):
                await delete_cell(msg["id"])
        except Exception as e:
            return Div(
                Div(f"Restart error: {e}", cls="status disconnected"),
                main_ui(),
                id="app-root"
            )
    demo.reset()
    return main_ui()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dialeng Demo Controller")
    parser.add_argument("--port", type=int, default=8001, help="Controller port (default: 8001)")
    parser.add_argument("--dialeng-port", type=int, default=8000, help="Dialeng port (default: 8000)")
    args = parser.parse_args()

    demo.dialeng_port = args.dialeng_port
    serve(port=args.port)
