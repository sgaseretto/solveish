"""
Output rendering utilities - ANSI conversion, MIME bundle rendering.

Shared between app.py (WebSocket streaming), code_cell.py (static rendering),
and oob.py (OOB swap rendering).
"""
import re
import json
import html as html_module


def ansi_to_html(text: str) -> str:
    """
    Convert ANSI escape codes to HTML spans with inline styles.

    Handles common ANSI codes for colors (30-37, 90-97), backgrounds (40-47),
    bold (1), and reset (0).
    """
    ANSI_COLORS = {
        '30': '#000', '31': '#c00', '32': '#0a0', '33': '#a50',
        '34': '#00a', '35': '#a0a', '36': '#0aa', '37': '#aaa',
        '90': '#555', '91': '#f55', '92': '#5f5', '93': '#ff5',
        '94': '#55f', '95': '#f5f', '96': '#5ff', '97': '#fff',
    }
    ANSI_BG_COLORS = {
        '40': '#000', '41': '#c00', '42': '#0a0', '43': '#a50',
        '44': '#00a', '45': '#a0a', '46': '#0aa', '47': '#aaa',
    }

    result = []
    open_spans = 0

    parts = re.split(r'(\x1b\[[0-9;]*m)', text)

    for part in parts:
        match = re.match(r'\x1b\[([0-9;]*)m', part)
        if match:
            codes = match.group(1).split(';')
            for code in codes:
                if code == '0' or code == '':
                    while open_spans > 0:
                        result.append('</span>')
                        open_spans -= 1
                elif code == '1':
                    result.append('<span style="font-weight:bold">')
                    open_spans += 1
                elif code in ANSI_COLORS:
                    result.append(f'<span style="color:{ANSI_COLORS[code]}">')
                    open_spans += 1
                elif code in ANSI_BG_COLORS:
                    result.append(f'<span style="background:{ANSI_BG_COLORS[code]}">')
                    open_spans += 1
        else:
            result.append(html_module.escape(part))

    while open_spans > 0:
        result.append('</span>')
        open_spans -= 1

    return ''.join(result)


def render_mime_bundle(data: dict, metadata: dict = None) -> str:
    """
    Convert Jupyter MIME bundle to HTML.
    Priority: text/html > image/svg+xml > image/png > image/jpeg > text/markdown > text/plain

    Args:
        data: Dict with MIME types as keys and content as values
        metadata: Optional dict with rendering hints (width, height, etc.)

    Returns:
        HTML string for rendering
    """
    metadata = metadata or {}

    # HTML - render directly (trusted user code, matches Jupyter behavior)
    if 'text/html' in data:
        return f'<div class="mime-html">{data["text/html"]}</div>'

    # SVG - render inline
    if 'image/svg+xml' in data:
        return f'<div class="mime-svg">{data["image/svg+xml"]}</div>'

    # PNG image - base64 data URL
    if 'image/png' in data:
        width = metadata.get('width', '')
        height = metadata.get('height', '')
        style_parts = []
        if width:
            style_parts.append(f'width:{width}px')
        if height:
            style_parts.append(f'height:{height}px')
        style = ';'.join(style_parts)
        style_attr = f' style="{style}"' if style else ''
        # Strip newlines from base64 data (Jupyter wire protocol may include them)
        b64 = data["image/png"].replace('\n', '').replace('\r', '')
        return f'<img class="mime-image" src="data:image/png;base64,{b64}"{style_attr} />'

    # JPEG image
    if 'image/jpeg' in data:
        b64 = data["image/jpeg"].replace('\n', '').replace('\r', '')
        return f'<img class="mime-image" src="data:image/jpeg;base64,{b64}" />'

    # GIF image
    if 'image/gif' in data:
        b64 = data["image/gif"].replace('\n', '').replace('\r', '')
        return f'<img class="mime-image" src="data:image/gif;base64,{b64}" />'

    # Markdown - convert to HTML using mistlefoot for extended features
    if 'text/markdown' in data:
        try:
            from mistletoe import markdown as md_render
            from mistlefoot import ExtendedHtmlRenderer
            rendered = md_render(data['text/markdown'], ExtendedHtmlRenderer)
            return f'<div class="mime-markdown">{rendered}</div>'
        except ImportError:
            return f'<div class="mime-markdown">{data["text/markdown"]}</div>'

    # LaTeX - wrap for MathJax/KaTeX processing
    if 'text/latex' in data:
        return f'<div class="mime-latex">{html_module.escape(data["text/latex"])}</div>'

    # JSON - pretty print
    if 'application/json' in data:
        json_str = json.dumps(data['application/json'], indent=2)
        return f'<pre class="mime-json">{html_module.escape(json_str)}</pre>'

    # Plain text fallback
    if 'text/plain' in data:
        return f'<pre class="mime-text">{html_module.escape(data["text/plain"])}</pre>'

    # Unknown format - show raw
    return f'<pre class="mime-unknown">{html_module.escape(str(data))}</pre>'
