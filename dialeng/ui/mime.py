"""
Output rendering utilities - ANSI conversion, MIME bundle rendering.

Shared between app.py (WebSocket streaming), code_cell.py (static rendering),
and oob.py (OOB swap rendering).
"""
import re
import json
import html as html_module


def _mime_text(value) -> str:
    """Coerce legacy line arrays and scalar MIME values to text."""
    if isinstance(value, list):
        return ''.join(str(part) for part in value)
    return str(value) if value is not None else ''


def _mime_metadata(metadata: dict, mime_type: str) -> dict:
    """Merge bundle-level and MIME-specific Jupyter metadata."""
    base = {key: value for key, value in metadata.items() if not isinstance(value, dict)}
    specific = metadata.get(mime_type)
    if isinstance(specific, dict):
        base.update(specific)
    return base


def _dimension_style(value) -> str:
    """Return a safe CSS image dimension from Jupyter metadata."""
    if isinstance(value, (int, float)) and value >= 0:
        return f'{value:g}px'
    if isinstance(value, str):
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)(px|%)?\s*', value)
        if match:
            return match.group(1) + (match.group(2) or 'px')
    return ''


def _image_html(data: dict, metadata: dict, mime_type: str) -> str:
    value = _mime_text(data[mime_type]).replace('\n', '').replace('\r', '')
    hints = _mime_metadata(metadata, mime_type)
    styles = []
    for key in ('width', 'height'):
        dimension = _dimension_style(hints.get(key))
        if dimension:
            styles.append(f'{key}:{dimension}')
    style = html_module.escape(';'.join(styles), quote=True)
    style_attr = f' style="{style}"' if style else ''
    alt = html_module.escape(_mime_text(hints.get('alt', 'Rich output image')), quote=True)
    return (
        f'<img class="mime-image" loading="lazy" decoding="async" '
        f'alt="{alt}" src="data:{mime_type};base64,{value}"{style_attr} />'
    )


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

    # Strip non-SGR ANSI sequences (cursor control, erase codes like \x1b[2K,
    # \x1b[?25h, etc.) that we don't render — tqdm uses these extensively
    text = re.sub(r'\x1b\[[0-9;]*[A-HJKSTfn]', '', text)  # cursor/erase (CSI ... letter)
    text = re.sub(r'\x1b\[\?[0-9;]*[hl]', '', text)       # private modes (?25h, ?25l)

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
    metadata = metadata if isinstance(metadata, dict) else {}
    if not isinstance(data, dict):
        data = {'text/plain': _mime_text(data)}

    # HTML - render directly (trusted user code, matches Jupyter behavior)
    if 'text/html' in data:
        return f'<div class="mime-html" data-mime-type="text/html">{_mime_text(data["text/html"])}</div>'

    # SVG - render inline
    if 'image/svg+xml' in data:
        return f'<div class="mime-svg" data-mime-type="image/svg+xml">{_mime_text(data["image/svg+xml"])}</div>'

    # PNG image - base64 data URL
    if 'image/png' in data:
        return _image_html(data, metadata, 'image/png')

    # JPEG image
    if 'image/jpeg' in data:
        return _image_html(data, metadata, 'image/jpeg')

    # GIF image
    if 'image/gif' in data:
        return _image_html(data, metadata, 'image/gif')

    if 'image/webp' in data:
        return _image_html(data, metadata, 'image/webp')

    # Markdown - convert to HTML using mistlefoot for extended features
    if 'text/markdown' in data:
        try:
            from mistletoe import markdown as md_render
            from mistlefoot import ExtendedHtmlRenderer
            rendered = md_render(_mime_text(data['text/markdown']), ExtendedHtmlRenderer)
            return f'<div class="mime-markdown" data-mime-type="text/markdown">{rendered}</div>'
        except ImportError:
            return f'<pre class="mime-markdown" data-mime-type="text/markdown">{html_module.escape(_mime_text(data["text/markdown"]))}</pre>'

    # LaTeX - wrap for MathJax/KaTeX processing
    if 'text/latex' in data:
        return f'<div class="mime-latex" data-mime-type="text/latex">{html_module.escape(_mime_text(data["text/latex"]))}</div>'

    if 'application/vnd.jupyter.widget-view+json' in data:
        return (
            '<div class="mime-unsupported" data-mime-type="application/vnd.jupyter.widget-view+json">'
            'Interactive Jupyter widgets require a widget manager and cannot be rendered by Dialeng yet.'
            '</div>'
        )

    # JSON - pretty print
    if 'application/json' in data:
        value = data['application/json']
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        json_str = json.dumps(value, indent=2, ensure_ascii=False) if not isinstance(value, str) else value
        return f'<pre class="mime-json" data-mime-type="application/json">{html_module.escape(json_str)}</pre>'

    # Plain text fallback
    if 'text/plain' in data:
        return f'<pre class="mime-text" data-mime-type="text/plain">{html_module.escape(_mime_text(data["text/plain"]))}</pre>'

    # Unknown format - show raw
    return f'<pre class="mime-unknown">{html_module.escape(str(data))}</pre>'
