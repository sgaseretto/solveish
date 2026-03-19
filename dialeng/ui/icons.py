"""
Dialeng UI - Lucide Icon Sprites

Shared SvgSprites instance for consistent icon rendering across components.
Icons use <use href="#name"> referencing a sprite sheet included in the page.
"""

from fastlucide import SvgSprites

# Shared sprite instance — register all icons used in the UI
sprites = SvgSprites(sz=16)

# Pre-register all icons so the sprite sheet includes their definitions
_ICON_NAMES = [
    'pin', 'pin-off',
    'eye', 'eye-closed',
    'bookmark', 'bookmark-check',
    # Kernel selection
    'house-plug', 'microchip', 'cpu', 'monitor', 'zap', 'circle', 'check',
    # File explorer
    'notebook', 'notebook-text', 'folder', 'folder-open',
    'file-plus', 'folder-plus', 'panel-left-close', 'panel-left-open',
    'chevron-right', 'trash', 'refresh-cw',
    # Toolbar actions
    'sun', 'moon', 'save', 'download', 'settings', 'rotate-ccw',
    # Cell actions
    'play', 'square', 'arrow-up', 'arrow-down', 'trash-2', 'plus', 'chevron-down',
    # Safe mode
    'shield-check', 'shield-off',
    # Settings
    'triangle-alert',
    # Keyboard shortcuts
    'keyboard',
]
for _name in _ICON_NAMES:
    sprites(_name)
