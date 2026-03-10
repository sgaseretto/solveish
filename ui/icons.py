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
]
for _name in _ICON_NAMES:
    sprites(_name)
