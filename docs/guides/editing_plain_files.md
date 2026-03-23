# Editing Plain Files In Dialeng

Dialeng can now open plain-text files directly from the file explorer.

## Supported Files

Typical text-based files open in a standalone editor page, for example:

- `.md`
- `.json`
- `.txt`
- `.py`
- `.js`
- `.ts`
- `.html`
- `.css`
- `.yaml`
- `.toml`

Notebook files (`.ipynb`) still open in normal dialog/notebook mode.

## Unsupported Files

Binary-like files are shown with a message instead of the editor. Examples:

- images
- zip files
- compiled binaries
- files Dialeng cannot decode as plain text

## Single-Writer Behavior

Plain-file editing is intentionally not collaborative.

- the first Dialeng browser tab to open a text file gets the edit lock
- another tab or user opening the same file sees a locked message instead of the editor
- if the first tab closes or disappears, the lock expires automatically after a short timeout

## Save Behavior

- use the toolbar save button
- or press `Ctrl/Cmd+S`

## Explorer Behavior

The file explorer now shows:

- folders
- notebooks
- other files

Clicking behavior:

- notebook file -> opens notebook/dialog mode
- text file -> opens the standalone file editor page
- non-editable file -> opens the standalone file page with a non-editable message
