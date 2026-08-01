#!/usr/bin/env bash
# Install the CompEx desktop entry and icon for the current user.
# Safe to re-run: it overwrites its own files and touches nothing else.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

APPS_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"

mkdir -p "$APPS_DIR" "$ICON_DIR"

install -m 0644 "$ROOT/src/compex/ui/icon.svg" "$ICON_DIR/compex.svg"
chmod +x "$ROOT/run_compex.sh"

sed "s|__EXEC__|$ROOT/run_compex.sh|" "$HERE/compex.desktop" > "$APPS_DIR/compex.desktop"
chmod 0644 "$APPS_DIR/compex.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
fi

echo "installed:"
echo "  $APPS_DIR/compex.desktop"
echo "  $ICON_DIR/compex.svg"
echo
echo "Look for 'CompEx' in your application menu."
