#!/usr/bin/env bash
# Install the CompEx desktop entry and icon for the current user.
# Safe to re-run: it overwrites its own files and touches nothing else.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

APPS_DIR="$HOME/.local/share/applications"
THEME_DIR="$HOME/.local/share/icons/hicolor"
ICON_DIR="$THEME_DIR/scalable/apps"

mkdir -p "$APPS_DIR" "$ICON_DIR"

install -m 0644 "$ROOT/src/compex/ui/icon.svg" "$ICON_DIR/compex.svg"

# Pre-rendered PNGs as well as the SVG. Some panels and docks prefer a raster at
# the exact size, and shipping them means the installer does not need an SVG
# rasteriser on the box it runs on.
for size in 16 24 32 48 64 128 256 512; do
  png="$ROOT/src/compex/ui/icons/compex-${size}.png"
  [ -f "$png" ] || continue
  mkdir -p "$THEME_DIR/${size}x${size}/apps"
  install -m 0644 "$png" "$THEME_DIR/${size}x${size}/apps/compex.png"
done

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
echo "  $THEME_DIR/{16,24,32,48,64,128,256,512}x*/apps/compex.png"
echo
echo "Look for 'CompEx' in your application menu."
