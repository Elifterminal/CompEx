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

# A copy on the Desktop, if there is one, with an ABSOLUTE icon path.
#
# The menu entry uses the themed name `Icon=compex`, which is correct and what
# the spec wants. The desktop copy cannot: GNOME Shell caches the icon theme at
# startup, so a freshly installed theme icon shows as a generic placeholder
# until the next login, and on Wayland the shell cannot be restarted without
# logging out. An absolute path skips theme lookup and works immediately.
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
  sed -e "s|__EXEC__|$ROOT/run_compex.sh|" \
      -e "s|^Icon=compex$|Icon=$THEME_DIR/256x256/apps/compex.png|" \
      "$HERE/compex.desktop" > "$DESKTOP_DIR/compex.desktop"
  chmod +x "$DESKTOP_DIR/compex.desktop"
  # GNOME will not run an untrusted launcher; it opens it as a text file.
  gio set "$DESKTOP_DIR/compex.desktop" metadata::trusted true 2>/dev/null || true
  DESKTOP_COPY="$DESKTOP_DIR/compex.desktop"
fi

# hicolor needs an index.theme to be a valid theme; without one
# gtk-update-icon-cache refuses and lookups can fall back to a generic icon.
if [ ! -f "$THEME_DIR/index.theme" ]; then
  cat > "$THEME_DIR/index.theme" <<'THEME'
[Icon Theme]
Name=Hicolor
Comment=Fallback icon theme
Hidden=true
Directories=16x16/apps,24x24/apps,32x32/apps,48x48/apps,64x64/apps,128x128/apps,256x256/apps,512x512/apps,scalable/apps
THEME
  for size in 16 24 32 48 64 128 256 512; do
    printf '\n[%sx%s/apps]\nSize=%s\nContext=Applications\nType=Fixed\n' \
      "$size" "$size" "$size" >> "$THEME_DIR/index.theme"
  done
  printf '\n[scalable/apps]\nSize=48\nMinSize=8\nMaxSize=512\nContext=Applications\nType=Scalable\n' \
    >> "$THEME_DIR/index.theme"
fi

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
[ -n "${DESKTOP_COPY:-}" ] && echo "  $DESKTOP_COPY"
echo
echo "Look for 'CompEx' in your application menu."
