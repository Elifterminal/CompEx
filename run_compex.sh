#!/usr/bin/env bash
# Launch the Compex UI. Works whether or not the package has been pip-installed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

# A source checkout always wins, and the test is for the file rather than for
# whether `import compex` works.
#
# The previous check was `python3 -c "import compex"`, and this repository's own
# directory name defeats it. The checkout lives at $HOME/compex; a desktop
# launcher runs with the working directory set to $HOME; Python puts the working
# directory first on sys.path and treats any matching directory without an
# __init__.py as a namespace package. So `import compex` SUCCEEDED — resolving
# to the repo folder itself, with __file__ None and no submodules — the script
# concluded the package was installed, never added src/ to the path, and then
# `python3 -m compex.ui.server` died with "No module named compex.ui".
#
# From a terminal it worked, because the working directory was almost never
# $HOME. Double-clicking the icon did nothing at all, because Terminal=false
# throws the traceback away.
if [ -f "$HERE/src/compex/__init__.py" ]; then
  export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# Surface a failure instead of vanishing. A desktop launcher has nowhere to
# print to, so a crash at startup is indistinguishable from nothing happening.
if ! python3 -c "import compex.ui.server" >/dev/null 2>&1; then
  why="$(python3 -c "import compex.ui.server" 2>&1 | tail -1)"
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="CompEx" --width=460 \
      --text="CompEx could not start.\n\n$why\n\nTried: $HERE/src" 2>/dev/null
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send -u critical "CompEx could not start" "$why"
  fi
  echo "CompEx could not start: $why" >&2
  exit 1
fi

exec python3 -m compex.ui.server "$@"
