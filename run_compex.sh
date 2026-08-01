#!/usr/bin/env bash
# Launch the Compex UI. Works whether or not the package has been pip-installed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

if ! python3 -c "import compex" >/dev/null 2>&1; then
  export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
fi

exec python3 -m compex.ui.server "$@"
