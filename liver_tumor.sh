#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -r .venv/bin/activate ]]; then
    printf 'No se encontró el entorno virtual en %s/.venv\n' "$PWD" >&2
    exit 1
fi

source .venv/bin/activate
exec python3 scripts/gui.py
