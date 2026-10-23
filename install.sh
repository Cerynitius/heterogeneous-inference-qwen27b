#!/bin/bash
# Creates the venv, installs vLLM 0.30.0 and its runtime pieces, overlays the modified vLLM files from ./vllm,
# and puts b12x 1.3.0 into third_party/b12x (outside the venv: it pins an older nvidia-cutlass-dsl than vLLM).
#   VENV    venv dir (.venv)          PYTHON  interpreter for the venv (python3.12)
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$HERE/.venv}
PYTHON=${PYTHON:-python3.12}
[[ -x $VENV/bin/python ]] || "$PYTHON" -m venv "$VENV"
P=$VENV/bin/python
"$P" -m pip install -U pip
"$P" -m pip install "vllm==0.30.0" "nixl[cu13]==1.5.0" aiohttp cuda-python ninja
SP=$("$P" -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
[[ $("$P" -c "import vllm; print(vllm.__version__)") == 0.30.0* ]] || { echo "vllm 0.30.0 expected"; exit 1; }
(cd "$HERE/vllm" && find . -name "*.py") | while read -r f; do
  install -D -m 644 "$HERE/vllm/$f" "$SP/$f"
done
"$P" -m pip install --no-deps --target "$HERE/third_party/b12x" "b12x==1.3.0"
echo "installed: $VENV  (vllm overlay -> $SP)"
