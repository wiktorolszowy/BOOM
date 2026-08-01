#!/usr/bin/env bash
# Sequentially fine-tune MolFormer on remaining QM9 endpoints, one subprocess
# per endpoint. Each invocation writes its result to the shared results JSON
# via the runner's incremental save. Memory is reclaimed between endpoints
# because the process fully exits.
#
# HoF and Density already finished in the previous long-running process and
# are already in the JSON, so we skip them here.

set -u
# Script lives in reproduce/experiments/molformer/. All paths are relative
# to this directory.
cd "$(dirname "$0")"

PY=./.venv_molformer/bin/python
RUNNER=./run_molformer.py
EPOCHS=20
SEED=42
LOG_DIR=./logs/qm9
mkdir -p "$LOG_DIR"

ENDPOINTS=(homo lumo gap zpve r2 alpha mu cv)

echo "=== QM9 sequential driver started $(date '+%F %T') ==="
echo "epochs=$EPOCHS  seed=$SEED  endpoints=${ENDPOINTS[*]}"

for ep in "${ENDPOINTS[@]}"; do
    ts="$(date '+%F %T')"
    log="$LOG_DIR/${ep}_seed${SEED}.log"
    echo "[$ts] ---> starting $ep  (log: $log)"
    "$PY" -u "$RUNNER" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --endpoints "$ep" \
        --allow-mps \
        > "$log" 2>&1
    rc=$?
    ts_done="$(date '+%F %T')"
    if [[ $rc -ne 0 ]]; then
        echo "[$ts_done] !!! $ep FAILED (exit $rc), continuing"
    else
        echo "[$ts_done] <--- $ep done"
    fi
done

echo "=== QM9 sequential driver finished $(date '+%F %T') ==="
