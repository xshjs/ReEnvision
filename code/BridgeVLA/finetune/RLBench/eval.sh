cd finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_bridgevla}"
export PYTHONPATH="$(pwd):$(pwd)/bridgevla/libs/RLBench:$(pwd)/bridgevla/libs/PyRep:$(pwd)/bridgevla/libs/YARR:$(pwd)/bridgevla/libs/peract:$(pwd)/bridgevla/libs/peract_colab:$(pwd)/bridgevla/libs/point-renderer${PYTHONPATH:+:$PYTHONPATH}"
cd RLBench

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FOLDER="${MODEL_FOLDER:-/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench}"
EVAL_DATAFOLDER="${EVAL_DATAFOLDER:-/mnt/workspace/manipulation/BridgeVLA/finetune/rlbench_eval_subset}"
TASKS="${TASKS:-all}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
LOG_NAME="${LOG_NAME:-debug_opensource_our_train}"
DEVICE="${DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-model_80.pth}"
EXTRA_EVAL_ARGS=()
if [ "${BRIDGEVLA_COLLECT_REPAIR_AFTER_FAILURE:-0}" = "1" ]; then
 EXTRA_EVAL_ARGS+=(--collect-repair-after-failure)
fi
echo "[eval.sh] using python: $(which "$PYTHON_BIN")"
"$PYTHON_BIN" - <<'PY'
import sys
print(f"[eval.sh] python version: {sys.version.splitlines()[0]}")
for name in ("torch", "transformers", "numpy"):
    mod = __import__(name)
    print(f"[eval.sh] {name}={getattr(mod, '__version__', 'unknown')}")
PY


# Do not mutate the active Python environment during evaluation.
# Recent opencv-python-headless wheels may pull in numpy>=2 and break
# the Paligemma/transformers stack used by BridgeVLA.
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac'  "$PYTHON_BIN" eval.py \
 --model-folder "$MODEL_FOLDER" \
 --eval-datafolder "$EVAL_DATAFOLDER" \
 --tasks $TASKS \
 --eval-episodes "$EVAL_EPISODES" \
 --log-name "$LOG_NAME" \
 --device "$DEVICE" \
 --headless \
 --model-name "$MODEL_NAME" \
 "${EXTRA_EVAL_ARGS[@]}"
