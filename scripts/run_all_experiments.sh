#!/usr/bin/env bash
# =============================================================================
# Cross-Lingual Consistency & Quantization — Full Experiment Command Reference
# =============================================================================
#
# Prerequisites (run once before any experiment):
#
#   cd "$(git rev-parse --show-toplevel)"
#   uv sync                          # core dependencies
#   uv sync --extra full             # optional: COMET + calibration backends
#   huggingface-cli login            # required for gated models (Llama, Gemma)
#
# Usage:
#
#   # Run the full grid (all models × precisions × tasks):
#   bash scripts/run_all_experiments.sh
#
#   # Run a single experiment family:
#   EXPERIMENT=profile  bash scripts/run_all_experiments.sh
#   EXPERIMENT=parity   bash scripts/run_all_experiments.sh
#   EXPERIMENT=reasoning bash scripts/run_all_experiments.sh
#   EXPERIMENT=translate bash scripts/run_all_experiments.sh
#   EXPERIMENT=calibrate bash scripts/run_all_experiments.sh
#   EXPERIMENT=plot     bash scripts/run_all_experiments.sh
#
#   # Limit to one model or precision:
#   MODEL=qwen3-8b PRECISION=int4 bash scripts/run_all_experiments.sh
#
# Environment overrides:
#   REASONING_MAX_SAMPLES  (default: 50)
#   TRANSLATE_MAX_SAMPLES  (default: 100)
#   OUTPUT_DIR             (default: results)
#
# Grid: 10 model families × up to 4 quantization levels = 39 configurations
# (mistral-small-3.1-24b omits fp16 due to memory constraints).
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${RUN:-uv run python -m src.main}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
REASONING_MAX_SAMPLES="${REASONING_MAX_SAMPLES:-50}"
TRANSLATE_MAX_SAMPLES="${TRANSLATE_MAX_SAMPLES:-100}"
EXPERIMENT="${EXPERIMENT:-all}"

# ── Model slugs (must match config/models.yaml) ──────────────────────────────
ALL_MODELS=(
  qwen3-8b
  qwen2.5-7b
  llama-3.1-8b
  gemma-3-4b
  gemma-3-12b
  gemma-2-9b
  phi-4-mini
  mistral-7b-v0.3
  mistral-small-3.1-24b
  deepseek-r1-distill-7b
)

models_to_run() {
  if [[ -n "${MODEL:-}" ]]; then
    echo "$MODEL"
  else
    printf '%s\n' "${ALL_MODELS[@]}"
  fi
}

precisions_for_model() {
  local model="$1"
  if [[ -n "${PRECISION:-}" ]]; then
    echo "$PRECISION"
    return
  fi
  case "$model" in
    mistral-small-3.1-24b) echo "int8 int4 int2" ;;
    *)                     echo "fp16 int8 int4 int2" ;;
  esac
}

run_cmd() {
  echo ""
  echo ">>> $*"
  eval "$@"
}

# ── §4.1 Latency & memory profiling ──────────────────────────────────────────
run_profile_experiments() {
  echo "=== §4.1 Latency & Memory Profiling ==="
  while IFS= read -r model; do
    for prec in $(precisions_for_model "$model"); do
      run_cmd "$RUN --model $model --precision $prec --output-dir $OUTPUT_DIR profile"
    done
  done < <(models_to_run)
}

# ── §4.2 Compression Parity ──────────────────────────────────────────────────
run_parity_experiments() {
  echo "=== §4.2 Compression Parity ==="
  while IFS= read -r model; do
    for prec in $(precisions_for_model "$model"); do
      run_cmd "$RUN --model $model --precision $prec --output-dir $OUTPUT_DIR parity"
    done
  done < <(models_to_run)
}

# ── §4.3 Cross-Lingual Consistency (CLC) on MGSM ────────────────────────────
run_reasoning_experiments() {
  echo "=== §4.3 CLC Reasoning (T=0.5) ==="
  while IFS= read -r model; do
    for prec in $(precisions_for_model "$model"); do
      run_cmd "$RUN --model $model --precision $prec --output-dir $OUTPUT_DIR reasoning \
        --max-samples $REASONING_MAX_SAMPLES --temperature 0.5"
    done
  done < <(models_to_run)

  echo "=== §4.3 CLC Reasoning Baseline (T=0.0) ==="
  while IFS= read -r model; do
    for prec in $(precisions_for_model "$model"); do
      run_cmd "$RUN --model $model --precision $prec --output-dir $OUTPUT_DIR reasoning \
        --max-samples $REASONING_MAX_SAMPLES --baseline"
    done
  done < <(models_to_run)
}

# ── §4.4 Machine Translation (FLORES-200 / WMT-style) ────────────────────────
run_translate_experiments() {
  echo "=== §4.4 Machine Translation (German, FLORES) ==="
  while IFS= read -r model; do
    for prec in $(precisions_for_model "$model"); do
      run_cmd "$RUN --model $model --precision $prec --output-dir $OUTPUT_DIR translate \
        --dataset google/fleurs \
        --config-name de_de \
        --split train \
        --max-samples $TRANSLATE_MAX_SAMPLES \
        --target-lang German"
    done
  done < <(models_to_run)
}

# ── §4.4 Target-language calibration (subset: int4 only) ─────────────────
run_calibration_experiments() {
  echo "=== §4.4 Target-Language Calibration (int4, requires --extra calibration) ==="
  local calib_methods=(autoround awq gptq)
  while IFS= read -r model; do
    for method in "${calib_methods[@]}"; do
      run_cmd "$RUN --model $model --precision int4 --output-dir $OUTPUT_DIR translate \
        --dataset google/fleurs \
        --config-name de_de \
        --split train \
        --max-samples $TRANSLATE_MAX_SAMPLES \
        --target-lang German \
        --calibrate $method"
    done
  done < <(models_to_run)
}

# ── Regenerate plots from saved JSON ─────────────────────────────────────────
run_plot_experiments() {
  echo "=== Regenerate Plots ==="
  while IFS= read -r model; do
    for prec in $(precisions_for_model "$model"); do
      local latency_json="$OUTPUT_DIR/latency/$model/$prec/profile.json"
      local parity_json="$OUTPUT_DIR/parity/$model/$prec/metrics.json"
      local reasoning_json="$OUTPUT_DIR/reasoning/$model/$prec/clc_results.json"

      if [[ -f "$latency_json" ]]; then
        run_cmd "$RUN --model $model --output-dir $OUTPUT_DIR plot \
          --input $latency_json --plot-type latency"
      fi
      if [[ -f "$parity_json" ]]; then
        run_cmd "$RUN --model $model --output-dir $OUTPUT_DIR plot \
          --input $parity_json --plot-type metrics"
      fi
      if [[ -f "$reasoning_json" ]]; then
        run_cmd "$RUN --model $model --output-dir $OUTPUT_DIR plot \
          --input $reasoning_json --plot-type metrics"
      fi
    done
  done < <(models_to_run)
}

# =============================================================================
# Explicit command list (39 configs × experiment families)
# Copy-paste individual commands below if you prefer not to use the loop runner.
# =============================================================================
#
# ── qwen3-8b ─────────────────────────────────────────────────────────────────
# uv run python -m src.main --model qwen3-8b --precision fp16 profile
# uv run python -m src.main --model qwen3-8b --precision int8  profile
# uv run python -m src.main --model qwen3-8b --precision int4  profile
# uv run python -m src.main --model qwen3-8b --precision int2  profile
# uv run python -m src.main --model qwen3-8b --precision fp16 parity
# uv run python -m src.main --model qwen3-8b --precision int8  parity
# uv run python -m src.main --model qwen3-8b --precision int4  parity
# uv run python -m src.main --model qwen3-8b --precision int2  parity
# uv run python -m src.main --model qwen3-8b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen3-8b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen3-8b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen3-8b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen3-8b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen3-8b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen3-8b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen3-8b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen3-8b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model qwen3-8b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model qwen3-8b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model qwen3-8b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── qwen2.5-7b ───────────────────────────────────────────────────────────────
# uv run python -m src.main --model qwen2.5-7b --precision fp16 profile
# uv run python -m src.main --model qwen2.5-7b --precision int8  profile
# uv run python -m src.main --model qwen2.5-7b --precision int4  profile
# uv run python -m src.main --model qwen2.5-7b --precision int2  profile
# uv run python -m src.main --model qwen2.5-7b --precision fp16 parity
# uv run python -m src.main --model qwen2.5-7b --precision int8  parity
# uv run python -m src.main --model qwen2.5-7b --precision int4  parity
# uv run python -m src.main --model qwen2.5-7b --precision int2  parity
# uv run python -m src.main --model qwen2.5-7b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen2.5-7b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen2.5-7b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen2.5-7b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model qwen2.5-7b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen2.5-7b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen2.5-7b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen2.5-7b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model qwen2.5-7b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model qwen2.5-7b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model qwen2.5-7b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model qwen2.5-7b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── llama-3.1-8b (gated — requires huggingface-cli login) ────────────────────
# uv run python -m src.main --model llama-3.1-8b --precision fp16 profile
# uv run python -m src.main --model llama-3.1-8b --precision int8  profile
# uv run python -m src.main --model llama-3.1-8b --precision int4  profile
# uv run python -m src.main --model llama-3.1-8b --precision int2  profile
# uv run python -m src.main --model llama-3.1-8b --precision fp16 parity
# uv run python -m src.main --model llama-3.1-8b --precision int8  parity
# uv run python -m src.main --model llama-3.1-8b --precision int4  parity
# uv run python -m src.main --model llama-3.1-8b --precision int2  parity
# uv run python -m src.main --model llama-3.1-8b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model llama-3.1-8b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model llama-3.1-8b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model llama-3.1-8b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model llama-3.1-8b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model llama-3.1-8b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model llama-3.1-8b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model llama-3.1-8b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model llama-3.1-8b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model llama-3.1-8b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model llama-3.1-8b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model llama-3.1-8b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── gemma-3-4b (gated) ───────────────────────────────────────────────────────
# uv run python -m src.main --model gemma-3-4b --precision fp16 profile
# uv run python -m src.main --model gemma-3-4b --precision int8  profile
# uv run python -m src.main --model gemma-3-4b --precision int4  profile
# uv run python -m src.main --model gemma-3-4b --precision int2  profile
# uv run python -m src.main --model gemma-3-4b --precision fp16 parity
# uv run python -m src.main --model gemma-3-4b --precision int8  parity
# uv run python -m src.main --model gemma-3-4b --precision int4  parity
# uv run python -m src.main --model gemma-3-4b --precision int2  parity
# uv run python -m src.main --model gemma-3-4b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-4b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-4b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-4b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-4b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-4b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-4b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-4b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-4b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-3-4b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-3-4b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-3-4b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── gemma-3-12b (gated; fp16 requires ≥24 GB UMA) ────────────────────────────
# uv run python -m src.main --model gemma-3-12b --precision fp16 profile
# uv run python -m src.main --model gemma-3-12b --precision int8  profile
# uv run python -m src.main --model gemma-3-12b --precision int4  profile
# uv run python -m src.main --model gemma-3-12b --precision int2  profile
# uv run python -m src.main --model gemma-3-12b --precision fp16 parity
# uv run python -m src.main --model gemma-3-12b --precision int8  parity
# uv run python -m src.main --model gemma-3-12b --precision int4  parity
# uv run python -m src.main --model gemma-3-12b --precision int2  parity
# uv run python -m src.main --model gemma-3-12b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-12b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-12b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-12b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-3-12b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-12b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-12b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-12b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-3-12b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-3-12b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-3-12b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-3-12b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── gemma-2-9b (gated) ───────────────────────────────────────────────────────
# uv run python -m src.main --model gemma-2-9b --precision fp16 profile
# uv run python -m src.main --model gemma-2-9b --precision int8  profile
# uv run python -m src.main --model gemma-2-9b --precision int4  profile
# uv run python -m src.main --model gemma-2-9b --precision int2  profile
# uv run python -m src.main --model gemma-2-9b --precision fp16 parity
# uv run python -m src.main --model gemma-2-9b --precision int8  parity
# uv run python -m src.main --model gemma-2-9b --precision int4  parity
# uv run python -m src.main --model gemma-2-9b --precision int2  parity
# uv run python -m src.main --model gemma-2-9b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-2-9b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-2-9b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-2-9b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model gemma-2-9b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-2-9b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-2-9b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-2-9b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model gemma-2-9b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-2-9b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-2-9b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model gemma-2-9b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── phi-4-mini ───────────────────────────────────────────────────────────────
# uv run python -m src.main --model phi-4-mini --precision fp16 profile
# uv run python -m src.main --model phi-4-mini --precision int8  profile
# uv run python -m src.main --model phi-4-mini --precision int4  profile
# uv run python -m src.main --model phi-4-mini --precision int2  profile
# uv run python -m src.main --model phi-4-mini --precision fp16 parity
# uv run python -m src.main --model phi-4-mini --precision int8  parity
# uv run python -m src.main --model phi-4-mini --precision int4  parity
# uv run python -m src.main --model phi-4-mini --precision int2  parity
# uv run python -m src.main --model phi-4-mini --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model phi-4-mini --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model phi-4-mini --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model phi-4-mini --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model phi-4-mini --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model phi-4-mini --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model phi-4-mini --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model phi-4-mini --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model phi-4-mini --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model phi-4-mini --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model phi-4-mini --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model phi-4-mini --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── mistral-7b-v0.3 ──────────────────────────────────────────────────────────
# uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 profile
# uv run python -m src.main --model mistral-7b-v0.3 --precision int8  profile
# uv run python -m src.main --model mistral-7b-v0.3 --precision int4  profile
# uv run python -m src.main --model mistral-7b-v0.3 --precision int2  profile
# uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 parity
# uv run python -m src.main --model mistral-7b-v0.3 --precision int8  parity
# uv run python -m src.main --model mistral-7b-v0.3 --precision int4  parity
# uv run python -m src.main --model mistral-7b-v0.3 --precision int2  parity
# uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-7b-v0.3 --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-7b-v0.3 --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-7b-v0.3 --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-7b-v0.3 --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-7b-v0.3 --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-7b-v0.3 --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model mistral-7b-v0.3 --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model mistral-7b-v0.3 --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model mistral-7b-v0.3 --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── mistral-small-3.1-24b (fp16 omitted — requires ≥48 GB UMA) ───────────────
# uv run python -m src.main --model mistral-small-3.1-24b --precision int8 profile
# uv run python -m src.main --model mistral-small-3.1-24b --precision int4 profile
# uv run python -m src.main --model mistral-small-3.1-24b --precision int2 profile
# uv run python -m src.main --model mistral-small-3.1-24b --precision int8 parity
# uv run python -m src.main --model mistral-small-3.1-24b --precision int4 parity
# uv run python -m src.main --model mistral-small-3.1-24b --precision int2 parity
# uv run python -m src.main --model mistral-small-3.1-24b --precision int8 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-small-3.1-24b --precision int4 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-small-3.1-24b --precision int2 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model mistral-small-3.1-24b --precision int8 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-small-3.1-24b --precision int4 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-small-3.1-24b --precision int2 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model mistral-small-3.1-24b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model mistral-small-3.1-24b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model mistral-small-3.1-24b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── deepseek-r1-distill-7b ───────────────────────────────────────────────────
# uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 profile
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int8  profile
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int4  profile
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int2  profile
# uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 parity
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int8  parity
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int4  parity
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int2  parity
# uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int8  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int4  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int2  reasoning --max-samples 50 --temperature 0.5
# uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 reasoning --max-samples 50 --baseline
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int8  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int4  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int2  reasoning --max-samples 50 --baseline
# uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int8  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int4  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
# uv run python -m src.main --model deepseek-r1-distill-7b --precision int2  translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
#
# ── Calibration examples (int4, one model shown; repeat for all models) ──────
# uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
# uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
# uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
# =============================================================================

case "$EXPERIMENT" in
  all)
    run_profile_experiments
    run_parity_experiments
    run_reasoning_experiments
    run_translate_experiments
    run_calibration_experiments
    run_plot_experiments
    ;;
  profile)    run_profile_experiments ;;
  parity)     run_parity_experiments ;;
  reasoning)  run_reasoning_experiments ;;
  translate)  run_translate_experiments ;;
  calibrate)  run_calibration_experiments ;;
  plot)       run_plot_experiments ;;
  *)
    echo "Unknown EXPERIMENT='$EXPERIMENT'. Use: all|profile|parity|reasoning|translate|calibrate|plot"
    exit 1
    ;;
esac

echo ""
echo "Done. Results saved under $OUTPUT_DIR/"
