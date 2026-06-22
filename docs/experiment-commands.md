# Experiment Commands

Complete list of Python commands to run all cross-lingual consistency and quantization experiments.
Run each command from the project root after `uv sync` (and `uv sync --extra full` for COMET/calibration).

## Setup

```bash
uv sync
uv sync --extra full   # optional: COMET + calibration backends
huggingface-cli login  # required for gated models (Llama, Gemma)
```

Grid: 10 model families × up to 4 quantization levels = **39 configurations**.

## §4.1 Latency & Memory Profiling

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b --precision fp16 profile
```

```bash
uv run python -m src.main --model qwen3-8b --precision int8 profile
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 profile
```

```bash
uv run python -m src.main --model qwen3-8b --precision int2 profile
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b --precision fp16 profile
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int8 profile
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 profile
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int2 profile
```

### llama-3.1-8b
_Gated — requires `huggingface-cli login` and license acceptance._

```bash
uv run python -m src.main --model llama-3.1-8b --precision fp16 profile
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int8 profile
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 profile
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int2 profile
```

### gemma-3-4b
_Gated — requires `huggingface-cli login` and license acceptance._

```bash
uv run python -m src.main --model gemma-3-4b --precision fp16 profile
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int8 profile
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 profile
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int2 profile
```

### gemma-3-12b
_Gated — fp16 requires ≥24 GB UMA._

```bash
uv run python -m src.main --model gemma-3-12b --precision fp16 profile
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int8 profile
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 profile
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int2 profile
```

### gemma-2-9b
_Gated — requires `huggingface-cli login` and license acceptance._

```bash
uv run python -m src.main --model gemma-2-9b --precision fp16 profile
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int8 profile
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 profile
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int2 profile
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini --precision fp16 profile
```

```bash
uv run python -m src.main --model phi-4-mini --precision int8 profile
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 profile
```

```bash
uv run python -m src.main --model phi-4-mini --precision int2 profile
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 profile
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int8 profile
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 profile
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int2 profile
```

### mistral-small-3.1-24b
_fp16 omitted — requires ≥48 GB UMA._

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int8 profile
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 profile
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int2 profile
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 profile
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int8 profile
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 profile
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int2 profile
```

## §4.2 Compression Parity

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b --precision fp16 parity
```

```bash
uv run python -m src.main --model qwen3-8b --precision int8 parity
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 parity
```

```bash
uv run python -m src.main --model qwen3-8b --precision int2 parity
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b --precision fp16 parity
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int8 parity
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 parity
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int2 parity
```

### llama-3.1-8b
_Gated — requires `huggingface-cli login` and license acceptance._

```bash
uv run python -m src.main --model llama-3.1-8b --precision fp16 parity
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int8 parity
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 parity
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int2 parity
```

### gemma-3-4b
_Gated — requires `huggingface-cli login` and license acceptance._

```bash
uv run python -m src.main --model gemma-3-4b --precision fp16 parity
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int8 parity
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 parity
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int2 parity
```

### gemma-3-12b
_Gated — fp16 requires ≥24 GB UMA._

```bash
uv run python -m src.main --model gemma-3-12b --precision fp16 parity
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int8 parity
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 parity
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int2 parity
```

### gemma-2-9b
_Gated — requires `huggingface-cli login` and license acceptance._

```bash
uv run python -m src.main --model gemma-2-9b --precision fp16 parity
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int8 parity
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 parity
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int2 parity
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini --precision fp16 parity
```

```bash
uv run python -m src.main --model phi-4-mini --precision int8 parity
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 parity
```

```bash
uv run python -m src.main --model phi-4-mini --precision int2 parity
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 parity
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int8 parity
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 parity
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int2 parity
```

### mistral-small-3.1-24b
_fp16 omitted — requires ≥48 GB UMA._

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int8 parity
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 parity
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int2 parity
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 parity
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int8 parity
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 parity
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int2 parity
```

## §4.3 CLC Reasoning — MGSM (T=0.5)

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model qwen3-8b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model qwen3-8b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### llama-3.1-8b
```bash
uv run python -m src.main --model llama-3.1-8b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### gemma-3-4b
```bash
uv run python -m src.main --model gemma-3-4b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### gemma-3-12b
```bash
uv run python -m src.main --model gemma-3-12b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### gemma-2-9b
```bash
uv run python -m src.main --model gemma-2-9b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model phi-4-mini --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model phi-4-mini --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### mistral-small-3.1-24b
```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int8 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 reasoning --max-samples 50 --temperature 0.5
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int2 reasoning --max-samples 50 --temperature 0.5
```

## §4.3 CLC Reasoning — Baseline (T=0.0)

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model qwen3-8b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model qwen3-8b --precision int2 reasoning --max-samples 50 --baseline
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int2 reasoning --max-samples 50 --baseline
```

### llama-3.1-8b
```bash
uv run python -m src.main --model llama-3.1-8b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int2 reasoning --max-samples 50 --baseline
```

### gemma-3-4b
```bash
uv run python -m src.main --model gemma-3-4b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int2 reasoning --max-samples 50 --baseline
```

### gemma-3-12b
```bash
uv run python -m src.main --model gemma-3-12b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int2 reasoning --max-samples 50 --baseline
```

### gemma-2-9b
```bash
uv run python -m src.main --model gemma-2-9b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int2 reasoning --max-samples 50 --baseline
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model phi-4-mini --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model phi-4-mini --precision int2 reasoning --max-samples 50 --baseline
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int2 reasoning --max-samples 50 --baseline
```

### mistral-small-3.1-24b
```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int2 reasoning --max-samples 50 --baseline
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int8 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 reasoning --max-samples 50 --baseline
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int2 reasoning --max-samples 50 --baseline
```

## §4.4 Machine Translation (FLORES German)

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model qwen3-8b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model qwen3-8b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### llama-3.1-8b
```bash
uv run python -m src.main --model llama-3.1-8b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### gemma-3-4b
```bash
uv run python -m src.main --model gemma-3-4b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### gemma-3-12b
```bash
uv run python -m src.main --model gemma-3-12b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### gemma-2-9b
```bash
uv run python -m src.main --model gemma-2-9b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model phi-4-mini --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model phi-4-mini --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### mistral-small-3.1-24b
```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision fp16 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int8 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int2 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German
```

## §4.4 Target-Language Calibration (int4)

Requires `uv sync --extra calibration` or `--extra full`.

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model qwen3-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model qwen2.5-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### llama-3.1-8b
```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model llama-3.1-8b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### gemma-3-4b
```bash
uv run python -m src.main --model gemma-3-4b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model gemma-3-4b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### gemma-3-12b
```bash
uv run python -m src.main --model gemma-3-12b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model gemma-3-12b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### gemma-2-9b
```bash
uv run python -m src.main --model gemma-2-9b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model gemma-2-9b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model phi-4-mini --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### mistral-small-3.1-24b
```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate autoround
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate awq
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b --precision int4 translate --dataset google/fleurs --config-name de_de --split train --max-samples 100 --target-lang German --calibrate gptq
```

## Plot Regeneration

### qwen3-8b
```bash
uv run python -m src.main --model qwen3-8b plot --input results/latency/qwen3-8b/fp16/profile.json --plot-type latency
uv run python -m src.main --model qwen3-8b plot --input results/parity/qwen3-8b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model qwen3-8b plot --input results/reasoning/qwen3-8b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model qwen3-8b plot --input results/latency/qwen3-8b/int8/profile.json --plot-type latency
uv run python -m src.main --model qwen3-8b plot --input results/parity/qwen3-8b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model qwen3-8b plot --input results/reasoning/qwen3-8b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model qwen3-8b plot --input results/latency/qwen3-8b/int4/profile.json --plot-type latency
uv run python -m src.main --model qwen3-8b plot --input results/parity/qwen3-8b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model qwen3-8b plot --input results/reasoning/qwen3-8b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model qwen3-8b plot --input results/latency/qwen3-8b/int2/profile.json --plot-type latency
uv run python -m src.main --model qwen3-8b plot --input results/parity/qwen3-8b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model qwen3-8b plot --input results/reasoning/qwen3-8b/int2/clc_results.json --plot-type metrics
```

### qwen2.5-7b
```bash
uv run python -m src.main --model qwen2.5-7b plot --input results/latency/qwen2.5-7b/fp16/profile.json --plot-type latency
uv run python -m src.main --model qwen2.5-7b plot --input results/parity/qwen2.5-7b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model qwen2.5-7b plot --input results/reasoning/qwen2.5-7b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model qwen2.5-7b plot --input results/latency/qwen2.5-7b/int8/profile.json --plot-type latency
uv run python -m src.main --model qwen2.5-7b plot --input results/parity/qwen2.5-7b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model qwen2.5-7b plot --input results/reasoning/qwen2.5-7b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model qwen2.5-7b plot --input results/latency/qwen2.5-7b/int4/profile.json --plot-type latency
uv run python -m src.main --model qwen2.5-7b plot --input results/parity/qwen2.5-7b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model qwen2.5-7b plot --input results/reasoning/qwen2.5-7b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model qwen2.5-7b plot --input results/latency/qwen2.5-7b/int2/profile.json --plot-type latency
uv run python -m src.main --model qwen2.5-7b plot --input results/parity/qwen2.5-7b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model qwen2.5-7b plot --input results/reasoning/qwen2.5-7b/int2/clc_results.json --plot-type metrics
```

### llama-3.1-8b
```bash
uv run python -m src.main --model llama-3.1-8b plot --input results/latency/llama-3.1-8b/fp16/profile.json --plot-type latency
uv run python -m src.main --model llama-3.1-8b plot --input results/parity/llama-3.1-8b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model llama-3.1-8b plot --input results/reasoning/llama-3.1-8b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model llama-3.1-8b plot --input results/latency/llama-3.1-8b/int8/profile.json --plot-type latency
uv run python -m src.main --model llama-3.1-8b plot --input results/parity/llama-3.1-8b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model llama-3.1-8b plot --input results/reasoning/llama-3.1-8b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model llama-3.1-8b plot --input results/latency/llama-3.1-8b/int4/profile.json --plot-type latency
uv run python -m src.main --model llama-3.1-8b plot --input results/parity/llama-3.1-8b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model llama-3.1-8b plot --input results/reasoning/llama-3.1-8b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model llama-3.1-8b plot --input results/latency/llama-3.1-8b/int2/profile.json --plot-type latency
uv run python -m src.main --model llama-3.1-8b plot --input results/parity/llama-3.1-8b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model llama-3.1-8b plot --input results/reasoning/llama-3.1-8b/int2/clc_results.json --plot-type metrics
```

### gemma-3-4b
```bash
uv run python -m src.main --model gemma-3-4b plot --input results/latency/gemma-3-4b/fp16/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-4b plot --input results/parity/gemma-3-4b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-4b plot --input results/reasoning/gemma-3-4b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-3-4b plot --input results/latency/gemma-3-4b/int8/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-4b plot --input results/parity/gemma-3-4b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-4b plot --input results/reasoning/gemma-3-4b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-3-4b plot --input results/latency/gemma-3-4b/int4/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-4b plot --input results/parity/gemma-3-4b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-4b plot --input results/reasoning/gemma-3-4b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-3-4b plot --input results/latency/gemma-3-4b/int2/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-4b plot --input results/parity/gemma-3-4b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-4b plot --input results/reasoning/gemma-3-4b/int2/clc_results.json --plot-type metrics
```

### gemma-3-12b
```bash
uv run python -m src.main --model gemma-3-12b plot --input results/latency/gemma-3-12b/fp16/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-12b plot --input results/parity/gemma-3-12b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-12b plot --input results/reasoning/gemma-3-12b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-3-12b plot --input results/latency/gemma-3-12b/int8/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-12b plot --input results/parity/gemma-3-12b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-12b plot --input results/reasoning/gemma-3-12b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-3-12b plot --input results/latency/gemma-3-12b/int4/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-12b plot --input results/parity/gemma-3-12b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-12b plot --input results/reasoning/gemma-3-12b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-3-12b plot --input results/latency/gemma-3-12b/int2/profile.json --plot-type latency
uv run python -m src.main --model gemma-3-12b plot --input results/parity/gemma-3-12b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-3-12b plot --input results/reasoning/gemma-3-12b/int2/clc_results.json --plot-type metrics
```

### gemma-2-9b
```bash
uv run python -m src.main --model gemma-2-9b plot --input results/latency/gemma-2-9b/fp16/profile.json --plot-type latency
uv run python -m src.main --model gemma-2-9b plot --input results/parity/gemma-2-9b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-2-9b plot --input results/reasoning/gemma-2-9b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-2-9b plot --input results/latency/gemma-2-9b/int8/profile.json --plot-type latency
uv run python -m src.main --model gemma-2-9b plot --input results/parity/gemma-2-9b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-2-9b plot --input results/reasoning/gemma-2-9b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-2-9b plot --input results/latency/gemma-2-9b/int4/profile.json --plot-type latency
uv run python -m src.main --model gemma-2-9b plot --input results/parity/gemma-2-9b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-2-9b plot --input results/reasoning/gemma-2-9b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model gemma-2-9b plot --input results/latency/gemma-2-9b/int2/profile.json --plot-type latency
uv run python -m src.main --model gemma-2-9b plot --input results/parity/gemma-2-9b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model gemma-2-9b plot --input results/reasoning/gemma-2-9b/int2/clc_results.json --plot-type metrics
```

### phi-4-mini
```bash
uv run python -m src.main --model phi-4-mini plot --input results/latency/phi-4-mini/fp16/profile.json --plot-type latency
uv run python -m src.main --model phi-4-mini plot --input results/parity/phi-4-mini/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model phi-4-mini plot --input results/reasoning/phi-4-mini/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model phi-4-mini plot --input results/latency/phi-4-mini/int8/profile.json --plot-type latency
uv run python -m src.main --model phi-4-mini plot --input results/parity/phi-4-mini/int8/metrics.json --plot-type metrics
uv run python -m src.main --model phi-4-mini plot --input results/reasoning/phi-4-mini/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model phi-4-mini plot --input results/latency/phi-4-mini/int4/profile.json --plot-type latency
uv run python -m src.main --model phi-4-mini plot --input results/parity/phi-4-mini/int4/metrics.json --plot-type metrics
uv run python -m src.main --model phi-4-mini plot --input results/reasoning/phi-4-mini/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model phi-4-mini plot --input results/latency/phi-4-mini/int2/profile.json --plot-type latency
uv run python -m src.main --model phi-4-mini plot --input results/parity/phi-4-mini/int2/metrics.json --plot-type metrics
uv run python -m src.main --model phi-4-mini plot --input results/reasoning/phi-4-mini/int2/clc_results.json --plot-type metrics
```

### mistral-7b-v0.3
```bash
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/latency/mistral-7b-v0.3/fp16/profile.json --plot-type latency
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/parity/mistral-7b-v0.3/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/reasoning/mistral-7b-v0.3/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/latency/mistral-7b-v0.3/int8/profile.json --plot-type latency
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/parity/mistral-7b-v0.3/int8/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/reasoning/mistral-7b-v0.3/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/latency/mistral-7b-v0.3/int4/profile.json --plot-type latency
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/parity/mistral-7b-v0.3/int4/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/reasoning/mistral-7b-v0.3/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/latency/mistral-7b-v0.3/int2/profile.json --plot-type latency
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/parity/mistral-7b-v0.3/int2/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-7b-v0.3 plot --input results/reasoning/mistral-7b-v0.3/int2/clc_results.json --plot-type metrics
```

### mistral-small-3.1-24b
```bash
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/latency/mistral-small-3.1-24b/int8/profile.json --plot-type latency
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/parity/mistral-small-3.1-24b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/reasoning/mistral-small-3.1-24b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/latency/mistral-small-3.1-24b/int4/profile.json --plot-type latency
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/parity/mistral-small-3.1-24b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/reasoning/mistral-small-3.1-24b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/latency/mistral-small-3.1-24b/int2/profile.json --plot-type latency
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/parity/mistral-small-3.1-24b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model mistral-small-3.1-24b plot --input results/reasoning/mistral-small-3.1-24b/int2/clc_results.json --plot-type metrics
```

### deepseek-r1-distill-7b
```bash
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/latency/deepseek-r1-distill-7b/fp16/profile.json --plot-type latency
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/parity/deepseek-r1-distill-7b/fp16/metrics.json --plot-type metrics
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/reasoning/deepseek-r1-distill-7b/fp16/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/latency/deepseek-r1-distill-7b/int8/profile.json --plot-type latency
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/parity/deepseek-r1-distill-7b/int8/metrics.json --plot-type metrics
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/reasoning/deepseek-r1-distill-7b/int8/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/latency/deepseek-r1-distill-7b/int4/profile.json --plot-type latency
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/parity/deepseek-r1-distill-7b/int4/metrics.json --plot-type metrics
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/reasoning/deepseek-r1-distill-7b/int4/clc_results.json --plot-type metrics
```

```bash
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/latency/deepseek-r1-distill-7b/int2/profile.json --plot-type latency
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/parity/deepseek-r1-distill-7b/int2/metrics.json --plot-type metrics
uv run python -m src.main --model deepseek-r1-distill-7b plot --input results/reasoning/deepseek-r1-distill-7b/int2/clc_results.json --plot-type metrics
```
