#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 SWQ.gguf Q4_K_M.gguf Q8_0.gguf [perplexity-text]"
    exit 1
fi

bin_dir="${LLAMA_BIN_DIR:-./build/bin}"

run_with_memory() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        /usr/bin/time -l "$@"
    else
        /usr/bin/time -v "$@"
    fi
}

for model in "$1" "$2" "$3"; do
    echo "model: $model"
    stat -f "file bytes: %z" "$model" 2>/dev/null || stat -c "file bytes: %s" "$model"
    run_with_memory "$bin_dir/llama-bench" -m "$model" -ngl 0 -p 512 -n 128
    if [[ $# -eq 4 ]]; then
        "$bin_dir/llama-perplexity" -m "$model" -ngl 0 -f "$4"
    fi
done
