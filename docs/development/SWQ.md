# Experimental SWQ prototype

`Q_SWQ_4` is an experimental CPU-only block-wise codebook format. Each block
contains 128 weights, 16 FP16 codebook values, and 128 packed 4-bit indices.
The block is 96 bytes, or 6 bits per weight. It is smaller than FP16, F32, and
Q8_0, but still larger than Q4_K_M.

The reference quantizer initializes the codebook across the block minimum and
maximum, then runs four Lloyd clustering iterations. CPU matrix multiplication
uses a scalar fused dot product against Q8_0 activations. GPU kernels are not
implemented.

Build:

```sh
cmake -B build -DGGML_METAL=OFF
cmake --build build --config Release -j
```

Convert an F16 or F32 GGUF model:

```sh
./build/bin/llama-quantize --swq-stats model-f16.gguf model-q-swq-4.gguf Q_SWQ_4
```

Load and run on CPU while printing SWQ memory statistics before inference:

```sh
./build/bin/llama-cli -m model-q-swq-4.gguf -ngl 0 --swq-stats -p "Hello" -n 32
```

Compare file size and CPU tokens/second against Q4_K_M and Q8_0. If a text file
is supplied, the helper also runs perplexity:

```sh
LLAMA_BIN_DIR=./build/bin tools/swq-bench.sh \
    model-q-swq-4.gguf model-q4-k-m.gguf model-q8-0.gguf \
    wiki.test.raw
```

The helper reports peak process RAM through `/usr/bin/time`, using the native
macOS or Linux output format.

Implementation locations:

- `ggml/src/ggml-common.h`: experimental block layout.
- `ggml/src/ggml-quants.c`: clustering, packing, and dequantization.
- `ggml/src/ggml-cpu/quants.c`: scalar CPU dot product.
- `src/llama-quant.cpp`: conversion selection and conversion statistics.
- `src/llama-model-loader.cpp`: load-time memory reporting.
