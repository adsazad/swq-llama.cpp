# Experimental SWQ prototype

`Q_SWQ_4` is an experimental CPU-only block-wise codebook format. Each block
contains 128 weights, 16 FP16 codebook values, and 128 packed 4-bit indices.
The block is 96 bytes, or 6 bits per weight. It is smaller than FP16, F32, and
Q8_0, but still larger than Q4_K_M.

`Q_SWQ_FIT_2` is a separate experimental format. Each block contains 128
weights represented by four FP16 cubic-fit coefficients, four FP16 residual
values, and 128 packed 2-bit residual indices. The block is 48 bytes, or 3 bits
per weight.

`Q_SWQ_FIT_3` is a second equation-based experimental format. Each block
contains 128 weights represented by four FP16 cubic-fit coefficients, eight
FP16 residual values, and 128 packed 3-bit residual indices. The block is 72
bytes, or 4.5 bits per weight. This is intended to keep compression near the
70% target while reducing the large reconstruction error seen in `Q_SWQ_FIT_2`.

The reference quantizer initializes the codebook across the block minimum and
maximum, then runs four Lloyd clustering iterations. CPU matrix multiplication
uses scalar fused dot products against Q8_0 activations. GPU kernels are not
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

Convert to the equation-based FIT experiment:

```sh
./build/bin/llama-quantize --swq-stats model-f16.gguf model-q-swq-fit-2.gguf Q_SWQ_FIT_2
```

Convert to the 3-bit FIT experiment:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 72 \
    --swq-fit-residual-epochs 4 \
    model-f16.gguf \
    model-q-swq-fit-3.gguf \
    Q_SWQ_FIT_3
```

Configure FIT conversion epochs without rebuilding:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 24 \
    --swq-fit-residual-epochs 4 \
    --swq-fit-progress \
    model-f16.gguf \
    model-q-swq-fit-2.gguf \
    Q_SWQ_FIT_2
```

`--swq-fit-progress` prints per-tensor, per-epoch reconstruction stats:

```text
SWQ_FIT_2 epochs: tensor=blk.0.attn_k.weight blocks=896 fit_epochs=2 residual_epochs=1
  epoch   1/  2 - rmse 0.029877587 - rel_rmse 0.47250399
  epoch   2/  2 - rmse 0.027121888 - rel_rmse 0.42892354
```

For clean epoch logs, FIT progress mode runs FIT quantization single-threaded.

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
- `tools/swq-accuracy.py`: tensor and layer reconstruction-error reporting.
- `tools/swq-layer-report.py`: standalone HTML graphs for original-vs-predicted weights.

Generate layer-by-layer graphs:

```sh
python3 tools/swq-layer-report.py \
    model-f16.gguf \
    model-q-swq-fit-2.gguf \
    --out swq-layer-report.html
```

The report shows per-layer relative RMSE, cosine similarity, and original vs
predicted weight curves for representative tensors in each layer.
