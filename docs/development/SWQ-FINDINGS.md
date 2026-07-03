# SWQ prototype findings

Date: 2026-07-03

## Test system

- Model: Qwen2.5-0.5B-Instruct FP16 GGUF
- Source: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- Build: `b9860-fdb1db877`
- Platform: macOS arm64, CPU-only build
- SWQ layout: 128 weights, 16 FP16 codebook entries, 128 packed 4-bit indices
- SWQ block size: 96 bytes, or 6 bits per weight

## Accomplishments

- Added the experimental `Q_SWQ_4` GGML and llama file types.
- Added FP16/F32-to-SWQ conversion using min/max initialization and four Lloyd iterations.
- Added GGUF loading and scalar CPU inference support.
- Added `--swq-stats` conversion and load-time reporting support.
- Added per-tensor conversion statistics.
- Added the `tools/swq-bench.sh` comparison helper.
- Fixed missing SWQ data validation found by the first conversion test.
- Fixed missing quantized `GET_ROWS` dispatch found by the first inference test.
- Added a first fused scalar `Q_SWQ_4 x Q8_0` CPU dot-product path so runtime
  activations use Q8_0 instead of F32.
- Built `llama-cli`, `llama-quantize`, `llama-bench`, and `llama-perplexity` successfully.

## Conversion command

```sh
./build/bin/llama-quantize --swq-stats \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-4.gguf \
    Q_SWQ_4
```

Conversion completed in 22.30 seconds wall time. One tensor, `output.weight`,
fell back to Q8_0 because its 896 columns are not divisible by the Q6_K block
requirement selected by the existing output-tensor policy. The other eligible
matrix tensors were converted to SWQ.

## Compression result

| Measurement | FP16 | SWQ | Change |
|---|---:|---:|---:|
| Tensor bytes reported by converter | 1,260,477,952 | 885,871,104 | 374,606,848 fewer |
| GGUF file bytes | 1,266,425,696 | 891,818,848 | 374,606,848 fewer |
| GGUF file size | 1,207.76 MiB | 850.50 MiB | 357.26 MiB fewer |
| Whole-file compression | - | 1.420x | 29.58% saved |
| Converter BPW | 16.00 | 11.25 | 29.69% lower |

An individual FP16 tensor converted entirely to SWQ saves exactly 25% because
64 bytes of FP16 weights become a 48-byte SWQ block. Whole-model savings are
higher because `output.weight` fell back from FP16 to Q8_0.

## Smoke test

The CLI automatically enables conversation mode for this model. `--single-turn`
was added so the predefined prompt exits after one response.

```sh
./build/bin/llama-cli \
    -m models/swq/qwen2.5-0.5b-instruct-q-swq-4.gguf \
    -ngl 0 --swq-stats --single-turn \
    -p "The capital of India is" \
    -n 16 -t 4 --temp 0
```

Exact generated text:

```text
The capital of India is New Delhi.
```

- Model loads: yes
- Tokenizer works: yes
- Generation works: yes
- Output is coherent: yes
- Crash: no

## RAM and speed

Both models used the same prompt, token limit, thread count, temperature, CPU
offload setting, and `--single-turn` behavior. Peak RSS came from
`/usr/bin/time -l`.

| Measurement | FP16 | SWQ | Change |
|---|---:|---:|---:|
| Maximum resident set size | 1,453,572,096 bytes | 1,063,698,432 bytes | 389,873,664 bytes fewer |
| Maximum resident set size | 1,386.23 MiB | 1,014.42 MiB | 371.81 MiB fewer |
| RAM ratio | - | 1.367x | 26.82% saved |
| Prompt speed | 41.7 tokens/sec | 37.1 tokens/sec | 11.03% slower |
| Generation speed | 12.4 tokens/sec | 3.0 tokens/sec | 75.81% slower |
| Wall time | 4.35 seconds | 8.37 seconds | 92.41% longer |

## Q8_0 activation kernel follow-up

After adding the scalar `Q_SWQ_4 x Q8_0` dot path, `llama-cli` and
`llama-quantize` were rebuilt successfully. The same 16-token smoke prompt was
then rerun sequentially with `/usr/bin/time -l`.

| Measurement | FP16 | SWQ with Q8_0 activations | Change |
|---|---:|---:|---:|
| Maximum resident set size | 1,437,073,408 bytes | 1,097,121,792 bytes | 339,951,616 bytes fewer |
| Maximum resident set size | 1,370.50 MiB | 1,046.30 MiB | 324.20 MiB fewer |
| RAM ratio | - | 1.310x | 23.66% saved |
| Prompt speed | 95.2 tokens/sec | 74.2 tokens/sec | 22.06% slower |
| Generation speed | 16.9 tokens/sec | 3.9 tokens/sec | 76.92% slower |
| Wall time | 3.07 seconds | 5.82 seconds | 89.58% longer |

Exact generated text remained:

```text
The capital of India is New Delhi.
```

This confirms the Q8_0 activation path works and improves SWQ generation from
the earlier 3.0 tokens/sec to 3.9 tokens/sec on this short run. It is still much
slower than FP16 because the SWQ kernel is scalar and still performs codebook
selection for every weight. The next speed step would need ARM NEON or a
different block layout that reduces codebook lookup overhead.

## SWQ128 and mixed-format follow-up

The SWQ block was changed from 32 weights to 128 weights while keeping 16 FP16
codebook entries and 4-bit indices. This changes the core block size from 48
bytes per 32 weights, or 12 bpw, to 96 bytes per 128 weights, or 6 bpw.

Full SWQ128 conversion:

```sh
./build/bin/llama-quantize --swq-stats \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-4-128.gguf \
    Q_SWQ_4
```

Full SWQ128 compressed much better but was still too slow:

| Model | File bytes | Max RSS | Prompt speed | Generation speed |
|---|---:|---:|---:|---:|
| FP16 | 1,266,425,696 | 1,437,073,408 | 95.2 t/s | 16.9 t/s |
| Full SWQ128 | 521,347,936 | 726,646,784 | 55.1 t/s | 2.8 t/s |

Converter summary for full SWQ128:

- Quant size: 491.52 MiB, 6.54 BPW
- SWQ tensor bytes: 515,400,192
- SWQ compression ratio: 2.45x
- SWQ percentage saved: 59.11%

A bucketed scalar dot-product experiment was also tried, but it reduced full
SWQ128 generation speed to 2.5 t/s, so it was not kept.

To meet the local speed target of more than 10 generation tokens/sec, a mixed
model was tested. Most tensors were Q8_0, while only attention K/V tensors used
SWQ128:

```sh
./build/bin/llama-quantize --swq-stats \
    --tensor-type attn_k=q_swq_4 \
    --tensor-type attn_v=q_swq_4 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q8-swq-kv-128.gguf \
    Q8_0
```

This met the speed target:

| Model | File bytes | Max RSS | Prompt speed | Generation speed |
|---|---:|---:|---:|---:|
| Q8_0 | 675,710,816 | 1,265,451,008 | 265.1 t/s | 16.4 t/s |
| Q8_0 + SWQ128 K/V | 673,990,496 | 1,208,811,520 | 119.0 t/s | 14.9 t/s |
| Q8_0 + SWQ128 attention | 661,948,256 | 1,184,022,528 | 98.0 t/s | 8.0 t/s |

The K/V-only mixed model saves little file size compared with Q8_0, but it does
save about 54.02 MiB peak RSS on this run while staying above 10 generation
tokens/sec. Expanding SWQ to all attention tensors saves more RAM and file size
but falls below the speed target.

## Equation-based SWQ FIT follow-up

A separate experimental type, `Q_SWQ_FIT_2`, was added. It stores each
128-weight block as a cubic equation plus 2-bit residuals:

```text
4 FP16 cubic coefficients
4 FP16 residual values
128 packed 2-bit residual indices
48 bytes per 128 weights
3 bpw
```

Full `Q_SWQ_FIT_2` conversion:

```sh
./build/bin/llama-quantize --swq-stats \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-fit-2.gguf \
    Q_SWQ_FIT_2
```

Full `Q_SWQ_FIT_2` compressed strongly, but output quality and speed were bad:

| Model | File bytes | Max RSS | Prompt speed | Generation speed | Output |
|---|---:|---:|---:|---:|---|
| Full Q_SWQ_FIT_2 | 336,112,480 | 545,701,888 | 62.4 t/s | 1.2 t/s | incoherent |

Converter summary:

- Quant size: 314.87 MiB, 4.19 BPW
- SWQ tensor bytes: 330,164,736
- SWQ compression ratio: 3.82x
- SWQ percentage saved: 73.81%

The smoke output was:

```text
. I. I.
.
The and

2
```

Layer-by-layer approximation error was measured with:

```sh
python3 tools/swq-accuracy.py \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-fit-2.gguf \
    --csv models/swq/accuracy-swq-fit-2.csv
```

Layer relative RMSE was roughly 0.35 across most transformer blocks, and the
worst tensors were above 0.42 relative RMSE. This confirms that the cubic
equation plus 2-bit residual approximation is too lossy for full-model use.

Selective K/V-only `Q_SWQ_FIT_2` over a Q8_0 base:

```sh
./build/bin/llama-quantize --swq-stats \
    --tensor-type attn_k=q_swq_fit_2 \
    --tensor-type attn_v=q_swq_fit_2 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q8-swq-fit-kv.gguf \
    Q8_0
```

| Model | File bytes | Max RSS | Prompt speed | Generation speed | Output |
|---|---:|---:|---:|---:|---|
| Q8_0 + Q_SWQ_FIT_2 K/V | 671,926,112 | 1,257,635,840 | 139.7 t/s | 10.8 t/s | coherent |

This is above the 10 t/s target, but it only saved about 7.45 MiB peak RSS
versus Q8_0 in this run. The earlier `Q8_0 + Q_SWQ_4 K/V` model remains the
better usable result.

### Conversion-time FIT epochs

`Q_SWQ_FIT_2` conversion was updated to run alternating fit epochs. Each block
now repeats:

```text
fit cubic equation
assign residual indices
update residual codebook
refit cubic equation against residual-corrected weights
```

This does not change the file layout or model size. It only spends more time
during conversion to improve the coefficients and residual assignments.

Full `Q_SWQ_FIT_2` with conversion-time epochs:

| Model | File bytes | Max RSS | Prompt speed | Generation speed | Output |
|---|---:|---:|---:|---:|---|
| Full Q_SWQ_FIT_2, epochs | 336,112,480 | 544,587,776 | 95.9 t/s | 2.7 t/s | incoherent |

Compared with the first FIT attempt:

- conversion time increased from about 5.47s to about 22.52s
- layer relative RMSE improved from roughly 0.35 to roughly 0.32
- worst-tensor relative RMSE improved from above 0.42 to about 0.39
- generation speed improved from 1.2 t/s to 2.7 t/s
- output remained incoherent

Updated graph report:

```text
models/swq/swq-fit-2-epochs-layer-report.html
```

The epoch system helps, but the 3 bpw equation-plus-2-bit-residual format is
still too inaccurate for full-model use.

### Higher epoch run: 24 fit epochs x 4 residual iterations

The FIT converter was then increased from:

```text
6 fit epochs x 2 residual iterations
```

to:

```text
24 fit epochs x 4 residual iterations
```

This is much more expensive during conversion but keeps the same file format and
same 3 bpw block size.

Full `Q_SWQ_FIT_2` 24x4 result:

| Model | File bytes | Max RSS | Prompt speed | Generation speed | Output |
|---|---:|---:|---:|---:|---|
| Full Q_SWQ_FIT_2, 24x4 | 336,112,480 | 548,159,488 | 10.2 t/s | 1.1 t/s | incoherent |

Conversion time:

```text
6x2:  22.52s
24x4: 110.90s
```

Tensor-level reconstruction summary:

| Run | Mean tensor rel RMSE | Worst tensor rel RMSE | Mean tensor cosine |
|---|---:|---:|---:|
| first FIT | 0.36596 | 0.42471 | 0.93033 |
| 6x2 epochs | 0.33654 | 0.39187 | 0.94141 |
| 24x4 epochs | 0.33495 | 0.38997 | 0.94199 |

The 24x4 run only slightly improved reconstruction error compared with 6x2, but
conversion time increased by about 4.9x. Output remained incoherent:

```text
`. A
.主义
.在
.主义
.

.
```

This shows diminishing returns from more epochs. Further epoch increases are
unlikely to fix the format by themselves. The next meaningful accuracy step is
probably a larger residual budget, such as `Q_SWQ_FIT_4`, not just more epochs.

Updated graph report:

```text
models/swq/swq-fit-2-epochs24-layer-report.html
```

### Triple epoch run: 72 fit epochs x 4 residual iterations

The fit epochs were tripled again:

```text
72 fit epochs x 4 residual iterations
```

This run confirmed convergence. Accuracy was effectively unchanged from 24x4,
while conversion time increased heavily.

| Run | Conversion time | Mean tensor rel RMSE | Worst tensor rel RMSE | Mean tensor cosine |
|---|---:|---:|---:|---:|
| first FIT | 5.47s | 0.365959 | 0.424710 | 0.930333 |
| 6x2 epochs | 22.52s | 0.336538 | 0.391867 | 0.941415 |
| 24x4 epochs | 110.90s | 0.334946 | 0.389966 | 0.941987 |
| 72x4 epochs | 484.64s | 0.334946 | 0.389966 | 0.941987 |

Full `Q_SWQ_FIT_2` 72x4 smoke result:

| Model | File bytes | Max RSS | Prompt speed | Generation speed | Output |
|---|---:|---:|---:|---:|---|
| Full Q_SWQ_FIT_2, 72x4 | 336,112,480 | 547,209,216 | 69.3 t/s | 1.4 t/s | incoherent |

Output remained bad:

```text
`.在
.主义
.
```

Conclusion from the 72x4 run: more epochs no longer improve the reconstruction.
The 2-bit residual budget is the limiting factor.

Updated graph report:

```text
models/swq/swq-fit-2-epochs72-layer-report.html
```

### Higher epoch check: 144 fit epochs x 4 residual iterations

One more increased-epoch run was tested after making epochs configurable:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 144 \
    --swq-fit-residual-epochs 4 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-fit-2-epochs144.gguf \
    Q_SWQ_FIT_2
```

Conversion completed, but `/usr/bin/time -l` returned code 1 after conversion
because `sysctl kern.clockrate` was blocked in the sandbox. The GGUF file was
still written successfully.

| Run | Conversion time | Mean tensor rel RMSE | Worst tensor rel RMSE | Mean tensor cosine |
|---|---:|---:|---:|---:|
| first FIT | 5.47s | 0.365959 | 0.424710 | 0.930333 |
| 6x2 epochs | 22.52s | 0.336538 | 0.391867 | 0.941415 |
| 24x4 epochs | 110.90s | 0.334946 | 0.389966 | 0.941987 |
| 72x4 epochs | 484.64s | 0.334946 | 0.389966 | 0.941987 |
| 144x4 epochs | 862.09s | 0.334946 | 0.389966 | 0.941987 |

Full `Q_SWQ_FIT_2` 144x4 conversion result:

| Model | File bytes | Quant size | SWQ original tensor bytes | SWQ tensor bytes | SWQ ratio | SWQ saved |
|---|---:|---:|---:|---:|---:|---:|
| Full Q_SWQ_FIT_2, 144x4 | 336,112,480 | 314.87 MiB | 1,260,477,952 | 330,164,736 | 3.82x | 73.81% |

The 144x4 accuracy is bit-for-bit equivalent to 72x4 at the summary level. That
means more epochs are not useful for this `Q_SWQ_FIT_2` design. The next useful
accuracy experiment is a larger residual format, such as a 4-bit residual
variant, not more epochs.

Updated graph report:

```text
models/swq/swq-fit-2-epochs144-layer-report.html
```

### Lossless predictor experiment

A separate offline analyzer was added to test exact reconstruction before adding
another runtime format:

```text
tools/swq-lossless-analyze.py
```

This tests the predictive lossless idea:

```text
original FP16 bits = predicted FP16 bits XOR exact residual bits
```

The predictor is trained per block, then the exact residual stream is compressed
with zlib. This guarantees exact reconstruction of the original FP16 bits. The
test answers only one question: does equation-plus-exact-residual compression
save enough space to justify a runtime format?

Focused K/V-only test:

```sh
python3 tools/swq-lossless-analyze.py \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    --include 'blk\.[0-9]+\.attn_[kv]\.weight' \
    --block-sizes 32,128,512 \
    --predictors mean,cubic,delta \
    --zlib-level 1 \
    --csv models/swq/swq-lossless-analysis-kv.csv \
    --html models/swq/swq-lossless-analysis-kv.html
```

K/V exact result:

| Tensor set | Original bytes | Best exact bytes | Compression ratio | Saved |
|---|---:|---:|---:|---:|
| K/V weights | 11,010,048 | 10,521,011 | 1.046x | 4.44% |

Broader transformer-weight test:

```sh
python3 tools/swq-lossless-analyze.py \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    --include 'blk\.[0-9]+\..*\.weight' \
    --block-sizes 128,512 \
    --predictors cubic,delta \
    --zlib-level 1 \
    --csv models/swq/swq-lossless-analysis-blocks.csv \
    --html models/swq/swq-lossless-analysis-blocks.html
```

Transformer exact result:

| Tensor set | Original bytes | Best exact bytes | Compression ratio | Saved |
|---|---:|---:|---:|---:|
| Transformer weights | 715,739,136 | 676,472,697 | 1.058x | 5.49% |

The best predictor was usually delta with block size 512. This is important:
exact predictive compression works, but savings are too small for the current
goal. A runtime format would also need on-the-fly decompression, likely making
inference slower. This path should not be implemented in llama.cpp unless a much
better residual compressor or predictor is found.

Saved reports:

```text
models/swq/swq-lossless-analysis-kv.html
models/swq/swq-lossless-analysis-blocks.html
```

### FIT_2 / FIT_3 / FIT_4 correction sweep

After the lossless test showed too little compression, a lossy correction-width
sweep was added:

```text
tools/swq-fit-sweep.py
```

This tests the same high-compression family as `Q_SWQ_FIT_2`, but estimates
larger correction codebooks before adding new runtime tensor types. The sweep
uses:

```text
128 weights per block
cubic predictor per block
uniform residual codebook
2-bit, 3-bit, or 4-bit correction indices
```

The broader transformer-weight sweep used:

```sh
python3 tools/swq-fit-sweep.py \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    --include 'blk\.[0-9]+\..*\.weight' \
    --bits 2,3,4 \
    --block-size 128 \
    --csv models/swq/swq-fit-sweep-blocks.csv \
    --html models/swq/swq-fit-sweep-blocks.html
```

Transformer-weight sweep result:

| Format | Original bytes | Estimated SWQ bytes | Compression ratio | Saved | Rel RMSE |
|---|---:|---:|---:|---:|---:|
| FIT_2 | 715,739,136 | 134,201,088 | 5.333x | 81.25% | 0.398676 |
| FIT_3 | 715,739,136 | 201,301,632 | 3.556x | 71.88% | 0.169957 |
| FIT_4 | 715,739,136 | 290,769,024 | 2.462x | 59.38% | 0.078962 |

This shows the useful tradeoff clearly:

```text
FIT_2: close to current high compression, but too inaccurate
FIT_3: still near 70% savings, much lower error
FIT_4: best accuracy, but compression drops to about 59%
```

For the user's target of roughly 70% compression, `FIT_3` is the most relevant
next runtime experiment. `FIT_4` is the safer accuracy experiment if `FIT_3`
still produces incoherent output.

Saved reports:

```text
models/swq/swq-fit-sweep-blocks.html
models/swq/swq-fit-sweep-blocks.csv
models/swq/swq-fit-sweep-blocks.log
```

A layer-by-layer visual comparison report was also generated:

```text
models/swq/swq-fit-comparison-report.html
models/swq/swq-fit-comparison-report.csv
models/swq/swq-fit-comparison-report.log
```

The graph report overlays original FP16 layer samples with the values predicted
by FIT_2, FIT_3, and FIT_4. Metrics are computed over all selected transformer
weight values; the plotted lines are downsampled so the browser remains usable.

### Runtime Q_SWQ_FIT_3 implementation

`Q_SWQ_FIT_3` was then implemented in llama.cpp as a real GGUF tensor type:

```text
GGML_TYPE_Q_SWQ_FIT_3
LLAMA_FTYPE_MOSTLY_Q_SWQ_FIT_3
```

Block layout:

```text
128 weights per block
4 FP16 cubic coefficients
8 FP16 residual values
128 packed 3-bit residual indices
72 bytes per block
4.5 bpw
```

The implementation is CPU-only and experimental. It reuses the same conversion
controls as `Q_SWQ_FIT_2`:

```text
--swq-fit-epochs
--swq-fit-residual-epochs
--swq-fit-progress
```

Full `Q_SWQ_FIT_3` conversion, 72 fit epochs x 4 residual iterations:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 72 \
    --swq-fit-residual-epochs 4 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-fit-3-epochs72.gguf \
    Q_SWQ_FIT_3
```

72x4 conversion result:

| Model | File size | Quant size | SWQ tensor bytes | SWQ ratio | SWQ saved | Conversion time |
|---|---:|---:|---:|---:|---:|---:|
| Full Q_SWQ_FIT_3, 72x4 | 409 MB | 403.20 MiB | 422,782,464 | 2.98x | 66.46% | 388.79s |

Accuracy compared with `Q_SWQ_FIT_2`:

| Run | Mean tensor rel RMSE | Worst tensor rel RMSE | Mean tensor cosine |
|---|---:|---:|---:|
| FIT_2 72x4 | 0.334946 | 0.389966 | 0.941987 |
| FIT_3 72x4 | 0.177571 | 0.219497 | 0.984012 |

Smoke result:

| Model | Prompt speed | Generation speed | Output |
|---|---:|---:|---|
| Full Q_SWQ_FIT_3, 72x4 | 62.9 t/s | 2.7 t/s | `The capital of India is Delhi.` |

This is a meaningful quality improvement over `Q_SWQ_FIT_2`: the output became
coherent on the tiny prompt and reconstruction error dropped heavily. It is
still too slow for the user's 10 t/s target with the current scalar CPU dot
kernel.

Full `Q_SWQ_FIT_3` conversion, 144 fit epochs x 4 residual iterations:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 144 \
    --swq-fit-residual-epochs 4 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-fit-3-epochs144.gguf \
    Q_SWQ_FIT_3
```

144x4 result:

| Run | Conversion time | Mean tensor rel RMSE | Worst tensor rel RMSE | Mean tensor cosine |
|---|---:|---:|---:|---:|
| FIT_3 72x4 | 388.79s | 0.177571 | 0.219497 | 0.984012 |
| FIT_3 144x4 | 1494.58s | 0.177571 | 0.219497 | 0.984012 |

Conclusion: `Q_SWQ_FIT_3` converges by 72x4 for this model. More epochs did not
improve reconstruction, and 144x4 was not worth the extra conversion time.

Saved files:

```text
models/swq/qwen2.5-0.5b-instruct-q-swq-fit-3-epochs72.gguf
models/swq/conversion-swq-fit-3-epochs72.log
models/swq/smoke-swq-fit-3-epochs72.log
models/swq/accuracy-swq-fit-3-epochs72.txt
models/swq/accuracy-swq-fit-3-epochs72.csv
models/swq/swq-fit-3-epochs72-layer-report.html
models/swq/qwen2.5-0.5b-instruct-q-swq-fit-3-epochs144.gguf
models/swq/conversion-swq-fit-3-epochs144.log
models/swq/accuracy-swq-fit-3-epochs144.txt
models/swq/accuracy-swq-fit-3-epochs144.csv
models/swq/swq-fit-3-epochs144-layer-report.html
```

### Configurable FIT epochs and progress logging

FIT epochs are now configurable from `llama-quantize`, so conversion experiments
do not require rebuilding:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 24 \
    --swq-fit-residual-epochs 4 \
    --swq-fit-progress \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-fit-2.gguf \
    Q_SWQ_FIT_2
```

New flags:

```text
--swq-fit-epochs N
--swq-fit-residual-epochs N
--swq-fit-progress
```

`--swq-fit-progress` prints per-tensor, per-epoch reconstruction stats. Example
from a short 2x1 K/V-only verification run:

```text
SWQ_FIT_2 epochs: tensor=blk.0.attn_k.weight blocks=896 fit_epochs=2 residual_epochs=1
  epoch   1/  2 - rmse 0.029877587 - rel_rmse 0.47250399
  epoch   2/  2 - rmse 0.027121888 - rel_rmse 0.42892354
```

Verification command:

```sh
./build/bin/llama-quantize --swq-stats \
    --swq-fit-epochs 2 \
    --swq-fit-residual-epochs 1 \
    --swq-fit-progress \
    --tensor-type attn_k=q_swq_fit_2 \
    --tensor-type attn_v=q_swq_fit_2 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q8-swq-fit-kv-e2.gguf \
    Q8_0
```

The progress verification completed successfully and wrote:

```text
models/swq/conversion-q8-swq-fit-kv-e2-progress.log
```

## Current conclusions

## Hierarchical INT8 FIT3-256 and exact-anchor experiment

A separate offline design tested 256-weight groups containing two local
128-weight cubic predictors, one shared eight-value residual codebook, INT8
coefficients, and packed 3-bit residual indices. This preserves the local
predictor span while sharing correction metadata.

The 12x2 tensor-scale experiment produced:

| Anchors per 256 weights | Estimated saved | Relative RMSE | Cosine |
|---:|---:|---:|---:|
| 0 | 76.56% | 0.134605 | 0.990899 |
| 1 | 75.98% | 0.132332 | 0.991207 |
| 2 | 75.39% | 0.130659 | 0.991432 |
| 3 | 74.80% | 0.129236 | 0.991620 |

Two anchors represent 0.78% of each block. Each anchor stores a one-byte
position and exact FP16 value. Compared with the 128-weight offline FIT3 result
of 0.169957 relative RMSE, the two-anchor result reduces reconstruction error
while retaining an estimated rate just below 4 bpw.

A self-contained per-block coefficient scale was then tested because normal
GGML dot kernels cannot access tensor-level side scales. At 6x2:

| Layout | Estimated saved | Relative RMSE | Cosine |
|---|---:|---:|---:|
| Per-block scale, no anchors | 76.17% | 0.137083 | 0.990560 |
| Per-block scale, 2 anchors | 75.00% | 0.133266 | 0.991086 |

The per-block, two-anchor layout is exactly 128 bytes per 256 weights, or 4.0
bpw. The minimal `llama-quantize` and `llama-cli` build passed after adding an
experimental runtime type.

The runtime compatibility check exposed a blocking GGML row-layout constraint:

```text
Qwen matrix tensors:              170
row width divisible by 256:        24
row width not divisible by 256:   146
dominant incompatible row width:  896
```

GGML quantized blocks cannot cross tensor rows. Therefore a standard 256-weight
block would apply to only 24 matrices and force most Qwen matrices to another
format. No full-model conversion was run because it would not represent the
offline compression result. The next runtime design must either use a block
size dividing both 896 and 4864, whose greatest common divisor is 128, or add a
more invasive row-aware storage mechanism.

Saved reports:

```text
models/swq/swq-hfit3-256-e6.html
models/swq/swq-hfit3-256-e12.html
models/swq/swq-hfit3-256-blockscale-e6.html
models/swq/swq-hfit3-256-e6.csv
models/swq/swq-hfit3-256-e12.csv
models/swq/swq-hfit3-256-blockscale-e6.csv
```

The prototype works end to end on CPU and provides measurable file and RAM
savings relative to FP16. Full SWQ128 gives strong compression, but scalar
codebook lookup is too slow for generation on this CPU. Full `Q_SWQ_FIT_2`
compresses even more, but its approximation error is too high and output becomes
incoherent. A mixed Q8_0 + SWQ128 K/V model is the current best local tradeoff:
it stays above 10 generation tokens/sec and saves some RAM versus Q8_0. The next
useful experiment is an ARM NEON SWQ-by-Q8 activation dot kernel; without that,
broad SWQ use is unlikely to be fast enough.

Perplexity, KL divergence, and direct Q4_K_M/Q8_0 benchmark runs have not yet
been measured.

## Runtime HFIT3-128 experiment

`Q_SWQ_HFIT_3_128` was added as a row-compatible hierarchical FIT variant after
the 256-weight design was blocked by Qwen row widths. The physical block is 128
weights:

- two 64-weight cubic predictors
- one INT8 scale for the eight coefficients
- eight INT8 cubic coefficients
- eight FP16 residual codebook entries
- two exact FP16 anchor values
- 128 packed 3-bit residual indices

The block size is 80 bytes, or 5.0 raw bits per weight.

Build:

```sh
cmake --build build --target llama-quantize llama-cli llama-completion -j4
```

Conversion command:

```sh
./build/bin/llama-quantize \
    --swq-stats \
    --swq-fit-epochs 6 \
    --swq-fit-residual-epochs 2 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-hfit-3-128-e6.gguf \
    Q_SWQ_HFIT_3_128
```

Conversion result:

| Metric | Value |
|---|---:|
| Model size before | 1202.09 MiB |
| Quant size after | 432.64 MiB |
| Quant BPW | 5.76 |
| SWQ original tensor bytes | 1,260,477,952 |
| SWQ tensor bytes | 453,655,040 |
| SWQ compression ratio | 2.78x |
| SWQ percentage saved | 64.01% |
| Conversion time | 50.29 s |

`output.weight` fell back to Q8_0, so this is not a fully HFIT3-128 file.
Converted HFIT3-128 tensors individually reported 3.20x compression and 68.75%
savings versus FP16 tensor bytes.

Smoke command:

```sh
/usr/bin/time -l ./build/bin/llama-completion \
    -m models/swq/qwen2.5-0.5b-instruct-q-swq-hfit-3-128-e6.gguf \
    -p "The capital of India is" \
    -n 16 \
    -t 4 \
    --temp 0 \
    --no-display-prompt
```

Smoke result:

| Model | Output | Max RSS | Prompt eval | Generation |
|---|---|---:|---:|---:|
| Q8_0 | `New Delhi` | 1,184,448,512 bytes | 70.46 t/s | 17.31 t/s |
| HFIT3-128 e6 | `The capital of India is New Delhi.` | 676,708,352 bytes | 0.97 t/s | 0.87 t/s |

Measured max RSS saving versus Q8_0 on this prompt was 507,740,160 bytes, about
484 MiB. The quality on this one smoke prompt was correct. Runtime speed is not
usable yet because the HFIT3-128 CPU dot path is scalar and reconstructs each
weight through polynomial and residual lookup work.

### Anchor correction moved out of the inner loop

The first HFIT3-128 CPU dot loop checked both exact anchor positions for every
weight:

```text
if current position is anchor 0, replace predicted weight
if current position is anchor 1, replace predicted weight
```

That was replaced with a post-loop correction:

```text
dot += (exact_anchor - predicted_anchor) * activation_at_anchor
```

This keeps the same mathematical result while removing two branches per weight.

Retest on the same HFIT3-128 e6 GGUF:

| Variant | Output | Max RSS | Prompt eval | Generation | Instructions retired |
|---|---|---:|---:|---:|---:|
| Before anchor change | `The capital of India is New Delhi.` | 676,708,352 bytes | 0.97 t/s | 0.87 t/s | 410,776,567,430 |
| After anchor correction | `The capital of India is New Delhi.` | 641,204,224 bytes | 1.00 t/s | 0.85 t/s | 385,398,740,793 |

The branch removal reduced retired instructions by about 6%, but it did not
materially improve tokens/sec. The remaining bottleneck is the scalar per-weight
math: 3-bit unpack, cubic evaluation, residual lookup, and Q8 activation multiply.

## Runtime HFIT4-128 experiment

`Q_SWQ_HFIT_4_128` was added to test whether simpler 4-bit residual indices are
faster than packed 3-bit indices. This format keeps the same row-compatible
128-weight physical block and two 64-weight cubic predictors, but changes the
residual path:

- 16 FP16 residual codebook entries
- 4-bit residual indices, packed two per byte
- same two exact anchors as HFIT3-128
- ARM NEON accumulation path for `Q_SWQ_HFIT_4_128 x Q8_0`

The block size is 112 bytes, or 7.0 raw bits per weight. This is intentionally
larger than HFIT3-128; the point of the test was speed and reconstruction quality,
not maximum compression.

Conversion command:

```sh
./build/bin/llama-quantize \
    --swq-stats \
    --swq-fit-epochs 6 \
    --swq-fit-residual-epochs 2 \
    models/swq/qwen2.5-0.5b-instruct-fp16.gguf \
    models/swq/qwen2.5-0.5b-instruct-q-swq-hfit-4-128-e6.gguf \
    Q_SWQ_HFIT_4_128
```

Conversion result:

| Metric | Value |
|---|---:|
| Model size before | 1202.09 MiB |
| Quant size after | 550.41 MiB |
| Quant BPW | 7.33 |
| SWQ original tensor bytes | 1,260,477,952 |
| SWQ tensor bytes | 577,145,344 |
| SWQ compression ratio | 2.18x |
| SWQ percentage saved | 54.21% |
| Conversion time | 50.13 s |

Smoke result:

| Model | File size | Output | Max RSS | Prompt eval | Generation | Instructions retired |
|---|---:|---|---:|---:|---:|---:|
| Q8_0 | 644 MB | `New Delhi` | 1,184,448,512 bytes | 70.46 t/s | 17.31 t/s | 14,703,410,751 |
| HFIT3-128 e6 | 438 MB | `The capital of India is New Delhi.` | 641,204,224 bytes | 1.00 t/s | 0.85 t/s | 385,398,740,793 |
| HFIT4-128 e6 | 556 MB | `New Delhi` | 775,307,264 bytes | 1.26 t/s | 1.08 t/s | 135,556,027,146 |

HFIT4-128 improved speed and retired instructions versus HFIT3-128, but it is
still far slower than Q8_0. It also gives up much of the RAM gain: the file is
118 MB larger than HFIT3-128 and max RSS is about 128 MiB higher. The experiment
shows that 4-bit indices help, but the dominant runtime cost is still computing
equation-derived weights during the dot product.

## Raw logs

- `models/swq/conversion.log`
- `models/swq/smoke-swq.log`
- `models/swq/smoke-fp16.log`
- `models/swq/smoke-swq-q8kernel.log`
- `models/swq/smoke-fp16-q8kernel-compare.log`
- `models/swq/conversion-swq128.log`
- `models/swq/smoke-swq128.log`
- `models/swq/conversion-q8-0.log`
- `models/swq/smoke-q8-0.log`
- `models/swq/conversion-q8-swq-kv-128.log`
- `models/swq/smoke-q8-swq-kv-128.log`
- `models/swq/conversion-q8-swq-attn-128.log`
- `models/swq/smoke-q8-swq-attn-128.log`
- `models/swq/conversion-swq-fit-2.log`
- `models/swq/smoke-swq-fit-2.log`
- `models/swq/accuracy-swq-fit-2.txt`
- `models/swq/accuracy-swq-fit-2.csv`
- `models/swq/conversion-q8-swq-fit-kv.log`
- `models/swq/smoke-q8-swq-fit-kv.log`
- `models/swq/conversion-swq-fit-2-epochs.log`
- `models/swq/smoke-swq-fit-2-epochs.log`
- `models/swq/accuracy-swq-fit-2-epochs.txt`
- `models/swq/accuracy-swq-fit-2-epochs.csv`
- `models/swq/swq-fit-2-epochs-layer-report.html`
- `models/swq/conversion-swq-fit-2-epochs24.log`
- `models/swq/smoke-swq-fit-2-epochs24.log`
- `models/swq/accuracy-swq-fit-2-epochs24.txt`
- `models/swq/accuracy-swq-fit-2-epochs24.csv`
- `models/swq/swq-fit-2-epochs24-layer-report.html`
- `models/swq/conversion-swq-fit-2-epochs72.log`
- `models/swq/smoke-swq-fit-2-epochs72.log`
- `models/swq/accuracy-swq-fit-2-epochs72.txt`
- `models/swq/accuracy-swq-fit-2-epochs72.csv`
- `models/swq/swq-fit-2-epochs72-layer-report.html`
- `models/swq/conversion-swq-fit-2-epochs144.log`
- `models/swq/accuracy-swq-fit-2-epochs144.csv`
- `models/swq/swq-fit-2-epochs144-layer-report.html`
- `models/swq/swq-lossless-analysis-kv.log`
- `models/swq/swq-lossless-analysis-kv.csv`
- `models/swq/swq-lossless-analysis-kv.html`
- `models/swq/swq-lossless-analysis-blocks.log`
- `models/swq/swq-lossless-analysis-blocks.csv`
- `models/swq/swq-lossless-analysis-blocks.html`
- `models/swq/swq-fit-sweep-smoke.csv`
- `models/swq/swq-fit-sweep-smoke.html`
- `models/swq/swq-fit-sweep-blocks.log`
- `models/swq/swq-fit-sweep-blocks.csv`
- `models/swq/swq-fit-sweep-blocks.html`
- `models/swq/swq-fit-comparison-report.log`
- `models/swq/swq-fit-comparison-report.csv`
- `models/swq/swq-fit-comparison-report.html`
- `models/swq/conversion-swq-fit-3-epochs72.log`
- `models/swq/smoke-swq-fit-3-epochs72.log`
- `models/swq/accuracy-swq-fit-3-epochs72.txt`
- `models/swq/accuracy-swq-fit-3-epochs72.csv`
- `models/swq/swq-fit-3-epochs72-layer-report.html`
- `models/swq/conversion-swq-fit-3-epochs144.log`
- `models/swq/accuracy-swq-fit-3-epochs144.txt`
- `models/swq/accuracy-swq-fit-3-epochs144.csv`
- `models/swq/swq-fit-3-epochs144-layer-report.html`
- `models/swq/conversion-q8-swq-fit-kv-e2-progress.log`
