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

## Current conclusions

The prototype works end to end on CPU and provides measurable file and RAM
savings relative to FP16. Full SWQ128 gives strong compression, but scalar
codebook lookup is too slow for generation on this CPU. A mixed Q8_0 + SWQ128
K/V model is the current best local tradeoff: it stays above 10 generation
tokens/sec and saves some RAM versus Q8_0. The next useful experiment is an ARM
NEON SWQ-by-Q8 activation dot kernel; without that, broad SWQ use is unlikely to
be fast enough.

Perplexity, KL divergence, and direct Q4_K_M/Q8_0 benchmark runs have not yet
been measured.

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
