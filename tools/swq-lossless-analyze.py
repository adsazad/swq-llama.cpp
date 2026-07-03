#!/usr/bin/env python3
import argparse
import csv
import html
import re
import sys
import types
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / "gguf-py" / "gguf"
pkg = types.ModuleType("gguf")
pkg.__path__ = [str(GGUF_PY)]
sys.modules.setdefault("gguf", pkg)

from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.gguf_reader import GGUFReader  # noqa: E402


def tensor_to_f16_bits(tensor):
    if tensor.tensor_type == GGMLQuantizationType.F16:
        f16 = np.asarray(tensor.data, dtype=np.float16).reshape(-1)
    elif tensor.tensor_type == GGMLQuantizationType.F32:
        f16 = np.asarray(tensor.data, dtype=np.float32).astype(np.float16).reshape(-1)
    else:
        return None
    return f16.view(np.uint16)


def layer_name(tensor_name):
    m = re.match(r"blk\.(\d+)\.", tensor_name)
    if m:
        return f"blk.{m.group(1)}"
    return "non_block"


def fit_poly_blocks(values, block_size, degree):
    n_blocks = (values.size + block_size - 1) // block_size
    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[:values.size] = values
    blocks = padded.reshape(n_blocks, block_size)

    x = np.linspace(-1.0, 1.0, block_size, dtype=np.float32)
    powers = np.stack([x ** i for i in range(degree + 1)], axis=1)
    pinv = np.linalg.pinv(powers).astype(np.float32)
    coeffs = blocks @ pinv.T
    pred = (coeffs @ powers.T).reshape(-1)[:values.size]
    return pred.astype(np.float16), coeffs.astype(np.float16)


def fit_delta_blocks(values, block_size):
    n_blocks = (values.size + block_size - 1) // block_size
    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[:values.size] = values
    blocks = padded.reshape(n_blocks, block_size)

    pred = np.zeros_like(blocks)
    pred[:, 0] = blocks[:, 0]
    pred[:, 1:] = blocks[:, :-1]
    coeffs = blocks[:, :1].astype(np.float16)
    return pred.reshape(-1)[:values.size].astype(np.float16), coeffs


def predictor_bytes(n_blocks, predictor, degree):
    if predictor == "delta":
        return n_blocks * 2
    return n_blocks * (degree + 1) * 2


def analyze_variant(bits, block_size, predictor, zlib_level):
    values = bits.view(np.float16).astype(np.float32)
    n_blocks = (values.size + block_size - 1) // block_size

    if predictor == "mean":
        pred, _ = fit_poly_blocks(values, block_size, 0)
        pbytes = predictor_bytes(n_blocks, predictor, 0)
    elif predictor == "linear":
        pred, _ = fit_poly_blocks(values, block_size, 1)
        pbytes = predictor_bytes(n_blocks, predictor, 1)
    elif predictor == "quadratic":
        pred, _ = fit_poly_blocks(values, block_size, 2)
        pbytes = predictor_bytes(n_blocks, predictor, 2)
    elif predictor == "cubic":
        pred, _ = fit_poly_blocks(values, block_size, 3)
        pbytes = predictor_bytes(n_blocks, predictor, 3)
    elif predictor == "delta":
        pred, _ = fit_delta_blocks(values, block_size)
        pbytes = predictor_bytes(n_blocks, predictor, 0)
    else:
        raise ValueError(f"unknown predictor: {predictor}")

    pred_bits = pred.view(np.uint16)
    residual = np.bitwise_xor(bits, pred_bits)
    residual_bytes = residual.astype("<u2", copy=False).tobytes()
    compressed = zlib.compress(residual_bytes, zlib_level)

    original_bytes = int(bits.size * 2)
    total_bytes = int(pbytes + len(compressed))
    return {
        "predictor_bytes": int(pbytes),
        "compressed_residual_bytes": int(len(compressed)),
        "total_bytes": total_bytes,
        "original_bytes": original_bytes,
        "ratio": original_bytes / max(total_bytes, 1),
        "saved_pct": (1.0 - total_bytes / max(original_bytes, 1)) * 100.0,
    }


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_words(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def write_html(path, rows, totals, original_path):
    best_rows = [r for r in rows if r["is_best"] == "yes"]
    best_rows.sort(key=lambda r: float(r["saved_pct"]))
    total_original = totals["original_bytes"]
    total_best = totals["best_bytes"]
    total_ratio = total_original / max(total_best, 1)
    total_saved = (1.0 - total_best / max(total_original, 1)) * 100.0

    max_orig = max((int(r["original_bytes"]) for r in best_rows), default=1)
    bars = []
    for r in best_rows[:80]:
        saved = float(r["saved_pct"])
        width = int(int(r["original_bytes"]) / max_orig * 480)
        color = "#2ca02c" if saved > 0 else "#d62728"
        bars.append(
            f"<tr><td>{html.escape(r['tensor'])}</td><td>{html.escape(r['layer'])}</td>"
            f"<td>{html.escape(r['predictor'])}</td><td>{r['block_size']}</td>"
            f"<td>{int(r['original_bytes']):,}</td><td>{int(r['total_bytes']):,}</td>"
            f"<td>{float(r['ratio']):.3f}x</td><td>{saved:.2f}%</td>"
            f"<td><div class='bar'><span style='width:{width}px;background:{color}'></span></div></td></tr>"
        )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SWQ lossless predictor analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f5f5f5; position: sticky; top: 0; }}
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-width: 180px; background: #fafafa; }}
.value {{ font-size: 22px; font-weight: 700; }}
.bar {{ width: 480px; height: 10px; background: #eee; }}
.bar span {{ display: block; height: 10px; }}
</style>
</head>
<body>
<h1>SWQ lossless predictor analysis</h1>
<p>Input: <code>{html.escape(str(original_path))}</code></p>
<p>This is exact recovery: original FP16 bits are reconstructed as predictor FP16 bits XOR exact residual bits.</p>
<div class="summary">
<div class="card"><div>Total original</div><div class="value">{total_original:,} bytes</div></div>
<div class="card"><div>Best exact size</div><div class="value">{total_best:,} bytes</div></div>
<div class="card"><div>Compression</div><div class="value">{total_ratio:.3f}x</div></div>
<div class="card"><div>Saved</div><div class="value">{total_saved:.2f}%</div></div>
</div>
<h2>Best exact predictor per tensor</h2>
<table>
<thead><tr><th>Tensor</th><th>Layer</th><th>Predictor</th><th>Block</th><th>Original</th><th>Exact size</th><th>Ratio</th><th>Saved</th><th>Scale</th></tr></thead>
<tbody>
{''.join(bars)}
</tbody>
</table>
</body>
</html>
"""
    Path(path).write_text(doc)


def main():
    parser = argparse.ArgumentParser(description="Analyze exact lossless predictive compression for GGUF FP16/F32 tensors.")
    parser.add_argument("model")
    parser.add_argument("--block-sizes", default="32,64,128,256,512,1024")
    parser.add_argument("--predictors", default="mean,linear,quadratic,cubic,delta")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--html", default=None)
    parser.add_argument("--limit-tensors", type=int, default=0)
    parser.add_argument("--include", default=None, help="Only analyze tensors matching this regex.")
    parser.add_argument("--exclude", default=None, help="Skip tensors matching this regex.")
    parser.add_argument("--zlib-level", type=int, default=6)
    args = parser.parse_args()

    block_sizes = parse_csv_ints(args.block_sizes)
    predictors = parse_csv_words(args.predictors)
    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    rows = []
    totals = {"original_bytes": 0, "best_bytes": 0}
    checked = 0

    for tensor in GGUFReader(args.model).tensors:
        if include_re and not include_re.search(tensor.name):
            continue
        if exclude_re and exclude_re.search(tensor.name):
            continue

        bits = tensor_to_f16_bits(tensor)
        if bits is None or bits.size == 0:
            continue

        variants = []
        for block_size in block_sizes:
            for predictor in predictors:
                try:
                    stats = analyze_variant(bits, block_size, predictor, args.zlib_level)
                except np.linalg.LinAlgError:
                    continue
                variants.append({
                    "tensor": tensor.name,
                    "layer": layer_name(tensor.name),
                    "source_type": tensor.tensor_type.name,
                    "n": int(bits.size),
                    "block_size": int(block_size),
                    "predictor": predictor,
                    **stats,
                    "is_best": "no",
                })

        if not variants:
            continue
        best = min(variants, key=lambda r: r["total_bytes"])
        best["is_best"] = "yes"
        rows.extend(variants)
        totals["original_bytes"] += best["original_bytes"]
        totals["best_bytes"] += best["total_bytes"]

        checked += 1
        print(
            f"{checked:4d} {tensor.name}: best={best['predictor']}/bs{best['block_size']} "
            f"ratio={best['ratio']:.3f}x saved={best['saved_pct']:.2f}%",
            flush=True,
        )
        if args.limit_tensors and checked >= args.limit_tensors:
            break

    best_ratio = totals["original_bytes"] / max(totals["best_bytes"], 1)
    best_saved = (1.0 - totals["best_bytes"] / max(totals["original_bytes"], 1)) * 100.0
    print()
    print("Lossless exact summary")
    print(f"original_bytes,{totals['original_bytes']}")
    print(f"best_exact_bytes,{totals['best_bytes']}")
    print(f"compression_ratio,{best_ratio:.6f}")
    print(f"saved_pct,{best_saved:.6f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = [
                "tensor", "layer", "source_type", "n", "block_size", "predictor",
                "original_bytes", "predictor_bytes", "compressed_residual_bytes",
                "total_bytes", "ratio", "saved_pct", "is_best",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if args.html:
        write_html(args.html, rows, totals, args.model)
        print(args.html)


if __name__ == "__main__":
    main()
