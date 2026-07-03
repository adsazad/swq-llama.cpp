#!/usr/bin/env python3
import argparse
import csv
import html
import re
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / "gguf-py" / "gguf"
pkg = types.ModuleType("gguf")
pkg.__path__ = [str(GGUF_PY)]
sys.modules.setdefault("gguf", pkg)

from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.gguf_reader import GGUFReader  # noqa: E402


def tensor_to_float(tensor):
    if tensor.tensor_type == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    return None


def layer_name(tensor_name):
    m = re.match(r"blk\.(\d+)\.", tensor_name)
    if m:
        return f"blk.{m.group(1)}"
    return "non_block"


def ceil_div(a, b):
    return (a + b - 1) // b


def estimated_block_bytes(block_size, bits):
    coeff_bytes = 4 * 2
    codebook_bytes = (1 << bits) * 2
    index_bytes = ceil_div(block_size * bits, 8)
    return coeff_bytes + codebook_bytes + index_bytes


def quantize_residual_uniform(residual, bits):
    levels = 1 << bits
    rmin = np.min(residual, axis=1, keepdims=True)
    rmax = np.max(residual, axis=1, keepdims=True)
    scale = (rmax - rmin) / max(levels - 1, 1)
    scale = np.where(scale == 0.0, 1.0, scale)
    idx = np.rint((residual - rmin) / scale)
    idx = np.clip(idx, 0, levels - 1)
    return rmin + idx * scale


def analyze_tensor(values, block_size, bits, chunk_blocks):
    n = values.size
    n_blocks = ceil_div(n, block_size)

    x = np.linspace(-1.0, 1.0, block_size, dtype=np.float32)
    powers = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=1)
    pinv = np.linalg.pinv(powers).astype(np.float32)

    sse = 0.0
    sae = 0.0
    orig2 = 0.0
    approx2 = 0.0
    dot = 0.0

    for start in range(0, n_blocks, chunk_blocks):
        count = min(chunk_blocks, n_blocks - start)
        offset = start * block_size
        take = min(count * block_size, n - offset)
        padded = np.zeros(count * block_size, dtype=np.float32)
        padded[:take] = values[offset:offset + take]
        blocks = padded.reshape(count, block_size)

        coeffs = blocks @ pinv.T
        base = coeffs @ powers.T
        corrected = base + quantize_residual_uniform(blocks - base, bits)

        orig = blocks.reshape(-1)[:take]
        approx = corrected.reshape(-1)[:take]
        diff = approx - orig

        sse += float(np.sum(diff * diff))
        sae += float(np.sum(np.abs(diff)))
        orig2 += float(np.sum(orig * orig))
        approx2 += float(np.sum(approx * approx))
        dot += float(np.dot(orig, approx))

    rmse = float(np.sqrt(sse / max(n, 1)))
    mae = sae / max(n, 1)
    rel_rmse = rmse / (float(np.sqrt(orig2 / max(n, 1))) + 1e-12)
    cosine = dot / (float(np.sqrt(orig2) * np.sqrt(approx2)) + 1e-12)

    original_bytes = n * 2
    swq_bytes = n_blocks * estimated_block_bytes(block_size, bits)
    ratio = original_bytes / max(swq_bytes, 1)
    saved_pct = (1.0 - swq_bytes / max(original_bytes, 1)) * 100.0

    return {
        "n": int(n),
        "blocks": int(n_blocks),
        "sse": float(sse),
        "sae": float(sae),
        "orig2": float(orig2),
        "approx2": float(approx2),
        "dot": float(dot),
        "original_bytes": int(original_bytes),
        "swq_bytes": int(swq_bytes),
        "ratio": float(ratio),
        "saved_pct": float(saved_pct),
        "rmse": rmse,
        "mae": mae,
        "rel_rmse": float(rel_rmse),
        "cosine": float(cosine),
    }


def parse_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_html(path, rows, summary):
    summary_rows = "".join(
        f"<tr><td>FIT_{bits}</td><td>{s['original_bytes']:,}</td><td>{s['swq_bytes']:,}</td>"
        f"<td>{s['ratio']:.3f}x</td><td>{s['saved_pct']:.2f}%</td>"
        f"<td>{s['rel_rmse']:.6f}</td><td>{s['cosine']:.6f}</td></tr>"
        for bits, s in sorted(summary.items())
    )
    body_rows = "".join(
        f"<tr><td>{html.escape(r['tensor'])}</td><td>{r['fit']}</td><td>{r['block_size']}</td>"
        f"<td>{r['swq_bytes']:,}</td><td>{r['ratio']:.3f}x</td><td>{r['saved_pct']:.2f}%</td>"
        f"<td>{r['rel_rmse']:.6f}</td><td>{r['cosine']:.6f}</td></tr>"
        for r in rows
    )
    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SWQ FIT correction sweep</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 28px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>SWQ FIT correction sweep</h1>
<p>Offline estimate using cubic prediction plus uniform residual codebooks. This is for choosing a format before runtime implementation.</p>
<h2>Summary</h2>
<table><thead><tr><th>Format</th><th>Original bytes</th><th>SWQ bytes</th><th>Ratio</th><th>Saved</th><th>Rel RMSE</th><th>Cosine</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Per tensor</h2>
<table><thead><tr><th>Tensor</th><th>Format</th><th>Block</th><th>SWQ bytes</th><th>Ratio</th><th>Saved</th><th>Rel RMSE</th><th>Cosine</th></tr></thead><tbody>{body_rows}</tbody></table>
</body>
</html>
"""
    Path(path).write_text(doc)


def main():
    parser = argparse.ArgumentParser(description="Sweep SWQ FIT residual bit widths before adding runtime formats.")
    parser.add_argument("model")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--include", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--limit-tensors", type=int, default=0)
    parser.add_argument("--chunk-blocks", type=int, default=4096)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--html", default=None)
    args = parser.parse_args()

    bit_widths = parse_ints(args.bits)
    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    rows = []
    summary = {
        bits: {"n": 0, "original_bytes": 0, "swq_bytes": 0, "sse": 0.0, "sae": 0.0, "orig2": 0.0, "approx2": 0.0, "dot": 0.0}
        for bits in bit_widths
    }

    checked = 0
    for tensor in GGUFReader(args.model).tensors:
        if include_re and not include_re.search(tensor.name):
            continue
        if exclude_re and exclude_re.search(tensor.name):
            continue
        values = tensor_to_float(tensor)
        if values is None or values.size == 0:
            continue

        checked += 1
        for bits in bit_widths:
            stats = analyze_tensor(values, args.block_size, bits, args.chunk_blocks)
            row = {
                "tensor": tensor.name,
                "layer": layer_name(tensor.name),
                "source_type": tensor.tensor_type.name,
                "fit": f"FIT_{bits}",
                "bits": bits,
                "block_size": args.block_size,
                **stats,
            }
            rows.append(row)

            st = summary[bits]
            st["n"] += stats["n"]
            st["original_bytes"] += stats["original_bytes"]
            st["swq_bytes"] += stats["swq_bytes"]
            st["sse"] += stats["sse"]
            st["sae"] += stats["sae"]
            st["orig2"] += stats["orig2"]
            st["approx2"] += stats["approx2"]
            st["dot"] += stats["dot"]

        best = min((r for r in rows[-len(bit_widths):]), key=lambda x: x["rel_rmse"])
        print(f"{checked:4d} {tensor.name}: best={best['fit']} rel_rmse={best['rel_rmse']:.6f} saved={best['saved_pct']:.2f}%", flush=True)
        if args.limit_tensors and checked >= args.limit_tensors:
            break

    final_summary = {}
    for bits, st in summary.items():
        n = max(st["n"], 1)
        rmse = float(np.sqrt(st["sse"] / n))
        mae = st["sae"] / n
        rel_rmse = rmse / (float(np.sqrt(st["orig2"] / n)) + 1e-12)
        ratio = st["original_bytes"] / max(st["swq_bytes"], 1)
        saved_pct = (1.0 - st["swq_bytes"] / max(st["original_bytes"], 1)) * 100.0
        cosine = st["dot"] / (float(np.sqrt(st["orig2"]) * np.sqrt(st["approx2"])) + 1e-12)
        final_summary[bits] = {
            "original_bytes": int(st["original_bytes"]),
            "swq_bytes": int(st["swq_bytes"]),
            "ratio": float(ratio),
            "saved_pct": float(saved_pct),
            "rmse": rmse,
            "mae": float(mae),
            "rel_rmse": float(rel_rmse),
            "cosine": float(cosine),
        }

    print()
    print("SWQ FIT sweep summary")
    print("fit,original_bytes,swq_bytes,ratio,saved_pct,rel_rmse")
    for bits, st in sorted(final_summary.items()):
        print(f"FIT_{bits},{st['original_bytes']},{st['swq_bytes']},{st['ratio']:.6f},{st['saved_pct']:.6f},{st['rel_rmse']:.6f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = [
                "tensor", "layer", "source_type", "fit", "bits", "block_size", "n", "blocks",
                "original_bytes", "swq_bytes", "ratio", "saved_pct", "rmse", "mae", "rel_rmse", "cosine",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: r[k] for k in fieldnames} for r in rows])

    if args.html:
        write_html(args.html, rows, final_summary)
        print(args.html)


if __name__ == "__main__":
    main()
