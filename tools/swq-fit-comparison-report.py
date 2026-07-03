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


def layer_sort_key(name):
    if name == "non_block":
        return -1
    if name.startswith("blk."):
        return int(name.split(".")[1])
    return 1000000


def ceil_div(a, b):
    return (a + b - 1) // b


def estimated_block_bytes(block_size, bits):
    return 8 + (1 << bits) * 2 + ceil_div(block_size * bits, 8)


def quantize_residual_uniform(residual, bits):
    levels = 1 << bits
    rmin = np.min(residual, axis=1, keepdims=True)
    rmax = np.max(residual, axis=1, keepdims=True)
    scale = (rmax - rmin) / max(levels - 1, 1)
    scale = np.where(scale == 0.0, 1.0, scale)
    idx = np.rint((residual - rmin) / scale)
    idx = np.clip(idx, 0, levels - 1)
    return rmin + idx * scale


def fit_predict(values, block_size, bits, chunk_blocks):
    n = values.size
    n_blocks = ceil_div(n, block_size)

    x = np.linspace(-1.0, 1.0, block_size, dtype=np.float32)
    powers = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=1)
    pinv = np.linalg.pinv(powers).astype(np.float32)

    pred = np.empty(n, dtype=np.float32)
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
        pred[offset:offset + take] = corrected.reshape(-1)[:take]
    return pred


def metrics(original, pred):
    diff = pred - original
    sse = float(np.sum(diff * diff))
    sae = float(np.sum(np.abs(diff)))
    orig2 = float(np.sum(original * original))
    pred2 = float(np.sum(pred * pred))
    dot = float(np.dot(original, pred))
    n = int(original.size)
    rmse = float(np.sqrt(sse / max(n, 1)))
    mae = sae / max(n, 1)
    rel_rmse = rmse / (float(np.sqrt(orig2 / max(n, 1))) + 1e-12)
    cosine = dot / (float(np.sqrt(orig2) * np.sqrt(pred2)) + 1e-12)
    return {
        "n": n,
        "sse": sse,
        "sae": sae,
        "orig2": orig2,
        "pred2": pred2,
        "dot": dot,
        "rmse": rmse,
        "mae": mae,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
    }


class LayerSampler:
    def __init__(self, sample_count):
        self.sample_count = sample_count
        self.original = []
        self.pred = {2: [], 3: [], 4: []}
        self.seen = 0

    def add(self, original, preds):
        n = original.size
        if n == 0:
            return
        take = min(self.sample_count, n)
        idx = np.linspace(0, n - 1, take, dtype=np.int64)
        self.original.extend(original[idx].tolist())
        for bits, pred in preds.items():
            self.pred[bits].extend(pred[idx].tolist())
        self.seen += n

    def arrays(self):
        if len(self.original) <= self.sample_count:
            idx = np.arange(len(self.original), dtype=np.int64)
        else:
            idx = np.linspace(0, len(self.original) - 1, self.sample_count, dtype=np.int64)
        original = np.asarray(self.original, dtype=np.float32)[idx]
        preds = {bits: np.asarray(vals, dtype=np.float32)[idx] for bits, vals in self.pred.items()}
        return original, preds


def polyline(values, width, height, pad, ymin, ymax):
    if values.size == 0:
        return ""
    if ymax <= ymin:
        ymax = ymin + 1.0
    xs = np.linspace(pad, width - pad, values.size)
    ys = height - pad - (values - ymin) / (ymax - ymin) * (height - 2 * pad)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def layer_svg(layer, original, preds, width=1100, height=310):
    pad = 38
    all_values = [original] + [p for p in preds.values()]
    ymin = float(min(np.min(v) for v in all_values if v.size))
    ymax = float(max(np.max(v) for v in all_values if v.size))
    lines = [
        ("original", original, "#111", 1.8),
        ("FIT_2", preds[2], "#d62728", 1.1),
        ("FIT_3", preds[3], "#ff7f0e", 1.1),
        ("FIT_4", preds[4], "#2ca02c", 1.1),
    ]
    paths = []
    legend = []
    for i, (name, vals, color, stroke) in enumerate(lines):
        paths.append(f'<polyline points="{polyline(vals, width, height, pad, ymin, ymax)}" fill="none" stroke="{color}" stroke-width="{stroke}"/>')
        legend.append(f'<span><i style="background:{color}"></i>{html.escape(name)}</span>')
    return f"""
<section class="chart">
<h2>{html.escape(layer)}</h2>
<svg viewBox="0 0 {width} {height}" role="img">
<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#ddd"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#ddd"/>
<text x="{pad}" y="22" font-size="12" fill="#555">min {ymin:.5g} / max {ymax:.5g}</text>
{''.join(paths)}
</svg>
<div class="legend">{''.join(legend)}</div>
</section>
"""


def write_html(path, layer_stats, samplers, rows, sample_count):
    summary_rows = []
    for layer in sorted(layer_stats, key=layer_sort_key):
        for bits in (2, 3, 4):
            st = layer_stats[layer][bits]
            n = max(st["n"], 1)
            rmse = float(np.sqrt(st["sse"] / n))
            rel_rmse = rmse / (float(np.sqrt(st["orig2"] / n)) + 1e-12)
            cosine = st["dot"] / (float(np.sqrt(st["orig2"]) * np.sqrt(st["pred2"])) + 1e-12)
            summary_rows.append(
                f"<tr><td>{html.escape(layer)}</td><td>FIT_{bits}</td><td>{st['n']:,}</td>"
                f"<td>{rel_rmse:.6f}</td><td>{cosine:.6f}</td><td>{st['saved_pct']:.2f}%</td></tr>"
            )

    charts = []
    for layer in sorted(samplers, key=layer_sort_key):
        original, preds = samplers[layer].arrays()
        charts.append(layer_svg(layer, original, preds))

    worst = sorted(rows, key=lambda r: r["rel_rmse"], reverse=True)[:40]
    worst_rows = "".join(
        f"<tr><td>{html.escape(r['tensor'])}</td><td>{r['fit']}</td><td>{r['rel_rmse']:.6f}</td>"
        f"<td>{r['cosine']:.6f}</td><td>{r['saved_pct']:.2f}%</td></tr>"
        for r in worst
    )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SWQ FIT original vs predicted</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 28px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.chart {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 18px 0; }}
.legend span {{ margin-right: 18px; font-size: 13px; }}
.legend i {{ display: inline-block; width: 16px; height: 3px; margin-right: 5px; vertical-align: middle; }}
code {{ background: #f5f5f5; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>SWQ FIT original vs predicted</h1>
<p>Each graph compares original FP16 layer samples against predicted values from FIT_2, FIT_3, and FIT_4. Metrics are computed over all selected tensor values. Graphs are downsampled to {sample_count} points per layer so the browser stays usable.</p>
<h2>Layer metrics</h2>
<table><thead><tr><th>Layer</th><th>Format</th><th>Values</th><th>Rel RMSE</th><th>Cosine</th><th>Saved</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<h2>Worst tensor-format pairs</h2>
<table><thead><tr><th>Tensor</th><th>Format</th><th>Rel RMSE</th><th>Cosine</th><th>Saved</th></tr></thead><tbody>{worst_rows}</tbody></table>
<h2>Layer comparison graphs</h2>
{''.join(charts)}
</body>
</html>
"""
    Path(path).write_text(doc)


def main():
    parser = argparse.ArgumentParser(description="Generate layer-by-layer original vs SWQ FIT prediction graphs.")
    parser.add_argument("model")
    parser.add_argument("--include", default=r"blk\.[0-9]+\..*\.weight")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=900)
    parser.add_argument("--chunk-blocks", type=int, default=4096)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--html", required=True)
    args = parser.parse_args()

    include_re = re.compile(args.include) if args.include else None
    rows = []
    layer_stats = {}
    samplers = {}
    checked = 0

    for tensor in GGUFReader(args.model).tensors:
        if include_re and not include_re.search(tensor.name):
            continue
        values = tensor_to_float(tensor)
        if values is None or values.size == 0:
            continue

        layer = layer_name(tensor.name)
        preds = {bits: fit_predict(values, args.block_size, bits, args.chunk_blocks) for bits in (2, 3, 4)}
        samplers.setdefault(layer, LayerSampler(args.sample_count)).add(values, preds)

        n_blocks = ceil_div(values.size, args.block_size)
        original_bytes = values.size * 2
        for bits, pred in preds.items():
            m = metrics(values, pred)
            swq_bytes = n_blocks * estimated_block_bytes(args.block_size, bits)
            saved_pct = (1.0 - swq_bytes / max(original_bytes, 1)) * 100.0
            row = {
                "tensor": tensor.name,
                "layer": layer,
                "fit": f"FIT_{bits}",
                "bits": bits,
                "n": int(values.size),
                "rel_rmse": m["rel_rmse"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "cosine": m["cosine"],
                "saved_pct": saved_pct,
            }
            rows.append(row)

            st = layer_stats.setdefault(layer, {}).setdefault(bits, {
                "n": 0, "sse": 0.0, "orig2": 0.0, "pred2": 0.0, "dot": 0.0,
                "original_bytes": 0, "swq_bytes": 0, "saved_pct": 0.0,
            })
            st["n"] += m["n"]
            st["sse"] += m["sse"]
            st["orig2"] += m["orig2"]
            st["pred2"] += m["pred2"]
            st["dot"] += m["dot"]
            st["original_bytes"] += original_bytes
            st["swq_bytes"] += swq_bytes
            st["saved_pct"] = (1.0 - st["swq_bytes"] / max(st["original_bytes"], 1)) * 100.0

        checked += 1
        print(f"{checked:4d} {tensor.name}", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = ["tensor", "layer", "fit", "bits", "n", "rel_rmse", "rmse", "mae", "cosine", "saved_pct"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_html(args.html, layer_stats, samplers, rows, args.sample_count)
    print(args.html)


if __name__ == "__main__":
    main()
