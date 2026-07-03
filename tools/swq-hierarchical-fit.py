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


BLOCK_SIZE = 256
SEGMENT_SIZE = 128
SEGMENTS = 2
COEFFICIENTS = 4
RESIDUAL_LEVELS = 8
BASE_BLOCK_BYTES = 8 + 16 + 96


def tensor_to_float(tensor):
    if tensor.tensor_type == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    return None


def ceil_div(a, b):
    return (a + b - 1) // b


def fp16_round(values):
    return values.astype(np.float16).astype(np.float32)


def quantize_coefficients(coefficients, scale_mode):
    if scale_mode == "tensor":
        axis = 0
    elif scale_mode == "block":
        axis = (1, 2)
    elif scale_mode == "segment":
        axis = 2
    else:
        raise ValueError(scale_mode)
    scales = np.max(np.abs(coefficients), axis=axis, keepdims=True) / 127.0
    scales = fp16_round(scales)
    scales = np.where(scales == 0.0, 1.0, scales)
    quantized = np.clip(np.rint(coefficients / scales), -127, 127).astype(np.int8)
    return quantized.astype(np.float32) * scales, scales


def predict(coefficients, powers):
    return np.matmul(coefficients, powers.T).reshape(-1, BLOCK_SIZE)


def initialize_codebook(residual):
    rmin = np.min(residual, axis=1, keepdims=True)
    rmax = np.max(residual, axis=1, keepdims=True)
    alpha = np.linspace(0.0, 1.0, RESIDUAL_LEVELS, dtype=np.float32)[None, :]
    return fp16_round(rmin + (rmax - rmin) * alpha)


def assign_codebook(residual, codebook):
    best_index = np.zeros(residual.shape, dtype=np.uint8)
    best_distance = np.full(residual.shape, np.inf, dtype=np.float32)
    for level in range(RESIDUAL_LEVELS):
        distance = np.abs(residual - codebook[:, level:level + 1])
        better = distance < best_distance
        best_distance = np.where(better, distance, best_distance)
        best_index = np.where(better, level, best_index)
    return best_index


def update_codebook(residual, indices, codebook):
    updated = codebook.copy()
    for level in range(RESIDUAL_LEVELS):
        mask = indices == level
        counts = np.sum(mask, axis=1)
        sums = np.sum(np.where(mask, residual, 0.0), axis=1)
        valid = counts > 0
        updated[valid, level] = sums[valid] / counts[valid]
    return fp16_round(updated)


def gather_codebook(codebook, indices):
    return np.take_along_axis(codebook, indices.astype(np.int64), axis=1)


def relative_rmse(original, predicted):
    diff = predicted - original
    rmse = float(np.sqrt(np.mean(diff * diff)))
    source_rms = float(np.sqrt(np.mean(original * original)))
    return rmse / (source_rms + 1e-12)


def reconstruction_stats(original, predicted, stored_bytes):
    n = original.size
    diff = predicted - original
    sse = float(np.sum(diff * diff))
    sae = float(np.sum(np.abs(diff)))
    orig2 = float(np.sum(original * original))
    pred2 = float(np.sum(predicted * predicted))
    dot = float(np.dot(original, predicted))
    rmse = float(np.sqrt(sse / max(n, 1)))
    return {
        "sse": sse,
        "sae": sae,
        "orig2": orig2,
        "pred2": pred2,
        "dot": dot,
        "rmse": rmse,
        "mae": sae / max(n, 1),
        "rel_rmse": rmse / (float(np.sqrt(orig2 / max(n, 1))) + 1e-12),
        "cosine": dot / (float(np.sqrt(orig2) * np.sqrt(pred2)) + 1e-12),
        "stored_bytes": int(stored_bytes),
    }


def scale_storage(n_blocks, scale_mode):
    if scale_mode == "tensor":
        return SEGMENTS * COEFFICIENTS * 2
    if scale_mode == "block":
        return n_blocks * 2
    if scale_mode == "segment":
        return n_blocks * SEGMENTS * 2
    raise ValueError(scale_mode)


def train_tensor(values, powers, pinv, epochs, residual_epochs, tolerance, patience, anchor_counts, scale_mode):
    n = values.size
    n_blocks = ceil_div(n, BLOCK_SIZE)
    padded = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.float32)
    padded[:n] = values
    blocks = padded.reshape(n_blocks, BLOCK_SIZE)
    segments = blocks.reshape(n_blocks, SEGMENTS, SEGMENT_SIZE)

    coefficients = np.matmul(segments, pinv.T)
    quantized_coefficients, scales = quantize_coefficients(coefficients, scale_mode)
    base = predict(quantized_coefficients, powers)
    codebook = initialize_codebook(blocks - base)

    history = []
    stale = 0
    previous = None
    for epoch in range(1, epochs + 1):
        quantized_coefficients, scales = quantize_coefficients(coefficients, scale_mode)
        base = predict(quantized_coefficients, powers)
        residual = blocks - base

        for _ in range(residual_epochs):
            indices = assign_codebook(residual, codebook)
            codebook = update_codebook(residual, indices, codebook)

        corrections = gather_codebook(codebook, indices)
        corrected_target = (blocks - corrections).reshape(n_blocks, SEGMENTS, SEGMENT_SIZE)
        coefficients = np.matmul(corrected_target, pinv.T)

        quantized_coefficients, scales = quantize_coefficients(coefficients, scale_mode)
        base = predict(quantized_coefficients, powers)
        residual = blocks - base
        indices = assign_codebook(residual, codebook)
        reconstructed = base + gather_codebook(codebook, indices)
        score = relative_rmse(blocks.reshape(-1)[:n], reconstructed.reshape(-1)[:n])
        history.append(score)

        if previous is not None and previous - score < tolerance:
            stale += 1
        else:
            stale = 0
        previous = score
        if stale >= patience:
            break

    original = blocks.reshape(-1)[:n]
    predicted_values = reconstructed.reshape(-1)[:n]
    original_bytes = n * 2
    base = reconstruction_stats(
        original,
        predicted_values,
        n_blocks * BASE_BLOCK_BYTES + scale_storage(n_blocks, scale_mode),
    )

    anchor_stats = {}
    block_rows = np.arange(n_blocks)[:, None]
    for anchor_count in anchor_counts:
        if anchor_count == 0:
            anchor_stats[anchor_count] = base
            continue
        anchored = reconstructed.copy()
        errors = np.abs(reconstructed - blocks)
        positions = np.argpartition(errors, -anchor_count, axis=1)[:, -anchor_count:]
        anchored[block_rows, positions] = fp16_round(blocks[block_rows, positions])
        anchor_stats[anchor_count] = reconstruction_stats(
            original,
            anchored.reshape(-1)[:n],
            n_blocks * (BASE_BLOCK_BYTES + 3 * anchor_count) + scale_storage(n_blocks, scale_mode),
        )

    return {
        "n": int(n),
        "blocks": int(n_blocks),
        "epochs_run": len(history),
        "first_epoch_rel_rmse": history[0],
        "final_epoch_rel_rmse": history[-1],
        "history": history,
        **base,
        "anchor_stats": anchor_stats,
        "original_bytes": int(original_bytes),
        "ratio": original_bytes / max(base["stored_bytes"], 1),
        "saved_pct": (1.0 - base["stored_bytes"] / max(original_bytes, 1)) * 100.0,
    }


def summary(rows):
    n = sum(row["n"] for row in rows)
    original_bytes = sum(row["original_bytes"] for row in rows)
    stored_bytes = sum(row["stored_bytes"] for row in rows)
    sse = sum(row["sse"] for row in rows)
    sae = sum(row["sae"] for row in rows)
    orig2 = sum(row["orig2"] for row in rows)
    pred2 = sum(row["pred2"] for row in rows)
    dot = sum(row["dot"] for row in rows)
    rmse = float(np.sqrt(sse / max(n, 1)))
    return {
        "n": n,
        "original_bytes": original_bytes,
        "stored_bytes": stored_bytes,
        "ratio": original_bytes / max(stored_bytes, 1),
        "saved_pct": (1.0 - stored_bytes / max(original_bytes, 1)) * 100.0,
        "rmse": rmse,
        "mae": sae / max(n, 1),
        "rel_rmse": rmse / (float(np.sqrt(orig2 / max(n, 1))) + 1e-12),
        "cosine": dot / (float(np.sqrt(orig2) * np.sqrt(pred2)) + 1e-12),
        "mean_epochs": sum(row["epochs_run"] for row in rows) / max(len(rows), 1),
    }


def anchor_summary(rows, anchor_count):
    n = sum(row["n"] for row in rows)
    original_bytes = sum(row["original_bytes"] for row in rows)
    stored_bytes = sum(row["anchor_stats"][anchor_count]["stored_bytes"] for row in rows)
    sse = sum(row["anchor_stats"][anchor_count]["sse"] for row in rows)
    orig2 = sum(row["anchor_stats"][anchor_count]["orig2"] for row in rows)
    pred2 = sum(row["anchor_stats"][anchor_count]["pred2"] for row in rows)
    dot = sum(row["anchor_stats"][anchor_count]["dot"] for row in rows)
    rmse = float(np.sqrt(sse / max(n, 1)))
    return {
        "anchors": anchor_count,
        "stored_bytes": stored_bytes,
        "ratio": original_bytes / max(stored_bytes, 1),
        "saved_pct": (1.0 - stored_bytes / max(original_bytes, 1)) * 100.0,
        "rel_rmse": rmse / (float(np.sqrt(orig2 / max(n, 1))) + 1e-12),
        "cosine": dot / (float(np.sqrt(orig2) * np.sqrt(pred2)) + 1e-12),
    }


def bar_svg(values, labels, title, maximum, suffix=""):
    width = 820
    height = 55 + len(values) * 44
    elements = []
    for i, (value, label) in enumerate(zip(values, labels)):
        y = 40 + i * 44
        bar_width = 520.0 * value / maximum
        elements.append(
            f'<text x="8" y="{y + 18}" font-size="13">{html.escape(label)}</text>'
            f'<rect x="185" y="{y}" width="{bar_width:.1f}" height="25" fill="#4472c4"/>'
            f'<text x="{195 + bar_width:.1f}" y="{y + 18}" font-size="13">{value:.6f}{suffix}</text>'
        )
    return f'<h2>{html.escape(title)}</h2><svg viewBox="0 0 {width} {height}">{"".join(elements)}</svg>'


def write_html(path, rows, total, anchors, epochs, residual_epochs, scale_mode):
    worst = sorted(rows, key=lambda row: row["rel_rmse"], reverse=True)[:50]
    worst_rows = "".join(
        f"<tr><td>{html.escape(row['tensor'])}</td><td>{row['epochs_run']}</td>"
        f"<td>{row['rel_rmse']:.6f}</td><td>{row['cosine']:.6f}</td>"
        f"<td>{row['saved_pct']:.2f}%</td></tr>"
        for row in worst
    )
    comparison = bar_svg(
        [0.169957, 0.193315, total["rel_rmse"]],
        ["FIT3-128 offline", "one-cubic FIT3-256", "hierarchical INT8 FIT3-256"],
        "Relative RMSE comparison",
        max(0.193315, total["rel_rmse"]) * 1.1,
    )
    savings = bar_svg(
        [71.875, 76.562, total["saved_pct"]],
        ["FIT3-128 offline", "one-cubic FIT3-256", "hierarchical INT8 FIT3-256"],
        "Estimated storage savings",
        100.0,
        "%",
    )
    anchor_rows = "".join(
        f"<tr><td>{item['anchors']}</td><td>{item['stored_bytes']:,}</td>"
        f"<td>{item['ratio']:.3f}x</td><td>{item['saved_pct']:.2f}%</td>"
        f"<td>{item['rel_rmse']:.6f}</td><td>{item['cosine']:.6f}</td></tr>"
        for item in anchors
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hierarchical INT8 FIT3 experiment</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 28px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
svg {{ max-width: 820px; width: 100%; }}
</style></head><body>
<h1>Hierarchical INT8 FIT3-256 joint-training experiment</h1>
<p>Two 128-weight cubic predictors, INT8 coefficients with tensor-level FP16 scales, one shared eight-value residual codebook, and 256 packed 3-bit residual indices.</p>
<h2>Summary</h2>
<table><tbody>
<tr><th>Configured epochs</th><td>{epochs} fit x {residual_epochs} residual</td></tr>
<tr><th>Coefficient scale</th><td>{html.escape(scale_mode)}</td></tr>
<tr><th>Mean epochs run</th><td>{total['mean_epochs']:.2f}</td></tr>
<tr><th>Stored bytes</th><td>{total['stored_bytes']:,}</td></tr>
<tr><th>Compression ratio</th><td>{total['ratio']:.3f}x</td></tr>
<tr><th>Estimated saved</th><td>{total['saved_pct']:.2f}%</td></tr>
<tr><th>Relative RMSE</th><td>{total['rel_rmse']:.6f}</td></tr>
<tr><th>Cosine similarity</th><td>{total['cosine']:.6f}</td></tr>
</tbody></table>
{comparison}{savings}
<h2>Exact FP16 anchor sweep</h2>
<table><thead><tr><th>Anchors per 256 weights</th><th>Stored bytes</th><th>Ratio</th><th>Saved</th><th>Rel RMSE</th><th>Cosine</th></tr></thead><tbody>{anchor_rows}</tbody></table>
<h2>Worst tensors</h2>
<table><thead><tr><th>Tensor</th><th>Epochs</th><th>Rel RMSE</th><th>Cosine</th><th>Saved</th></tr></thead><tbody>{worst_rows}</tbody></table>
</body></html>"""
    Path(path).write_text(document)


def main():
    parser = argparse.ArgumentParser(description="Train the hierarchical INT8-cubic SWQ FIT3-256 experiment.")
    parser.add_argument("model")
    parser.add_argument("--include", default=r"blk\.[0-9]+\..*\.weight")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--residual-epochs", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--anchors", default="0,1,2,3")
    parser.add_argument("--coefficient-scale", choices=("tensor", "block", "segment"), default="tensor")
    parser.add_argument("--limit-tensors", type=int, default=0)
    parser.add_argument("--csv")
    parser.add_argument("--html")
    args = parser.parse_args()
    anchor_counts = sorted({int(value) for value in args.anchors.split(",")})
    if not anchor_counts or anchor_counts[0] < 0 or anchor_counts[-1] >= BLOCK_SIZE:
        parser.error("--anchors values must be between 0 and 255")

    x = np.linspace(-1.0, 1.0, SEGMENT_SIZE, dtype=np.float32)
    powers = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=1)
    pinv = np.linalg.pinv(powers).astype(np.float32)
    include_re = re.compile(args.include) if args.include else None
    rows = []

    for tensor in GGUFReader(args.model).tensors:
        if include_re and not include_re.search(tensor.name):
            continue
        values = tensor_to_float(tensor)
        if values is None or values.size == 0:
            continue
        stats = train_tensor(
            values, powers, pinv, args.epochs, args.residual_epochs,
            args.tolerance, args.patience, anchor_counts, args.coefficient_scale,
        )
        row = {"tensor": tensor.name, **stats}
        rows.append(row)
        print(
            f"{len(rows):4d} {tensor.name}: epochs={stats['epochs_run']} "
            f"rel_rmse={stats['rel_rmse']:.6f} cosine={stats['cosine']:.6f} "
            f"saved={stats['saved_pct']:.2f}%",
            flush=True,
        )
        if args.limit_tensors and len(rows) >= args.limit_tensors:
            break

    total = summary(rows)
    anchor_totals = [anchor_summary(rows, count) for count in anchor_counts]
    print("\nHierarchical INT8 FIT3-256 summary")
    print(f"tensors={len(rows)}")
    for key in ("original_bytes", "stored_bytes", "ratio", "saved_pct", "rmse", "rel_rmse", "cosine", "mean_epochs"):
        print(f"{key}={total[key]}")
    print("anchors,stored_bytes,ratio,saved_pct,rel_rmse,cosine")
    for item in anchor_totals:
        print(
            f"{item['anchors']},{item['stored_bytes']},{item['ratio']:.6f},"
            f"{item['saved_pct']:.6f},{item['rel_rmse']:.6f},{item['cosine']:.6f}"
        )

    if args.csv:
        fieldnames = [
            "tensor", "n", "blocks", "epochs_run", "first_epoch_rel_rmse",
            "final_epoch_rel_rmse", "original_bytes", "stored_bytes", "ratio",
            "saved_pct", "rmse", "mae", "rel_rmse", "cosine",
        ]
        with open(args.csv, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{key: row[key] for key in fieldnames} for row in rows])
    if args.html:
        write_html(args.html, rows, total, anchor_totals, args.epochs, args.residual_epochs, args.coefficient_scale)
        print(args.html)


if __name__ == "__main__":
    main()
