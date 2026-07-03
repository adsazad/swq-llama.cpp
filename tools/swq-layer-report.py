#!/usr/bin/env python3
import argparse
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


QK_SWQ_4 = 128
QK_SWQ_FIT_2 = 128
QK_SWQ_FIT_3 = 128


def dequant_swq_4(data):
    raw = np.asarray(data, dtype=np.uint8).reshape(-1, 96)
    codebook = raw[:, 0:32].copy().view("<f2").astype(np.float32).reshape(-1, 16)
    qs = raw[:, 32:96]
    low = qs & 0x0f
    high = qs >> 4
    indices = np.concatenate([low, high], axis=1)
    return codebook[np.arange(codebook.shape[0])[:, None], indices].reshape(-1)


def dequant_swq_fit_2(data):
    raw = np.asarray(data, dtype=np.uint8).reshape(-1, 48)
    coeffs = raw[:, 0:8].copy().view("<f2").astype(np.float32).reshape(-1, 4)
    residuals = raw[:, 8:16].copy().view("<f2").astype(np.float32).reshape(-1, 4)
    qs = raw[:, 16:48]
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    indices = ((qs[:, :, None] >> shifts[None, None, :]) & 0x03).reshape(-1, QK_SWQ_FIT_2)

    t = np.linspace(-1.0, 1.0, QK_SWQ_FIT_2, dtype=np.float32)
    powers = np.stack([np.ones_like(t), t, t * t, t * t * t], axis=0)
    base = coeffs @ powers
    residual = residuals[np.arange(residuals.shape[0])[:, None], indices]
    return (base + residual).reshape(-1)


def dequant_swq_fit_3(data):
    raw = np.asarray(data, dtype=np.uint8).reshape(-1, 72)
    coeffs = raw[:, 0:8].copy().view("<f2").astype(np.float32).reshape(-1, 4)
    residuals = raw[:, 8:24].copy().view("<f2").astype(np.float32).reshape(-1, 8)
    qs = raw[:, 24:72]
    bit_pos = np.arange(QK_SWQ_FIT_3, dtype=np.uint16) * 3
    byte_pos = bit_pos // 8
    shifts = bit_pos % 8
    indices = qs[:, byte_pos] >> shifts
    crosses = shifts > 5
    if np.any(crosses):
        indices[:, crosses] |= qs[:, byte_pos[crosses] + 1] << (8 - shifts[crosses])
    indices &= 0x07

    t = np.linspace(-1.0, 1.0, QK_SWQ_FIT_3, dtype=np.float32)
    powers = np.stack([np.ones_like(t), t, t * t, t * t * t], axis=0)
    base = coeffs @ powers
    residual = residuals[np.arange(residuals.shape[0])[:, None], indices]
    return (base + residual).reshape(-1)


def tensor_to_float(tensor):
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.F16:
        return np.asarray(tensor.data).astype(np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.Q_SWQ_4:
        return dequant_swq_4(tensor.data)
    if tensor.tensor_type == GGMLQuantizationType.Q_SWQ_FIT_2:
        return dequant_swq_fit_2(tensor.data)
    if tensor.tensor_type == GGMLQuantizationType.Q_SWQ_FIT_3:
        return dequant_swq_fit_3(tensor.data)
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


def reconstruction_metrics(original, approx):
    diff = approx - original
    sse = float(np.sum(diff * diff))
    sae = float(np.sum(np.abs(diff)))
    orig2 = float(np.sum(original * original))
    approx2 = float(np.sum(approx * approx))
    dot = float(np.dot(original, approx))
    n = int(original.size)
    rmse = float(np.sqrt(sse / n))
    mae = sae / n
    rel_rmse = rmse / (float(np.sqrt(orig2 / n)) + 1e-12)
    cosine = dot / (float(np.sqrt(orig2) * np.sqrt(approx2)) + 1e-12)
    return {
        "n": n,
        "sse": sse,
        "sae": sae,
        "orig2": orig2,
        "approx2": approx2,
        "dot": dot,
        "rmse": rmse,
        "mae": mae,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
    }


def polyline(values, width, height, pad, ymin, ymax):
    if len(values) == 0:
        return ""
    if ymax <= ymin:
        ymax = ymin + 1.0
    xs = np.linspace(pad, width - pad, len(values))
    ys = height - pad - (values - ymin) / (ymax - ymin) * (height - 2 * pad)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def line_svg(original, approx, title, width=900, height=260):
    pad = 36
    ymin = float(min(np.min(original), np.min(approx)))
    ymax = float(max(np.max(original), np.max(approx)))
    op = polyline(original, width, height, pad, ymin, ymax)
    pp = polyline(approx, width, height, pad, ymin, ymax)
    safe_title = html.escape(title)
    return f"""
<div class="chart-card">
  <h3>{safe_title}</h3>
  <svg viewBox="0 0 {width} {height}" role="img">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
    <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#ddd"/>
    <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#ddd"/>
    <text x="{pad}" y="20" font-size="12" fill="#555">min {ymin:.5g} / max {ymax:.5g}</text>
    <polyline points="{op}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>
    <polyline points="{pp}" fill="none" stroke="#d62728" stroke-width="1.5"/>
  </svg>
  <div class="legend"><span class="blue"></span> original <span class="red"></span> prediction</div>
</div>
"""


def layer_bar_svg(layers, metric, title, width=1100, height=340):
    pad_l = 60
    pad_b = 60
    pad_t = 24
    values = np.array([l[metric] for l in layers], dtype=np.float32)
    vmax = float(max(np.max(values), 1e-9))
    bar_w = (width - pad_l - 20) / max(len(layers), 1)
    bars = []
    labels = []
    for i, layer in enumerate(layers):
        x = pad_l + i * bar_w + 1
        h = float(values[i] / vmax * (height - pad_t - pad_b))
        y = height - pad_b - h
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 2, 1):.2f}" height="{h:.2f}" fill="#6a5acd"/>')
        if layer["layer"] != "non_block":
            labels.append(f'<text x="{x + bar_w/2:.2f}" y="{height - 38}" transform="rotate(60 {x + bar_w/2:.2f},{height - 38})" font-size="10">{html.escape(layer["layer"])}</text>')
    return f"""
<div class="chart-card">
  <h3>{html.escape(title)}</h3>
  <svg viewBox="0 0 {width} {height}" role="img">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>
    <line x1="{pad_l}" y1="{height-pad_b}" x2="{width-20}" y2="{height-pad_b}" stroke="#ddd"/>
    <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#ddd"/>
    <text x="8" y="{pad_t + 8}" font-size="12">max {vmax:.5g}</text>
    {''.join(bars)}
    {''.join(labels)}
  </svg>
</div>
"""


def html_report(original_path, quant_path, output_path, sample_count, max_layers):
    original_reader = GGUFReader(original_path)
    quant_reader = GGUFReader(quant_path)
    originals = {t.name: t for t in original_reader.tensors}

    tensor_rows = []
    layer_acc = {}
    layer_examples = {}

    for qt in quant_reader.tensors:
        if qt.name not in originals:
            continue
        if qt.tensor_type not in (GGMLQuantizationType.Q_SWQ_4, GGMLQuantizationType.Q_SWQ_FIT_2, GGMLQuantizationType.Q_SWQ_FIT_3):
            continue
        orig = tensor_to_float(originals[qt.name])
        pred = tensor_to_float(qt)
        if orig is None or pred is None or orig.size != pred.size:
            continue

        m = reconstruction_metrics(orig, pred)
        layer = layer_name(qt.name)
        row = {
            "name": qt.name,
            "layer": layer,
            "type": qt.tensor_type.name,
            **m,
        }
        tensor_rows.append(row)

        st = layer_acc.setdefault(layer, {"layer": layer, "n": 0, "sse": 0.0, "sae": 0.0, "orig2": 0.0, "approx2": 0.0, "dot": 0.0})
        st["n"] += m["n"]
        st["sse"] += m["sse"]
        st["sae"] += m["sae"]
        st["orig2"] += m["orig2"]
        st["approx2"] += m["approx2"]
        st["dot"] += m["dot"]

        old = layer_examples.get(layer)
        if old is None or row["rel_rmse"] > old["row"]["rel_rmse"]:
            layer_examples[layer] = {"row": row, "orig": orig[:sample_count].copy(), "pred": pred[:sample_count].copy()}

    layers = []
    for st in layer_acc.values():
        n = st["n"]
        rmse = float(np.sqrt(st["sse"] / n))
        mae = st["sae"] / n
        rel_rmse = rmse / (float(np.sqrt(st["orig2"] / n)) + 1e-12)
        cosine = st["dot"] / (float(np.sqrt(st["orig2"]) * np.sqrt(st["approx2"])) + 1e-12)
        layers.append({"layer": st["layer"], "n": n, "rmse": rmse, "mae": mae, "rel_rmse": rel_rmse, "cosine": cosine})

    layers.sort(key=lambda x: layer_sort_key(x["layer"]))
    tensor_rows.sort(key=lambda x: x["rel_rmse"], reverse=True)

    shown_layers = [l["layer"] for l in layers if l["layer"] != "non_block"][:max_layers]
    charts = []
    for layer in shown_layers:
        ex = layer_examples[layer]
        row = ex["row"]
        charts.append(line_svg(ex["orig"], ex["pred"], f'{layer}: {row["name"]} - worst tensor sample, rel_rmse={row["rel_rmse"]:.4f}, cosine={row["cosine"]:.4f}'))

    layer_table = "\n".join(
        f"<tr><td>{html.escape(l['layer'])}</td><td>{l['n']:,}</td><td>{l['rmse']:.6g}</td><td>{l['mae']:.6g}</td><td>{l['rel_rmse']:.6g}</td><td>{l['cosine']:.6g}</td></tr>"
        for l in layers
    )
    worst_table = "\n".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{html.escape(r['type'])}</td><td>{r['n']:,}</td><td>{r['rmse']:.6g}</td><td>{r['mae']:.6g}</td><td>{r['rel_rmse']:.6g}</td><td>{r['cosine']:.6g}</td></tr>"
        for r in tensor_rows[:30]
    )

    mean_rel = float(np.mean([l["rel_rmse"] for l in layers])) if layers else 0.0
    mean_cos = float(np.mean([l["cosine"] for l in layers])) if layers else 0.0

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SWQ layer reconstruction report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 28px; color: #1f2328; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .meta {{ color: #555; line-height: 1.5; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px; min-width: 180px; background: #fafafa; }}
    .big {{ font-size: 24px; font-weight: 700; }}
    .chart-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 18px 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f4f4; }}
    .legend {{ color: #555; font-size: 13px; }}
    .blue, .red {{ display: inline-block; width: 22px; height: 3px; vertical-align: middle; margin: 0 4px 0 12px; }}
    .blue {{ background: #1f77b4; }}
    .red {{ background: #d62728; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>SWQ layer reconstruction report</h1>
  <div class="meta">
    <div><b>Original:</b> <code>{html.escape(str(original_path))}</code></div>
    <div><b>Quantized:</b> <code>{html.escape(str(quant_path))}</code></div>
    <div><b>Meaning:</b> blue line is original FP16/F32 weights, red line is SWQ prediction after dequantization.</div>
  </div>

  <div class="cards">
    <div class="card"><div>Quantized tensors</div><div class="big">{len(tensor_rows)}</div></div>
    <div class="card"><div>Layers</div><div class="big">{len(layers)}</div></div>
    <div class="card"><div>Mean layer rel RMSE</div><div class="big">{mean_rel:.4f}</div></div>
    <div class="card"><div>Mean layer cosine</div><div class="big">{mean_cos:.4f}</div></div>
  </div>

  {layer_bar_svg(layers, "rel_rmse", "Layer relative RMSE - lower is better")}
  {layer_bar_svg(layers, "cosine", "Layer cosine similarity - higher is better")}

  <h2>Original vs prediction by layer</h2>
  <p>For each layer, this shows the worst reconstructed SWQ tensor in that layer and the first {sample_count} weights from that tensor.</p>
  {''.join(charts)}

  <h2>Per-layer metrics</h2>
  <table>
    <thead><tr><th>Layer</th><th>Weights</th><th>RMSE</th><th>MAE</th><th>Rel RMSE</th><th>Cosine</th></tr></thead>
    <tbody>{layer_table}</tbody>
  </table>

  <h2>Worst tensors</h2>
  <table>
    <thead><tr><th>Tensor</th><th>Type</th><th>Weights</th><th>RMSE</th><th>MAE</th><th>Rel RMSE</th><th>Cosine</th></tr></thead>
    <tbody>{worst_table}</tbody>
  </table>
</body>
</html>
"""
    Path(output_path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML graph report comparing original weights with SWQ predictions.")
    parser.add_argument("original")
    parser.add_argument("quantized")
    parser.add_argument("--out", default="swq-layer-report.html")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--max-layers", type=int, default=24)
    args = parser.parse_args()
    html_report(args.original, args.quantized, args.out, args.sample_count, args.max_layers)
    print(args.out)


if __name__ == "__main__":
    main()
