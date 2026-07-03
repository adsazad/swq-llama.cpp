#!/usr/bin/env python3
import argparse
import csv
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


QK_SWQ_FIT_2 = 128
QK_SWQ_FIT_3 = 128
QK_SWQ_4 = 128


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
    lo = qs[:, byte_pos] >> shifts
    crosses = shifts > 5
    if np.any(crosses):
        lo[:, crosses] |= qs[:, byte_pos[crosses] + 1] << (8 - shifts[crosses])
    indices = lo & 0x07

    t = np.linspace(-1.0, 1.0, QK_SWQ_FIT_3, dtype=np.float32)
    powers = np.stack([np.ones_like(t), t, t * t, t * t * t], axis=0)
    base = coeffs @ powers
    residual = residuals[np.arange(residuals.shape[0])[:, None], indices]
    return (base + residual).reshape(-1)


def dequant_swq_4(data):
    raw = np.asarray(data, dtype=np.uint8).reshape(-1, 96)
    codebook = raw[:, 0:32].copy().view("<f2").astype(np.float32).reshape(-1, 16)
    qs = raw[:, 32:96]

    low = qs & 0x0f
    high = qs >> 4
    indices = np.concatenate([low, high], axis=1)
    return codebook[np.arange(codebook.shape[0])[:, None], indices].reshape(-1)


def tensor_to_float(tensor):
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.F16:
        return np.asarray(tensor.data).astype(np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.Q_SWQ_FIT_2:
        return dequant_swq_fit_2(tensor.data)
    if tensor.tensor_type == GGMLQuantizationType.Q_SWQ_FIT_3:
        return dequant_swq_fit_3(tensor.data)
    if tensor.tensor_type == GGMLQuantizationType.Q_SWQ_4:
        return dequant_swq_4(tensor.data)
    return None


def layer_name(tensor_name):
    m = re.match(r"blk\.(\d+)\.", tensor_name)
    if m:
        return f"blk.{m.group(1)}"
    return "non_block"


def metrics(name, original, approx):
    diff = approx - original
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(original * original)))
    rel_rmse = rmse / (rms + 1e-12)
    denom = float(np.linalg.norm(original) * np.linalg.norm(approx))
    cosine = float(np.dot(original, approx) / denom) if denom > 0 else 0.0
    return {
        "name": name,
        "n": int(original.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SWQ tensor reconstruction against an FP16/F32 GGUF.")
    parser.add_argument("original")
    parser.add_argument("quantized")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    original = {t.name: t for t in GGUFReader(args.original).tensors}
    quantized = GGUFReader(args.quantized).tensors

    rows = []
    layer_stats = {}
    checked = 0

    for qt in quantized:
        if qt.name not in original:
            continue
        if qt.tensor_type not in (GGMLQuantizationType.Q_SWQ_FIT_2, GGMLQuantizationType.Q_SWQ_FIT_3, GGMLQuantizationType.Q_SWQ_4):
            continue
        orig = tensor_to_float(original[qt.name])
        approx = tensor_to_float(qt)
        if orig is None or approx is None or orig.size != approx.size:
            continue
        row = metrics(qt.name, orig, approx)
        row["type"] = qt.tensor_type.name
        row["layer"] = layer_name(qt.name)
        rows.append(row)

        layer = row["layer"]
        st = layer_stats.setdefault(layer, {"n": 0, "sse": 0.0, "sae": 0.0, "orig2": 0.0, "dot": 0.0, "approx2": 0.0})
        diff = approx - orig
        st["n"] += int(orig.size)
        st["sse"] += float(np.sum(diff * diff))
        st["sae"] += float(np.sum(np.abs(diff)))
        st["orig2"] += float(np.sum(orig * orig))
        st["dot"] += float(np.dot(orig, approx))
        st["approx2"] += float(np.sum(approx * approx))

        checked += 1
        if args.limit and checked >= args.limit:
            break

    rows.sort(key=lambda r: r["rel_rmse"], reverse=True)

    print("Per-layer reconstruction summary")
    print("layer,n,rmse,mae,rel_rmse,cosine")
    for layer in sorted(layer_stats.keys(), key=lambda x: (x != "non_block", int(x.split(".")[1]) if x.startswith("blk.") else -1)):
        st = layer_stats[layer]
        rmse = np.sqrt(st["sse"] / st["n"])
        mae = st["sae"] / st["n"]
        rel_rmse = rmse / (np.sqrt(st["orig2"] / st["n"]) + 1e-12)
        denom = np.sqrt(st["orig2"]) * np.sqrt(st["approx2"])
        cosine = st["dot"] / denom if denom > 0 else 0.0
        print(f"{layer},{st['n']},{rmse:.8g},{mae:.8g},{rel_rmse:.8g},{cosine:.8g}")

    print()
    print("Worst tensors by relative RMSE")
    print("tensor,type,n,rmse,mae,max_abs,rel_rmse,cosine")
    for row in rows[:25]:
        print(f"{row['name']},{row['type']},{row['n']},{row['rmse']:.8g},{row['mae']:.8g},{row['max_abs']:.8g},{row['rel_rmse']:.8g},{row['cosine']:.8g}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "name", "type", "n", "mse", "rmse", "mae", "max_abs", "rel_rmse", "cosine"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
