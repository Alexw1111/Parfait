"""窗口级归一化工具。

价格通道按历史最后收盘做比值，volume 按历史 z-score。
提供反归一化与基础 OHLC 约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowNormStats:
    """窗口归一化统计量。"""
    price_scale: float
    vol_mean: float
    vol_std: float


def _find_idx(features: Sequence[str], candidates: Sequence[str]) -> int:
    for name in candidates:
        if name in features:
            return int(features.index(name))
    raise ValueError(f"None of {candidates} found in features={list(features)}")


def get_feature_indices(features: Sequence[str]) -> Dict[str, int]:
    """解析 OHLCV 索引。"""
    feats = list(features)
    return {
        "open": _find_idx(feats, ["open_qfq", "open", "Open", "OPEN"]),
        "high": _find_idx(feats, ["high_qfq", "high", "High", "HIGH"]),
        "low": _find_idx(feats, ["low_qfq", "low", "Low", "LOW"]),
        "close": _find_idx(feats, ["close_qfq", "close", "Close", "CLOSE"]),
        "volume": _find_idx(feats, ["volume", "Volume", "VOL", "vol"]),
    }


def normalize_history_future(
    history: np.ndarray,
    future: np.ndarray,
    features: Sequence[str],
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, WindowNormStats]:
    """归一化一个 history+future 窗口。"""
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if history.ndim != 2 or future.ndim != 2:
        raise ValueError(f"history/future must be 2D arrays. got {history.shape} and {future.shape}")
    if history.shape[1] != future.shape[1]:
        raise ValueError(f"feature dim mismatch: {history.shape} vs {future.shape}")

    idx = get_feature_indices(features)
    price_idx = [idx["open"], idx["high"], idx["low"], idx["close"]]
    vol_idx = idx["volume"]

    price_scale = float(history[-1, idx["close"]])
    if not np.isfinite(price_scale) or abs(price_scale) < eps:
        price_scale = 1.0

    history_norm = history.copy()
    future_norm = future.copy()
    history_norm[:, price_idx] = history_norm[:, price_idx] / (price_scale + eps)
    future_norm[:, price_idx] = future_norm[:, price_idx] / (price_scale + eps)

    vol_mean = float(np.mean(history[:, vol_idx]))
    vol_std = float(np.std(history[:, vol_idx]))
    if not np.isfinite(vol_std) or vol_std < eps:
        vol_std = 1.0

    history_norm[:, vol_idx] = (history_norm[:, vol_idx] - vol_mean) / (vol_std + eps)
    future_norm[:, vol_idx] = (future_norm[:, vol_idx] - vol_mean) / (vol_std + eps)

    stats = WindowNormStats(price_scale=price_scale, vol_mean=vol_mean, vol_std=vol_std)
    return history_norm, future_norm, stats


def normalize_history(
    history: np.ndarray,
    features: Sequence[str],
    eps: float = 1e-8,
) -> Tuple[np.ndarray, WindowNormStats]:
    """仅归一化 history 窗口。"""
    dummy_future = history[-1:, :].copy()
    hist_n, _fut_n, stats = normalize_history_future(history, dummy_future, features, eps=eps)
    return hist_n, stats


def denormalize_future(
    future_norm: np.ndarray,
    stats: WindowNormStats,
    features: Sequence[str],
    eps: float = 1e-8,
    clamp_log_volume: bool = True,
) -> np.ndarray:
    """反归一化 future。"""
    future_norm = np.asarray(future_norm, dtype=np.float32)
    if future_norm.ndim != 2:
        raise ValueError(f"future_norm must be 2D [T,F], got {future_norm.shape}")

    idx = get_feature_indices(features)
    price_idx = [idx["open"], idx["high"], idx["low"], idx["close"]]
    vol_idx = idx["volume"]

    out = future_norm.copy()
    out[:, price_idx] = out[:, price_idx] * (stats.price_scale + eps)
    out[:, vol_idx] = out[:, vol_idx] * (stats.vol_std + eps) + stats.vol_mean

    if clamp_log_volume:
        # log1p(volume) >= 0
        out[:, vol_idx] = np.maximum(out[:, vol_idx], 0.0)
    return out


def enforce_ohlc_constraints(
    ohlcv: np.ndarray,
    features: Sequence[str],
    eps: float = 1e-8,
    clamp_nonnegative_prices: bool = True,
) -> np.ndarray:
    """硬约束 OHLC。"""
    x = np.asarray(ohlcv, dtype=np.float32).copy()
    idx = get_feature_indices(features)

    o = x[:, idx["open"]]
    h = x[:, idx["high"]]
    l = x[:, idx["low"]]
    c = x[:, idx["close"]]

    # NaN/inf -> 1.0
    for arr in (o, h, l, c):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = 1.0

    if clamp_nonnegative_prices:
        o = np.maximum(o, eps)
        c = np.maximum(c, eps)
        h = np.maximum(h, eps)
        l = np.maximum(l, eps)

    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)
    h = np.maximum(h, max_oc)
    l = np.minimum(l, min_oc)
    h = np.maximum(h, l)  # high >= low

    x[:, idx["open"]] = o
    x[:, idx["high"]] = h
    x[:, idx["low"]] = l
    x[:, idx["close"]] = c
    return x
