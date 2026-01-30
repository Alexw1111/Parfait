import numpy as np
import pandas as pd

# Optional dependency: `arch` is not always installed.
try:
    from arch import arch_model
except Exception:
    arch_model = None


def calculate_hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    计算 Hurst 指数。
    Hurst 指数 < 0.5: 均值回归
    Hurst 指数 = 0.5: 随机游走
    Hurst 指数 > 0.5: 趋势增强
    """
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]
    if series.size < 20:
        # 短序列上 Hurst 估计方差极大，直接回退到 0.5
        return 0.5

    # max_lag 不允许超过序列长度，否则会出现空切片
    max_lag = int(min(max_lag, max(10, series.size // 2)))
    lags = np.arange(2, max_lag, dtype=np.int64)

    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        if diff.size == 0:
            continue
        v = np.std(diff)
        if np.isfinite(v) and v > 0:
            tau.append(np.sqrt(v))

    if len(tau) < 2:
        return 0.5

    tau = np.asarray(tau)
    lags = lags[: tau.size]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = float(poly[0])
    if not np.isfinite(hurst):
        return 0.5
    return hurst


def calculate_realized_volatility(log_returns: np.ndarray) -> float:
    """
    计算年化已实现波动率。
    """
    return np.sqrt(np.sum(log_returns**2)) * np.sqrt(252 / len(log_returns))


def detect_jumps_bipower(log_returns: np.ndarray, threshold_multiplier: float = 3.0) -> tuple[float, float, float]:
    """
    使用简化的阈值法检测跳跃。
    更严谨的方法是 Bipower Variation，但这里为了快速实现，采用标准差阈值。
    """
    if len(log_returns) < 2:
        return 0.0, 0.0, 0.0
        
    std_dev = np.std(log_returns)
    threshold = threshold_multiplier * std_dev
    
    jumps = log_returns[np.abs(log_returns) > threshold]
    
    jump_intensity = len(jumps) / len(log_returns) if len(log_returns) > 0 else 0.0
    jump_mean = np.mean(jumps) if len(jumps) > 0 else 0.0
    jump_vol = np.std(jumps) if len(jumps) > 0 else 0.0
    
    return jump_intensity, jump_mean, jump_vol


def fit_garch_params(log_returns: np.ndarray) -> dict:
    """
    拟合 GARCH(1,1) 模型并返回其关键参数。
    GARCH 模型用于描述波动率聚类现象。
    """
    if np.var(log_returns) < 1e-12:
        return {'omega': 0, 'alpha': 0, 'beta': 0}
        
    if arch_model is None:
        return {'omega': 0, 'alpha': 0, 'beta': 0}

    garch_model = arch_model(log_returns * 100, vol='Garch', p=1, q=1, dist='Normal')
    
    try:
        res = garch_model.fit(disp='off')
        params = res.params
        return {
            'omega': params.get('omega', 0),
            'alpha': params.get('alpha[1]', 0),
            'beta': params.get('beta[1]', 0)
        }
    except Exception:
        return {'omega': 0, 'alpha': 0, 'beta': 0}
