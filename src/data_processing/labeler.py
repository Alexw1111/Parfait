import pandas as pd
import numpy as np
from .loader import load_stock_data, create_sliding_windows
from ..quant_tools import estimators

def generate_labels_for_window(history_df: pd.DataFrame, future_df: pd.DataFrame) -> dict:
    history_prices = history_df['close_qfq'].values
    future_prices = future_df['close_qfq'].values
    history_log_ret = np.diff(np.log(history_prices))
    future_log_ret = np.diff(np.log(future_prices))
    drift = np.mean(future_log_ret) * 252
    volatility = estimators.calculate_realized_volatility(future_log_ret)
    hurst = estimators.calculate_hurst_exponent(future_prices)
    jump_intensity, jump_mean, jump_vol = estimators.detect_jumps_bipower(future_log_ret)
    garch_params = estimators.fit_garch_params(future_log_ret)
    peak = np.maximum.accumulate(future_prices)
    drawdown = (peak - future_prices) / peak
    max_drawdown = np.max(drawdown)

    labels = {
        "annualized_drift": drift,
        "realized_volatility": volatility,
        "hurst_exponent": hurst,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_volatility": jump_vol,
        "max_drawdown": max_drawdown,
        **garch_params
    }
    
    for k, v in labels.items():
        if not np.isfinite(v):
            labels[k] = 0.0
            
    return labels