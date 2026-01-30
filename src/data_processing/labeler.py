import pandas as pd
import numpy as np
from ..quant_tools import estimators 

def generate_labels_for_window(future_df: pd.DataFrame) -> dict:
    if 'close_qfq' in future_df.columns:
        future_prices = future_df['close_qfq'].values
    else:
        future_prices = future_df['close'].values

    if len(future_prices) < 5 or np.count_nonzero(future_prices) < 2:
        return {}
        
    non_zero_prices = future_prices[future_prices > 0]
    
    if len(non_zero_prices) < 2:
        return {}

    with np.errstate(divide='ignore', invalid='ignore'):
        future_log_ret = np.diff(np.log(non_zero_prices))
    
    future_log_ret = future_log_ret[np.isfinite(future_log_ret)]
    
    if len(future_log_ret) < 2:
        return {}

    drift = np.mean(future_log_ret) * 252
    volatility = estimators.calculate_realized_volatility(future_log_ret)
    hurst = estimators.calculate_hurst_exponent(non_zero_prices)
    jump_intensity, jump_mean, _jump_vol = estimators.detect_jumps_bipower(future_log_ret)
    garch_params = estimators.fit_garch_params(future_log_ret)
    
    peak = np.maximum.accumulate(future_prices)
    divisor_peak = np.where(peak == 0, 1.0, peak)
    drawdown = (peak - future_prices) / divisor_peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

    labels = {
        "annualized_drift": float(drift),
        "realized_volatility": float(volatility),
        "hurst_exponent": float(hurst),
        "jump_intensity": float(jump_intensity),
        "jump_mean": float(jump_mean),
        "max_drawdown": float(max_drawdown),
        **garch_params
    }
    
    for key, value in labels.items():
        if not np.isfinite(value):
            labels[key] = 0.0
            
    return labels
