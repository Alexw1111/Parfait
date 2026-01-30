import pandas as pd
import numpy as np
from ..quant_tools import estimators

def generate_labels_for_window(future_df: pd.DataFrame) -> dict:
    # 优先使用复权收盘价
    if 'close_qfq' in future_df.columns:
        future_prices = future_df['close_qfq'].values
    else:
        # 兼容没有复权列的情况
        future_prices = future_df['close'].values

    # 如果窗口长度太短，或者价格数据几乎全是0，则没有计算意义，提前返回空字典
    if len(future_prices) < 5 or np.count_nonzero(future_prices) < 2:
        return {}
        
    # 筛选出非零的价格点进行计算，避免 log(0)
    non_zero_prices = future_prices[future_prices > 0]
    
    # 如果筛选后有效价格点太少，也无法计算收益率
    if len(non_zero_prices) < 2:
        return {}

    # 使用筛选后的非零价格计算对数收益率
    # 不需要再加 epsilon，因为已经排除了0
    with np.errstate(divide='ignore', invalid='ignore'):
        future_log_ret = np.diff(np.log(non_zero_prices))
    
    future_log_ret = future_log_ret[np.isfinite(future_log_ret)]
    
    if len(future_log_ret) < 2:
        return {}

    drift = np.mean(future_log_ret) * 252
    volatility = estimators.calculate_realized_volatility(future_log_ret)
    # Hurst指数计算本身对0不敏感，但对常数序列敏感。我们已在开头排除了这种情况。
    hurst = estimators.calculate_hurst_exponent(future_prices)
    jump_intensity, jump_mean = estimators.detect_jumps_bipower(future_log_ret)
    garch_params = estimators.fit_garch_params(future_log_ret)
    peak = np.maximum.accumulate(future_prices)
    # 当 peak 值为 0 时，我们认为回撤也是 0，而不是 nan
    divisor_peak = np.where(peak == 0, 1.0, peak) # 将0替换为1，避免除以0
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
