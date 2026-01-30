import numpy as np
import torch


def generate_sde_path(
    sde_params: dict,
    initial_price: float,
    horizon: int,
    dt: float = 1/252
) -> np.ndarray:
    prices = np.zeros(horizon)
    prices[0] = initial_price
    
    mu = float(sde_params.get("annualized_drift", 0.0) or 0.0)
    sigma = float(sde_params.get("realized_volatility", 0.2) or 0.2)
    jump_intensity = float(sde_params.get("jump_intensity", 0.0) or 0.0)
    jump_mean = float(sde_params.get("jump_mean", 0.0) or 0.0)

    # sanitize (avoid NaNs blowing up the exp)
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sigma):
        sigma = 0.2
    sigma = abs(sigma)

    if not np.isfinite(jump_intensity):
        jump_intensity = 0.0
    jump_intensity = float(np.clip(jump_intensity, 0.0, 1.0))

    if not np.isfinite(jump_mean):
        jump_mean = 0.0

    for t in range(1, horizon):
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        drift = mu * dt

        jump = 0
        if np.random.rand() < jump_intensity:
            jump = jump_mean
        
        prices[t] = prices[t-1] * np.exp(drift - 0.5 * sigma**2 * dt + diffusion + jump)
        
    return prices


def normalize_path(path: np.ndarray) -> np.ndarray:
    return path / path[0]
