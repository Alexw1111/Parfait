import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

from ..models.llm_agent import LLMAgent
from ..models.ts_backbone import TransformerEncoder
from ..models.diffusion_unet import ConditionalUNet1D
from ..quant_tools.sde_solver import generate_sde_path, normalize_path
from ..data_processing.normalizer import (
    get_feature_indices,
    normalize_history,
    denormalize_future,
    enforce_ohlc_constraints,
)


def get_ddpm_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """Generate DDPM variance schedule (beta_t, alpha_bar_t, etc.)."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "posterior_variance": betas
        * (1.0 - torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]))
        / (1.0 - alphas_cumprod),
    }


class InferencePipeline:
    """文本指令 -> 场景参数 -> guide -> OHLCV。"""

    def __init__(
        self,
        llm_agent: LLMAgent,
        ts_backbone: TransformerEncoder,
        diffusion_model: ConditionalUNet1D,
        config: Dict,
        device: Optional[str] = None,
    ):
        self.llm = llm_agent
        self.backbone = ts_backbone
        self.diffusion = diffusion_model
        self.config = config

        if device is None:
            device = config.get("training", {}).get("device", "cuda")
        if "cuda" in str(device) and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # LLM 使用 device_map=auto，避免强制 .to()
        self.backbone.to(self.device)
        self.diffusion.to(self.device)

        self.num_timesteps = 1000
        self.schedule = {k: v.to(self.device) for k, v in get_ddpm_schedule(self.num_timesteps).items()}

        self.features = self.config["data"]["features"]
        self.idx = get_feature_indices(self.features)

    def run(self, instruction: str, history_data: np.ndarray, total_simulations: int = 100):
        """生成未来 OHLCV（与 loader.py 同尺度）。"""
        self.backbone.eval()
        self.diffusion.eval()

        # LLM 场景
        print("Step 1: LLM generating multi-modal scenarios...")
        scenarios = self.llm.generate_params(instruction)
        print(f"Generated {len(scenarios)} scenarios.")
        for s in scenarios:
            print(f"  - Scenario '{s.get('scenario', 'N/A')}': Probability {s.get('probability', 0):.2f}")

        # 归一化 history
        history_raw = np.asarray(history_data, dtype=np.float32)

        # volume 过大则转 log1p
        vol_idx = self.idx["volume"]
        if np.isfinite(history_raw[:, vol_idx]).any() and np.nanmax(history_raw[:, vol_idx]) > 1000:
            history_raw[:, vol_idx] = np.log1p(np.maximum(history_raw[:, vol_idx], 0.0))

        history_norm, stats = normalize_history(history_raw, self.features)
        initial_price = float(stats.price_scale)

        with torch.no_grad():
            # history context
            history_tensor = torch.from_numpy(history_norm).float().unsqueeze(0).to(self.device)
            base_style_vector = self.backbone(history_tensor)

            # 生成 guide
            horizon = int(self.config["data"]["future_horizon"])
            all_guides: List[np.ndarray] = []
            all_styles: List[torch.Tensor] = []

            print("Step 2: Preparing guide curves and context for all scenarios...")
            for scenario in scenarios:
                prob = float(scenario.get("probability", 0.0))
                prob = max(0.0, min(1.0, prob))
                sde_params = scenario.get("params", {}) or {}

                num_paths = max(1, int(round(prob * total_simulations)))
                for _ in range(num_paths):
                    guide_path = generate_sde_path(sde_params, initial_price, horizon)
                    all_guides.append(normalize_path(guide_path))
                    all_styles.append(base_style_vector)

            if not all_guides:
                print("No valid scenarios generated. Aborting.")
                return None

            num_total_paths = len(all_guides)
            guide_tensor = torch.from_numpy(np.array(all_guides)).float().unsqueeze(1).to(self.device)  # [B,1,T]
            style_tensor = torch.cat(all_styles, dim=0).to(self.device)  # [B,C]

            # 采样
            print(f"Step 3: Starting parallel diffusion sampling for {num_total_paths} paths...")
            shape = (num_total_paths, len(self.features), horizon)
            sample = torch.randn(shape, device=self.device)

            for t in tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling"):
                time_tensor = torch.full((num_total_paths,), t, device=self.device, dtype=torch.long)
                predicted_noise = self.diffusion(sample, time_tensor, style_tensor, guide_tensor)

                beta_t = self.schedule["betas"][t]
                sqrt_one_minus_alpha_t_cumprod = self.schedule["sqrt_one_minus_alphas_cumprod"][t]

                model_mean = (1.0 / torch.sqrt(1.0 - beta_t)) * (
                    sample - beta_t * predicted_noise / (sqrt_one_minus_alpha_t_cumprod + 1e-8)
                )

                if t > 0:
                    posterior_variance = self.schedule["posterior_variance"][t]
                    noise = torch.randn_like(sample)
                    sample = model_mean + torch.sqrt(posterior_variance) * noise
                else:
                    sample = model_mean

        print("Step 4: Denormalize and enforce OHLC constraints...")
        simulations_norm = sample.cpu().numpy()  # [B, F, T]

        simulations = []
        for i in range(simulations_norm.shape[0]):
            path_tf = simulations_norm[i].transpose(1, 0)  # [T, F]
            path_denorm = denormalize_future(path_tf, stats, self.features)
            path_denorm = enforce_ohlc_constraints(path_denorm, self.features)
            simulations.append(path_denorm.transpose(1, 0))  # [F, T]

        simulations = np.stack(simulations, axis=0).astype(np.float32)
        return simulations  # [num_paths, Features, Horizon]
