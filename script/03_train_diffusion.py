import torch
import torch.nn as nn
import yaml
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ts_backbone import TransformerEncoder
from src.models.diffusion_unet import ConditionalUNet1D
from src.quant_tools.sde_solver import generate_sde_path, normalize_path
from src.data_processing.normalizer import normalize_history_future


class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_cumprod = alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[timesteps]).sqrt()

        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise


class DiffusionDataset(Dataset):
    def __init__(self, data_list, features_idx):
        self.data = data_list
        self.features_idx = features_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 训练/推理一致的 guide 与归一化

        history_raw = item["history"]     # [H, F]
        future_raw = item["future_gt"]    # [T, F]
        labels = item.get("labels", {})   # dict of SDE params estimated from future

        hist_n, fut_n, stats = normalize_history_future(history_raw, future_raw, self.features_idx)

        # history: [H, F] -> [F, H]
        history = torch.tensor(hist_n).permute(1, 0)
        # future:  [T, F] -> [F, T]
        future = torch.tensor(fut_n).permute(1, 0)

        horizon = fut_n.shape[0]
        initial_price = float(stats.price_scale)

        guide_path = generate_sde_path(labels, initial_price, horizon)
        guide = torch.tensor(normalize_path(guide_path), dtype=torch.float32).unsqueeze(0)  # [1, T]

        return {"history": history, "future": future, "guide": guide}


def train():
    # 配置相对仓库根目录
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["training"]["seed"])

    os.makedirs(config["diffusion"]["output_dir"], exist_ok=True)

    data_path = os.path.join(repo_root, config["data"]["processed_path"], "dataset_v1.pkl")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Run script/01_generate_labels.py first.")
        return

    raw_data_list = pd.read_pickle(data_path)
    dataset = DiffusionDataset(raw_data_list, config["data"]["features"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    if len(dataloader) == 0:
        print("Error: DataLoader has 0 batches (empty dataset). Check data/raw and re-run script/01_generate_labels.py")
        return

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    backbone = TransformerEncoder(
        input_dim=config["data"]["features_num"],
        model_dim=config["diffusion"]["context_dim"],
        num_heads=8,
        num_layers=3,
    ).to(device)

    diffusion_model = ConditionalUNet1D(
        input_channels=config["data"]["features_num"],
        guide_channels=1,
        model_channels=config["diffusion"]["model_channels"],
        context_dim=config["diffusion"]["context_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(diffusion_model.parameters()),
        lr=float(config["diffusion"]["learning_rate"]),
        weight_decay=1e-5,
    )

    criterion = nn.MSELoss()

    print(f"Starting training on {device}...")

    for epoch in range(config["diffusion"]["num_epochs"]):
        backbone.train()
        diffusion_model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()

            # [B, F, H] -> [B, H, F]
            hist = batch["history"].permute(0, 2, 1).to(device)
            future = batch["future"].to(device)  # [B, F, T]
            guide = batch["guide"].to(device)    # [B, 1, T]

            style_context = backbone(hist)       # [B, context_dim]

            noise = torch.randn_like(future)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (future.shape[0],), device=device
            ).long()

            noisy_future = scheduler.add_noise(future, noise, timesteps)
            noise_pred = diffusion_model(noisy_future, timesteps, style_context, guide)

            loss = criterion(noise_pred, noise)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(diffusion_model.parameters()),
                1.0,
            )
            optimizer.step()

            epoch_loss += float(loss.item())
            progress_bar.set_postfix({"loss": float(loss.item())})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(backbone.state_dict(), os.path.join(config["diffusion"]["output_dir"], f"backbone_ep{epoch}.pt"))
            torch.save(diffusion_model.state_dict(), os.path.join(config["diffusion"]["output_dir"], f"unet_ep{epoch}.pt"))

    torch.save(backbone.state_dict(), os.path.join(config["diffusion"]["output_dir"], "backbone_final.pt"))
    torch.save(diffusion_model.state_dict(), os.path.join(config["diffusion"]["output_dir"], "unet_final.pt"))
    print("Training Completed.")


if __name__ == "__main__":
    train()
