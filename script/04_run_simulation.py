import torch
import yaml
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llm_agent import LLMAgent
from src.models.ts_backbone import TransformerEncoder
from src.models.diffusion_unet import ConditionalUNet1D
from src.inference.pipeline import InferencePipeline

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = LLMAgent(config['llm']['model_name'], use_lora=True)

    feat_num = len(config['data']['features'])
    backbone = TransformerEncoder(input_dim=feat_num, model_dim=config['diffusion']['context_dim'], num_heads=8, num_layers=3)
    backbone.load_state_dict(torch.load(os.path.join(config['diffusion']['output_dir'], "backbone_final.pt")))
    
    diffusion = ConditionalUNet1D(
        input_channels=feat_num, guide_channels=1,
        model_channels=config['diffusion']['model_channels'],
        context_dim=config['diffusion']['context_dim']
    )
    diffusion.load_state_dict(torch.load(os.path.join(config['diffusion']['output_dir'], "unet_final.pt")))

    pipeline = InferencePipeline(agent, backbone, diffusion, config)

    raw_dir = config['data']['raw_path']
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {raw_dir}.")
        return
    sample_df = pd.read_csv(os.path.join(raw_dir, csv_files[0]))
    history_data = sample_df[config['data']['features']].iloc[:int(config['data']['history_window'])].values

    instruction = "如果现在央行意外加息，且市场出现恐慌性抛售，但随后有国家队入场救市"
    simulated_paths = pipeline.run(instruction, history_data, total_simulations=50)

    np.save("simulated_results.npy", simulated_paths)
    print(f"Successfully generated 50 paths based on instruction. Saved to simulated_results.npy")

if __name__ == "__main__":
    main()