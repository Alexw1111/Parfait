import os
import sys
import yaml
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llm_agent import LLMAgent

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df_raw = pd.read_pickle(os.path.join(config['data']['processed_path'], "dataset_v1.pkl"))
    
    formatted_data = []
    for item in df_raw:
        labels_json = item['labels']
        instruction = f"Simulate a regime with drift {labels_json['annualized_drift']:.2f} and vol {labels_json['realized_volatility']:.2f}"
        
        target_output = [{"scenario": "observed", "probability": 1.0, "params": labels_json}]
        
        formatted_data.append({
            "instruction": instruction,
            "output": f"```json\n{target_output}\n```"
        })

    train_ds = Dataset.from_list(formatted_data)

    agent = LLMAgent(config['llm']['model_name'], use_lora=True)

    training_args = TrainingArguments(
        output_dir=config['llm']['output_dir'],
        per_device_train_batch_size=config['llm']['batch_size'],
        learning_rate=float(config['llm']['learning_rate']),
        num_train_epochs=config['llm']['num_epochs'],
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=False
    )

    trainer = SFTTrainer(
        model=agent.model,
        train_dataset=train_ds,
        dataset_text_field="instruction",
        max_seq_length=512,
        args=training_args,
    )

    trainer.train()
    agent.model.save_pretrained(config['llm']['output_dir'])

if __name__ == "__main__":
    main()