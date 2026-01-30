import torch
import json
import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

class LLMAgent:
    def __init__(self, model_name_or_path: str, use_lora: bool = True, use_4bit: bool = True):
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        if use_lora:
            if os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
                print(f"Loading Fine-tuned LoRA weights from {model_name_or_path}...")
                self.model = PeftModel.from_pretrained(self.model, model_name_or_path)
            else:
                print("Initializing new LoRA adapter for training...")
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, lora_config)

    def generate_params(self, instruction: str) -> List[Dict]:
        self.model.eval()
        
        system_prompt = (
            "You are a sophisticated financial analyst. Your task is to analyze a natural language "
            "instruction about a market scenario and break it down into a list of mutually exclusive potential outcomes."
        )
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nInstruction: {instruction}\nJSON Output:<|im_end|>\n"
            f"<|im_start|>assistant\n```json\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        try:
            clean_json_str = response_text.replace("```json", "").replace("```", "").strip()
            scenarios = json.loads(clean_json_str)
            
            if not isinstance(scenarios, list):
                if isinstance(scenarios, dict):
                    scenarios = [scenarios]
                else:
                    raise ValueError("Output is not a list or dict")
            
            for s in scenarios:
                if 'probability' not in s: s['probability'] = 1.0 / len(scenarios)
                if 'params' not in s: s['params'] = {}
                
            return scenarios

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Critical Error parsing LLM output: {e}")
            print(f"Raw response was: {response_text}")
            return [{
                "scenario": "Error Recovery (Baseline)",
                "probability": 1.0,
                "params": {
                    "annualized_drift": 0.0,
                    "realized_volatility": 0.2,
                    "hurst_exponent": 0.5,
                    "jump_intensity": 0.0
                }
            }]

    def save_adapter(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Adapter saved to {path}")