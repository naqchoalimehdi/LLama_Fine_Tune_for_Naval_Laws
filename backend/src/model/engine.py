
import os
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DefaultDataCollator
)
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login
from ..utils.config import Config

class ModelEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.setup_environment()

    def setup_environment(self):
        if Config.HF_TOKEN:
            login(Config.HF_TOKEN)
        else:
            print("Warning: No Hugging Face token provided.")

    def train(self, train_dataset, val_dataset):
        """Fine-tune the model."""
        print("Loading Base Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=Config.MAX_SEQ_LENGTH,
                padding="longest",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        print("Tokenizing datasets...")
        train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(Config.MODEL_DIR, Config.NEW_MODEL_NAME),
            num_train_epochs=Config.EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRAD_ACCUMULATION,
            learning_rate=Config.LEARNING_RATE,
            fp16=True,
            save_strategy="steps",
            save_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            logging_steps=10,
            warmup_steps=100,
            max_grad_norm=1.0,
            weight_decay=0.01,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            lr_scheduler_type="cosine",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=DefaultDataCollator(),
        )
        
        print("Starting Training...")
        trainer.train()
        
        print("Saving Model...")
        final_path = os.path.join(Config.MODEL_DIR, Config.NEW_MODEL_NAME)
        model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Model saved to {final_path}")
        
        # Google Drive Backup
        if Config.GOOGLE_DRIVE_PATH and os.path.exists(Config.GOOGLE_DRIVE_PATH):
            import shutil
            timestamp = os.path.basename(final_path) # or nice timestamp
            backup_dest = os.path.join(Config.GOOGLE_DRIVE_PATH, Config.NEW_MODEL_NAME)
            print(f"Backing up to Google Drive: {backup_dest}...")
            try:
                if os.path.exists(backup_dest):
                    shutil.rmtree(backup_dest)
                shutil.copytree(final_path, backup_dest)
                print("✓ Backup successful!")
            except Exception as e:
                print(f"❌ Backup failed: {e}")
        elif Config.GOOGLE_DRIVE_PATH:
            print(f"Warning: Google Drive path not found: {Config.GOOGLE_DRIVE_PATH}")

        return final_path

    def load_model_for_inference(self, model_path=None):
        """Load fine-tuned model for inference."""
        # Prioritize Google Drive if no specific path provided
        if not model_path:
            if Config.GOOGLE_DRIVE_PATH:
                drive_model_path = os.path.join(Config.GOOGLE_DRIVE_PATH, Config.NEW_MODEL_NAME)
                if os.path.exists(drive_model_path):
                    print(f"Found model on Google Drive. Loading from: {drive_model_path}")
                    model_path = drive_model_path
                else:
                    print(f"Model not found on Google Drive ({drive_model_path}). Falling back to local.")
                    
            if not model_path:
                model_path = os.path.join(Config.MODEL_DIR, Config.NEW_MODEL_NAME)
            
        print(f"Loading Fine-Tuned Model from {model_path}...")
        gc.collect()
        torch.cuda.empty_cache()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        print("Model loaded successfully.")

    def generate_response(self, question: str, temperature: float = 0.3, max_new_tokens: int = 250, top_p: float = 0.8, repetition_penalty: float = 1.2) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model_for_inference first.")

        test_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a legal expert on Royal Saudi Navy jurisdiction and maritime law. Provide accurate, concise answers based on the legal documents.<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=40,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()
        return answer
