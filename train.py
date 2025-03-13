import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

# Authenticate to Hugging Face (replace with your real token)
login(token="YOUR_HF_TOKEN", add_to_git_credential=True)

# Load dataset
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")

# Define model and processor
model_id = "Qwen/Qwen2-VL-7B-Instruct"

# BitsAndBytes (4-bit quantization) configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load pre-trained model & processor
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Define system prompt and data formatting
system_message = "You are an assistant generating product descriptions optimized for SEO."

def format_data(sample):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": f"Create a short description for {sample['product_name']}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["description"]}]}
        ]
    }

formatted_dataset = dataset.map(format_data)

# Set up PEFT LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Training arguments configuration
training_args = TrainingArguments(
    output_dir="./qwen2-finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs",
    save_strategy="epoch",
    bf16=True,
    push_to_hub=True,
    report_to="tensorboard",
    save_total_limit=2,
    remove_unused_columns=False
)

# Custom data collator to handle multimodal inputs
def collate_fn(samples):
    texts = [processor.apply_chat_template(sample["messages"], tokenize=False) for sample in samples]
    images = [process_vision_info(sample) for sample in samples]

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
    peft_config=peft_config
)

# Start training
trainer.train()

# Save model and processor locally
model.save_pretrained("qwen2-finetuned")
processor.save_pretrained("qwen2-finetuned")

# Push to Hugging Face hub
trainer.push_to_hub()
