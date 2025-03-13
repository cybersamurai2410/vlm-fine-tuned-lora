import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Define model paths (must match train.py)
base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
adapter_path = "qwen2-finetuned"
merged_model_path = "merged_qwen2_finetuned"

# Load the base model
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Merge LoRA adapter into the base model
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = peft_model.merge_and_unload()

# Save the fully merged model for faster inference
merged_model.save_pretrained(merged_model_path, safe_serialization=True, max_shard_size="2GB")

# Save the processor
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_model_path)

print(f"Merged model saved at '{merged_model_path}'")

# Reload the merged model for inference
model = AutoModelForVision2Seq.from_pretrained(
    merged_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(merged_model_path)

# Function to generate descriptions
def generate_description(sample, model, processor):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a product description AI."}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": f"Create a description for {sample['product_name']} in category {sample['category']}."}
        ]},
    ]
    # Prepare input for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate the product description
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text[0]

# Sample input for inference
sample = {
    "product_name": "Hasbro Marvel Avengers Iron Man Action Figure",
    "category": "Toys & Games",
    "image": "https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg"
}

# Generate descriptions from the merged fine-tuned model
fine_tuned_description = generate_description(sample, model, processor)
print("Fine-tuned Model Description:", fine_tuned_description)
