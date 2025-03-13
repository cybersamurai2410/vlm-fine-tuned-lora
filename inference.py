import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
 
adapter_path = "./qwen2-7b-instruct-amazon-description"
 
# Load Model base model
model = AutoModelForVision2Seq.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(model_id)

from qwen_vl_utils import process_vision_info
 
# sample from amazon.com
sample = {
  "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
  "catergory": "Toys & Games | Toy Figures & Playsets | Action Figures",
  "image": "https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg"
}
 
# prepare message
messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": sample["image"],
            },
            {"type": "text", "text": prompt.format(product_name=sample["product_name"], category=sample["catergory"])},
        ],
    }
]
 
def generate_description(sample, model, processor):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image","image": sample["image"]},
            {"type": "text", "text": prompt.format(product_name=sample["product_name"], category=sample["catergory"])},
        ]},
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
 
# let's generate the description
base_description = generate_description(sample, model, processor)
print(base_description)
# you can disable the active adapter if you want to rerun it with
# model.disable_adapters()

model.load_adapter(adapter_path) # load the adapter and activate
 
ft_description = generate_description(sample, model, processor)
print(ft_description)

import pandas as pd
from IPython.display import display, HTML
 
def compare_generations(base_gen, ft_gen):
    # Create a DataFrame
    df = pd.DataFrame({
        'Base Generation': [base_gen],
        'Fine-tuned Generation': [ft_gen]
    })
    # Style the DataFrame
    styled_df = df.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap',
        'border': '1px solid black',
        'padding': '10px',
        'width': '250px',  # Set width to 150px
        'overflow-wrap': 'break-word'  # Allow words to break and wrap as needed
    })
    
    # Display the styled DataFrame
    display(HTML(styled_df.to_html()))
  
compare_generations(base_description, ft_description)

from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq
 
adapter_path = "./qwen2-7b-instruct-amazon-description"
base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
merged_path = "merged"
 
# Load Model base model
model = AutoModelForVision2Seq.from_pretrained(model_id, low_cpu_mem_usage=True)
 
# Path to save the merged model
 
# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, adapter_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_path,safe_serialization=True, max_shard_size="2GB")
 
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_path)

