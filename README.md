# Fine-Tuning VLM with LoRA for E-commerce Product Descriptions 
Applied supervised fine-tuning with LoRA adapters for 7B parameter large multimodal model or VLM to generate e-commerce product descriptions from images by training the multimodal projector and text decoder while freezing the image encoder; achieved faster training with reduced memory usage while maintaining inference speed with 80% accuracy

## Inference

**Input:**
```python
sample = {
    "product_name": "Hasbro Marvel Avengers Iron Man Action Figure",
    "category": "Toys & Games",
    "image": "https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg"
}
```

**Ouptut:**
```
Unleash the power of Iron Man with this 30.5 cm Hasbro Marvel Avengers Titan Hero Action Figure! This highly detailed Iron Man figure is perfect for collectors and kids alike. Features a realistic design and articulated joints for dynamic poses. A must-have for any Marvel fan's collection!
```
