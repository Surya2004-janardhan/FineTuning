# FineTuning

## Hugging_Face : https://huggingface.co/sureal01/distilgpt2-code-generator/
# 🚀 AI Code Generator (Explanation → JavaScript Code)

This project fine-tunes **DistilGPT-2** to **generate JavaScript code** from natural language explanations.  
It uses **LoRA for efficient fine-tuning**, runs on **Google Colab (GPU-based)**, and supports **inference & deployment on Hugging Face**.

---

## **📌 What This Project Does**
✅ Fine-tunes **DistilGPT-2** to convert explanations into JavaScript code.  
✅ Uses **LoRA (Low-Rank Adaptation) for memory-efficient training**.  
✅ Supports **training on Google Colab (GPU optimized)**.  
✅ Provides **a script to test the fine-tuned model**.  
✅ Allows **uploading to Hugging Face for sharing & deployment**.

---

## **📊 Dataset Format**
This project trains on a JSON dataset structured as:
```json
[
    {
        "explanation": "This function takes a name as input and returns a greeting message.",
        "code": "function greet(name) { return 'Hello, ' + name + '!'; }"
    }
]
