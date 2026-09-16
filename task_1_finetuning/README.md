# Task 1 — Kiberxavfsizlik LLM Fine-Tuning (QLoRA / LoRA)

## Loyihaning maqsadi
Kichik open-source LLM (Qwen-2.5-7B-Base) kiberxavfsizlik (SOC agent) yo'nalishida fine-tune qilingan.

## Strukturasi
```
dataset/  — kiber_agent_dataset.json (200+ chat misol)
training/ — dataset builder, raw text extractor, SFT notebook (json)
inference/ — (sizning video/test natijangiz bu yerga qo'shiladi)
```

## Dataset
`dataset/kiber_agent_dataset.json` — ChatML formatidagi 200+ misol. Har bir misolda system/user/assistant/tool rollari mavjud. Manba: `books_for_agent` ichidagi PDF kitoblardan generatsiya qilingan.

## Training
`training/` papkasida:
- `kiber_dataset_builder.py` — PDF dan matn ajratish va chunklash
- `kiber_all_books_dataset_builder.py` — AI yordamida dataset generatsiya qilish scripti
- `kiber_sft_train_kaggle.json` — SFT notebookning JSON eksporti (Unsloth, 4-bit QLoRA)

Fine-tuning usuli: LoRA / QLoRA (4-bit quantization). Adapter saqlash va keyin yuklash mumkin.

## Model ishlatish
Colab link -->  https://colab.research.google.com/notebook#fileId=https%3A//huggingface.co/valixonov04/qwen-7b-kiberagent-full.ipynb

Kaggle link -- > https://www.kaggle.com/code/valixonovilyosbek/notebook5359b1a2aa


huggingface Transformer : 
```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("valixonov04/qwen-7b-kiberagent-full")
model = AutoModelForCausalLM.from_pretrained("valixonov04/qwen-7b-kiberagent-full", device_map="auto")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```
