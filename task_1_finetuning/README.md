# Task 1 — Kiberxavfsizlik LLM Fine-Tuning (QLoRA / LoRA)

## Maqsad

Kichik open-source LLM (**Qwen-2.5-7B-Base**) kiberxavfsizlik (SOC agent) yo'nalishida
fine-tune qilindi. Dataset RAG va Agent/Tool Use ko'rinishida tayyorlandi: model
`<thought>` va `<call:tool_name>` teglari bilan mulohaza yuritishni va tool chaqirishni o'rganadi.

## Tuzilma

```
task_1_finetuning/
├── README.md
├── dataset/
│   └── kiber_agent_dataset.json          # ChatML formatidagi SFT dataset (~15 MB)
├── notebooks/
│   └── kiber-sft-train-kaggle.ipynb      # Unsloth + 4-bit QLoRA training notebook
└── scripts/
    ├── kiber_all_books_dataset_builder.py    # PDF kitoblardan AI yordamida dataset yig'ish
    ├── kiber_dataset_builder_clean.py        # matnni ajratish va chunklash (yakuniy)
    ├── kiber_raw_text_extractor_clean.py     # PDF dan matn ajratish (yakuniy)
    └── legacy/                               # eski/draft versiyalar (arxiv uchun)
```

> `scripts/legacy/` dagi fayllar ish jarayonidagi oldingi versiyalar — yakuniy ishlar
> `scripts/` va `notebooks/` papkalarida.

## Dataset

`dataset/kiber_agent_dataset.json` — ChatML formatidagi misollar. Har bir misolda
`system` / `user` / `assistant` rollari va tool chaqiruvlari mavjud.
Manba: `books_for_agent` papkasidagi PDF kitoblar (`scripts/kiber_all_books_dataset_builder.py`
orqali generatsiya qilingan, resume va parallel ishlovni qo'llab-quvvatlaydi).

## Training

Fine-tuning usuli: **LoRA / QLoRA (4-bit quantization)** — Unsloth bilan, Kaggle muhitida.
Asosiy parametrlar: `max_seq_length=4096`, LoRA rank 16, target modullar
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

| Resurs | Havola |
|---|---|
| Training notebook (repo ichida) | `notebooks/kiber-sft-train-kaggle.ipynb` |
| Kaggle notebook | https://www.kaggle.com/code/valixonovilyosbek/notebook5359b1a2aa |
| Modal (train jarayoni) | https://modal.com/notebooks/talaba4077/main/nb-1qls1PSuJ03OVuu4bT1xt9 |
| Colab | https://colab.research.google.com/notebook#fileId=https%3A//huggingface.co/valixonov04/qwen-7b-kiberagent-full.ipynb |
| Model (Hugging Face) | https://huggingface.co/valixonov04/qwen-7b-kiberagent-full |

## Modeldan foydalanish

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("valixonov04/qwen-7b-kiberagent-full")
model = AutoModelForCausalLM.from_pretrained("valixonov04/qwen-7b-kiberagent-full", device_map="auto")

messages = [{"role": "user", "content": "Who are you?"}]
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

## Eslatma

Bu vazifa mustaqil: repo ildizidagi `task_2_clinic_chatbot` bilan bog'liq emas.