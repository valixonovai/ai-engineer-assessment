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
1. Datasetni yuklash: `dataset/kiber_agent_dataset.json`
2. Training: notebook yoki `training/*.py` orqali
3. Inference: adapter bilan yuklangan model orqali savollarga javob

## Eslatma
- `books_for_agent` ichidagi PDF kitoblar (Bug Bounty, CEH, Hacking APIs) manba sifatida ishlatilgan.
- Kodlarda ortiqcha AI kommentlari tozalangan, nomlar professional holatga keltirilgan.
