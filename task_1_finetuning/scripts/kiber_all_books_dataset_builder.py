import os
import re
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[WARN] PyMuPDF (fitz) topilmadi. O'rnatish uchun: pip install pymupdf")
try:
    from openai import OpenAI
except ImportError:
    print("[WARN] openai kutubxonasi topilmadi. O'rnatish uchun: pip install openai")

# =====================================================================
# 1. TIZIM VA PROMPTLAR SOZLAMALARI
# =====================================================================

SYSTEM_PROMPT = """Siz kiberxavfsizlik va axborot xavfsizligi bo'yicha dunyodagi eng tajribali SOC (Security Operations Center) tahlilchisi va o'qituvchisiz. 
Sizga taqdim etilgan kiberxavfsizlik matni asosida, Qwen-7B-Base modelini mustaqil kiber-agent qilib tarbiyalash uchun 2 xil toifadagi yuqori sifatli suhbat zanjirini (SFT Dataset) yarating.

Har bir matn bo'lagi uchun quyidagilarni generatsiya qiling:
1. "RAG/Faktik Tahlil" suhbati: Matndagi tushunchalar, zaifliklar yoki tahlillar bo'yicha aniq savol va faqat matnga asoslangan javob.
2. "Agent/Tool Use (Asbob ishlatish)" suhbati: Model mustaqil agent sifatida qachon o'z kuchi yetmasligini (masalan, real vaqtda port skanerlash, IP tekshirish, exploit ishga tushirish) anglab, tizimdagi asboblarni (masalan, `nmap_scan`, `sqlmap_check`, `check_vulnerability`, `block_ip`) qanday chaqirishini ko'rsatuvchi ReAct (Thought -> Action/Call -> Observation/Tool -> Final Answer) formatidagi mukammal ssenariy.

FORMAT QOIDASI:
Ssenariylarda quyidagi maxsus XML teglardan foydalaning:
- Ichki fikrlash uchun: <thought>...</thought>
- Asbobni chaqirish uchun: <call:tool_name>{"param": "val"}</call:tool_name>
- Yakuniy javob uchun toza matn.

Natijani FAQAT quyidagi JSON formatida qaytaring, ortiqcha tushuntirish yoki markdown teglari (masalan, ```json) qo'shmang:
[
  {
    "messages": [
      {"role": "system", "content": "Siz kiberxavfsizlik bo'yicha professional maslahatchisiz. Foydalanuvchi savollariga faqat taqdim etilgan faktlar doirasida aniq javob bering."},
      {"role": "user", "content": "Savol..."},
      {"role": "assistant", "content": "Javob..."}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "Siz kiberxavfsizlik bo'yicha SOC avtomatlashtirilgan tahlilchisisiz. Sizda tarmoqni tekshirish uchun `nmap_scan` asbobi mavjud."},
      {"role": "user", "content": "Muammo tavsifi..."},
      {"role": "assistant", "content": "<thought>Fikrlash...</thought>\\n<call:nmap_scan>{\\\"target\\\": \\\"10.0.0.1\\\"}</call:nmap_scan>"},
      {"role": "tool", "name": "nmap_scan", "content": "{\\\"status\\\": \\\"open\\\", \\\"ports\\\": [22, 80]}"},
      {"role": "assistant", "content": "<thought>Tahlil...</thought>\\nTizimda quyidagi portlar ochiq..."}
    ]
  }
]"""

# =====================================================================
# 2. PDF MATNNI TOZALASH VA BO'LAKLASH (CHUNKING)
# =====================================================================

def clean_text(text):
    """Matndagi ortiqcha keraksiz elementlarni, header/footerlarni tozalaydi"""
    # Sahifa raqamlarini tozalash (faqat raqamdan iborat qatorlar)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Sahifa raqami yoki takrorlanuvchi kitob nomlarini o'chirish
        if stripped.isdigit() or len(stripped) < 3:
            continue
        # Kiberxavfsizlik kitoblaridagi ko'p uchraydigan keraksiz belgilarni tozalash
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    # Bir nechta ketma-ket bo'shliqlarni bittaga keltirish
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_pdf_chunks(pdf_path, chunk_size=4000, overlap=600):
    """
    PDF faylidan matnni ajratib, belgilangan hajmda va overlap 
    (ustma-ust tushish) bilan bo'laklarga ajratadi.
    """
    print(f"[+] Yuklanmoqda: {os.path.basename(pdf_path)}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[-] PDF o'qishda xatolik {pdf_path}: {e}")
        return []

    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        full_text.append(text)
        
    combined_text = "\n".join(full_text)
    cleaned_combined = clean_text(combined_text)
    
    # Bo'laklarga ajratish (character-based sliding window)
    chunks = []
    start = 0
    text_len = len(cleaned_combined)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = cleaned_combined[start:end]
        
        # Jumlalar oxirida chiroyli kesishga harakat qilamiz
        if end < text_len:
            last_period = chunk.rfind('.')
            if last_period != -1 and last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = cleaned_combined[start:end]
                
        chunks.append({
            "source": os.path.basename(pdf_path),
            "content": chunk
        })
        start += (len(chunk) - overlap)
        if len(chunk) <= overlap:
            break
            
    print(f"[√] Muvaffaqiyatli bo'lindi: {len(chunks)} ta sifatli bo'lak hosil qilindi.")
    return chunks

# =====================================================================
# 3. LLM API ORQALI DATASET GENERATSIYASI
# =====================================================================

def call_llm_api(client, model_name, chunk_content):
    """O'rgatuvchi LLM modeliga murojaat qilib dataset ssenariysini oladi"""
    user_prompt = f"Mana kiberxavfsizlik kitobidan parcha:\n---\n{chunk_content}\n---\nUshbu matn asosida talab qilingan JSON datasetni yarating."
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Aniqlik uchun past harorat
            response_format={"type": "json_object"}
        )
        raw_json = response.choices[0].message.content.strip()
        # JSON parsingni tekshirish
        data = json.loads(raw_json)
        return data
    except Exception as e:
        print(f"[-] API xatoligi yoki JSON parsing xatosi: {e}")
        return None

# =====================================================================
# 4. ASOSIY PYTHONE ISHLASH KETMA-KETLIGI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Kiberxavfsizlik Kitoblaridan Agent Dataset Generator")
    parser.add_argument("--pdf_dir", type=str, required=True, help="PDF kitoblar joylashgan papka manzili")
    parser.add_argument("--output_file", type=str, default="kiber_agent_dataset.json", help="Chiquvchi dataset fayli nomi")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI/Qwen/DeepSeek API kalitingiz")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1", help="API asosi (masalan, Qwen/DeepSeek API manzillari)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="O'rgatuvchi model nomi")
    parser.add_argument("--workers", type=int, default=3, help="Parallel ishlovchi oqimlar (threads) soni")
    
    args = parser.parse_args()
    
    # Papkadagi barcha PDF fayllarni topish
    if not os.path.exists(args.pdf_dir):
        print(f"[-] Xatolik: {args.pdf_dir} papkasi topilmadi!")
        return
        
    pdf_files = [os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("[-] Papkada hech qanday PDF kitob topilmadi.")
        return
        
    print(f"[+] Jami topilgan PDF kitoblar soni: {len(pdf_files)}")
    
    # Barcha kitoblardan chunklarni yig'ish
    all_chunks = []
    for pdf_path in pdf_files:
        chunks = extract_pdf_chunks(pdf_path)
        all_chunks.extend(chunks)
        
    print(f"\n[+] Jami generatsiya qilinishi kerak bo'lgan bo'laklar soni: {len(all_chunks)}")
    print("[+] OpenAI Client sozlanmoqda...")
    
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    
    final_dataset = []
    processed_count = 0
    
    # Resume (ishni to'xtagan joyidan davom ettirish) imkoniyati
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                final_dataset = json.load(f)
            print(f"[+] Oldingi yuklangan dataset topildi. Unda {len(final_dataset)} ta namuna bor.")
        except:
            pass

    print(f"[+] Parallel oqimlar soni: {args.workers}. Dataset generatsiyasi boshlandi...\n")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Har bir chunk uchun API chaqiruv topshiriqlarini yaratamiz
        future_to_chunk = {
            executor.submit(call_llm_api, client, args.model, chunk['content']): chunk 
            for chunk in all_chunks
        }
        
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            processed_count += 1
            try:
                result = future.result()
                if result:
                    # Agar API to'g'ridan-to'g'ri ro'yxat (list) qaytargan bo'lsa
                    if isinstance(result, list):
                        final_dataset.extend(result)
                    elif isinstance(result, dict) and "messages" in result:
                        final_dataset.append(result)
                    else:
                        # Ba'zan kalit ostida qaytarishi mumkin
                        for key in result:
                            if isinstance(result[key], list):
                                final_dataset.extend(result[key])
                                break
                    
                    # Har bir muvaffaqiyatli qadamda zaxira nusxa saqlab boriladi
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
                        
                print(f"[{processed_count}/{len(all_chunks)}] Ishlov berildi. Jami yig'ilgan dataset hajmi: {len(final_dataset)} ta muloqot.")
            except Exception as exc:
                print(f"[-] Chunk ishlovida kutilmagan xatolik: {exc}")
                
            # API cheklovlariga tushmaslik uchun qisqa tanaffus
            time.sleep(0.5)

    print(f"\n[√] TABRIKLAYMAN! Butun kitoblar muvaffaqiyatli datasetga aylantirildi.")
    print(f"[√] Yakuniy dataset saqlangan fayl: {args.output_file}")
    print(f"[√] Jami SFT va Agent namunalari soni: {len(final_dataset)}")

if __name__ == "__main__":
    main()
