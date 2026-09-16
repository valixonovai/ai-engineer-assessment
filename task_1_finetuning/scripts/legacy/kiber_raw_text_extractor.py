"""
kiber_raw_text_extractor.py
===========================
Kiberxavfsizlik PDF kitoblaridan tozalangan xom matn ajratib,
Qwen-7B-Base modelini Continuous Pre-training qilish uchun
JSON dataset yaratadi.

Chiqish formati:
[
  {
    "text": "Tozalangan kiberxavfsizlik matni...",
    "meta": {
      "source": "kitob_nomi.pdf",
      "chunk_id": 0
    }
  },
  ...
]

Foydalanish:
  python kiber_raw_text_extractor.py --pdf_dir "./books" --output_file "kiber_pretrain_dataset.json" --chunk_size 1000
"""

import os
import re
import json
import argparse
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[XATOLIK] PyMuPDF (fitz) kutubxonasi topilmadi!")
    print("  O'rnatish uchun: pip install pymupdf")
    exit(1)


# =====================================================================
# MATN TOZALASH FUNKSIYALARI
# =====================================================================

def clean_text(text):
    """
    PDF dan olingan matnni tozalaydi:
    - Sahifa raqamlarini olib tashlaydi
    - Juda qisqa qatorlarni olib tashlaydi
    - Header/footer takrorlarini tozalaydi
    - Keraksiz belgilarni tozalaydi
    - Ko'p bo'shliqlarni bittaga keltiradi
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Sahifa raqami (faqat raqamdan iborat qatorlar)
        if stripped.isdigit():
            continue

        # Juda qisqa qatorlar (1-2 harf, keraksiz)
        if len(stripped) < 3:
            continue

        # FAQAT bosh harflar va raqamlardan iborat juda qisqa headerlar
        # (masalan: "CHAPTER 1", "PART II" kabi — lekin uzunroqlarini qoldiramiz)
        if len(stripped) < 5 and stripped.isupper():
            continue

        # Ortiqcha bo'shliqlarni tozalash
        cleaned_line = re.sub(r'[ \t]+', ' ', stripped)

        cleaned_lines.append(cleaned_line)

    # Qatorlarni birlashtirish
    text = '\n'.join(cleaned_lines)

    # Ko'p ketma-ket yangi qatorlarni 2 taga keltirish
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Ko'p bo'shliqlarni bittaga keltirish (yangi qatordan tashqari)
    text = re.sub(r'[^\S\n]+', ' ', text)

    return text.strip()


def extract_pdf_text(pdf_path):
    """
    PDF fayldan barcha sahifalarni o'qib, birlashtirilgan matn qaytaradi.
    """
    book_name = os.path.basename(pdf_path)
    print(f"  [KITOB YUKLANMOQDA] {book_name} ...")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [XATOLIK] PDF o'qishda xato: {book_name} -> {e}")
        return ""

    all_pages_text = []
    page_count = len(doc)

    for page_num in range(page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        if page_text.strip():
            all_pages_text.append(page_text)

    doc.close()

    combined = "\n".join(all_pages_text)
    cleaned = clean_text(combined)

    word_count = len(cleaned.split())
    print(f"  [TAYYOR] {book_name}: {page_count} sahifa, ~{word_count} so'z ajratildi.")

    return cleaned


def chunk_text_by_words(text, chunk_size=1000, overlap=100):
    """
    Matnni so'zlar soni bo'yicha bo'laklarga ajratadi.
    Jumlalar chegarasida kesishga harakat qiladi.

    Args:
        text: Tozalangan matn
        chunk_size: Har bir bo'lakdagi maksimal so'zlar soni
        overlap: Ustma-ust tushish (so'zlar soni)

    Returns:
        List of text chunks
    """
    words = text.split()
    total_words = len(words)
    chunks = []

    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)

        # Agar oxiriga yetmagan bo'lsak, jumla chegarasida kesishga harakat qilamiz
        if end < total_words:
            # Oxirgi 30% oraliqda nuqta (. ! ?) qidiramiz
            search_start = max(start + int(chunk_size * 0.7), start + 1)
            search_region = ' '.join(words[search_start:end])

            # Oxirgi jumla oxirini topish
            last_sentence_end = -1
            for punct in ['. ', '.\n', '! ', '? ']:
                idx = search_region.rfind(punct)
                if idx > last_sentence_end:
                    last_sentence_end = idx

            if last_sentence_end > 0:
                # Jumla chegarasida kesish
                actual_end_words = len(search_region[:last_sentence_end + 1].split())
                end = search_start + actual_end_words

        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words).strip()

        # Bo'sh yoki juda qisqa chunklarni o'tkazib yuborish
        if len(chunk_text) > 50:
            chunks.append(chunk_text)

        # Keyingi boshlanish nuqtasi (overlap bilan)
        start = end - overlap
        if start >= total_words:
            break
        # Agar chunk juda kichik bo'lsa, to'xtash
        if end >= total_words:
            break

    return chunks


# =====================================================================
# ASOSIY DASTUR
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kiberxavfsizlik PDF kitoblaridan Pre-train Dataset yaratish"
    )
    parser.add_argument(
        "--pdf_dir", type=str, required=True,
        help="PDF kitoblar joylashgan papka manzili"
    )
    parser.add_argument(
        "--output_file", type=str, default="kiber_pretrain_dataset.json",
        help="Chiquvchi pre-train dataset fayli nomi (default: kiber_pretrain_dataset.json)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000,
        help="Har bir matn bo'lagidagi maksimal so'zlar soni (default: 1000)"
    )
    parser.add_argument(
        "--overlap", type=int, default=100,
        help="Bo'laklar orasidagi ustma-ust tushish so'zlar soni (default: 100)"
    )

    args = parser.parse_args()

    # === 1. Papkani tekshirish ===
    print("=" * 60)
    print("  KIBERXAVFSIZLIK PRE-TRAIN DATASET EXTRACTOR")
    print("=" * 60)
    print(f"\n  PDF papka:       {args.pdf_dir}")
    print(f"  Chiqish fayli:   {args.output_file}")
    print(f"  Chunk hajmi:     {args.chunk_size} so'z")
    print(f"  Overlap:         {args.overlap} so'z")
    print()

    if not os.path.isdir(args.pdf_dir):
        print(f"[XATOLIK] Papka topilmadi: {args.pdf_dir}")
        exit(1)

    # === 2. PDF fayllarni topish ===
    pdf_files = sorted([
        os.path.join(args.pdf_dir, f)
        for f in os.listdir(args.pdf_dir)
        if f.lower().endswith('.pdf')
    ])

    if not pdf_files:
        print("[XATOLIK] Papkada hech qanday PDF fayl topilmadi!")
        exit(1)

    print(f"[+] Jami PDF kitoblar topildi: {len(pdf_files)} ta\n")

    # === 3. Har bir kitobdan matn ajratish va chunklarga bo'lish ===
    dataset = []
    global_chunk_id = 0
    total_words_processed = 0
    books_processed = 0
    books_failed = 0

    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        book_name = os.path.basename(pdf_path)
        print(f"\n--- [{i}/{len(pdf_files)}] ---")

        # Matn ajratish
        full_text = extract_pdf_text(pdf_path)

        if not full_text or len(full_text) < 100:
            print(f"  [OGOHLANTIRISH] {book_name}: matn juda qisqa yoki bo'sh. O'tkazib yuborildi.")
            books_failed += 1
            continue

        # So'zlar soni
        word_count = len(full_text.split())
        total_words_processed += word_count

        # Chunklarga bo'lish
        chunks = chunk_text_by_words(
            full_text,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

        # Dataset ga qo'shish
        book_chunk_count = 0
        for chunk_text in chunks:
            entry = {
                "text": chunk_text,
                "meta": {
                    "source": book_name,
                    "chunk_id": global_chunk_id
                }
            }
            dataset.append(entry)
            global_chunk_id += 1
            book_chunk_count += 1

        print(f"  [{book_chunk_count}] ta pre-train bo'lagiga ajratildi.")
        books_processed += 1

    elapsed = time.time() - start_time

    # === 4. Natijani JSON faylga saqlash ===
    print(f"\n{'=' * 60}")
    print(f"  NATIJALAR SAQLANMOQDA...")
    print(f"{'=' * 60}")

    output_path = args.output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.pdf_dir, output_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # === 5. Yakuniy hisobot ===
    print(f"\n{'=' * 60}")
    print(f"  YAKUNIY HISOBOT")
    print(f"{'=' * 60}")
    print(f"  Ishlov berilgan kitoblar:    {books_processed} ta")
    print(f"  Xatolik bilan o'tgan:        {books_failed} ta")
    print(f"  Jami so'zlar (xom):          {total_words_processed:,} ta")
    print(f"  Jami pre-train bo'laklar:    {len(dataset):,} ta")
    print(f"  O'rtacha bo'lak hajmi:       ~{total_words_processed // max(len(dataset),1)} so'z")
    print(f"  Dataset fayl hajmi:          {file_size_mb:.2f} MB")
    print(f"  Saqlangan manzil:            {output_path}")
    print(f"  Sarflangan vaqt:             {elapsed:.1f} soniya")
    print(f"{'=' * 60}")
    print(f"\n  [✓] Pre-train dataset muvaffaqiyatli yaratildi!")
    print(f"  [✓] Qwen-7B-Base CPT uchun tayyor.\n")


if __name__ == "__main__":
    main()
