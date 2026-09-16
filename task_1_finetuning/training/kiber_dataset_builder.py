import os
import re
import json
import argparse
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None


class KiberDatasetBuilder:
    def __init__(self, overlap_percent: float = 0.15, chunk_size_words: int = 600):
        self.overlap_percent = overlap_percent
        self.chunk_size_words = chunk_size_words

    def extract_text_from_pdf(self, pdf_path: str, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        PDF faylidan sahifalar bo'yicha matnni ajratib oladi va tozalaydi.
        """
        pages_data = []
        
        if fitz:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # 1-indexed sahifalarni 0-indexedga o'tkazamiz
            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page)
            
            for page_num in range(start_idx, end_idx):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                cleaned_text = self.clean_text(text)
                
                if cleaned_text.strip():
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": cleaned_text
                    })
            doc.close()
        elif pypdf:
            reader = pypdf.PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page)
            
            for page_num in range(start_idx, end_idx):
                page = reader.pages[page_num]
                text = page.extract_text()
                cleaned_text = self.clean_text(text)
                
                if cleaned_text.strip():
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": cleaned_text
                    })
        else:
            raise ImportError("Iltimos 'PyMuPDF' (fitz) yoki 'pypdf' kutubxonasini o'rnating.")
            
        return pages_data

    def clean_text(self, text: str) -> str:
        """
        Matndagi keraksiz elementlarni (header, footer, ortiqcha bo'shliqlar) tozalaydi.
        """
        # Satrlarga ajratamiz
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line_str = line.strip()
            
            # Sahifa raqamlarini o'chirish (faqat raqamdan iborat satrlar)
            if line_str.isdigit():
                continue
                
            # Kiberxavfsizlik kitoblarida ko'p uchraydigan doimiy sarlavhalarni tozalash (regex)
            # Masalan: "Chapter X", "Page X", "www.it-ebooks.info" kabi reklama havolalari
            if re.search(r'(chapter\s+\d+|page\s+\d+|www\.)', line_str, re.IGNORECASE):
                continue
                
            cleaned_lines.append(line)
            
        cleaned_text = "\n".join(cleaned_lines)
        
        # Ortiqcha ketma-ket kelgan yangi qatorlarni va bo'shliqlarni bittaga keltirish
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()

    def semantic_chunking(self, pages_data: List[Dict[str, Any]], pdf_name: str) -> List[Dict[str, Any]]:
        """
        Tozalangan matnlarni to'liq kontekstni saqlagan holda o'zaro ustma-ust tushish (overlap)
        chegarasi bilan semantik bo'laklarga ajratadi.
        """
        # Butun kitob matnini sahifa metama'lumoti bilan birlashtiramiz
        full_word_stream = []
        for page in pages_data:
            words = page["text"].split()
            for word in words:
                full_word_stream.append((word, page["page_number"]))
                
        chunks = []
        total_words = len(full_word_stream)
        step = int(self.chunk_size_words * (1 - self.overlap_percent))
        
        if step <= 0:
            step = self.chunk_size_words
            
        chunk_idx = 0
        for i in range(0, total_words, step):
            word_slice = full_word_stream[i : i + self.chunk_size_words]
            if not word_slice:
                continue
                
            chunk_text = " ".join([w[0] for w in word_slice])
            start_page = word_slice[0][1]
            end_page = word_slice[-1][1]
            
            chunks.append({
                "chunk_id": f"{pdf_name}_chunk_{chunk_idx}",
                "source_file": pdf_name,
                "pages": f"{start_page}-{end_page}" if start_page != end_page else str(start_page),
                "content": chunk_text
            })
            chunk_idx += 1
            
            # Agar oxirgi bo'lak to'liq qamrab olingan bo'lsa, to'xtatamiz
            if i + self.chunk_size_words >= total_words:
                break
                
        return chunks

    def process_book(self, pdf_path: str, start_page: int, end_page: int, output_json_path: str):
        """
        Kitobni to'liq qayta ishlab, tayyor JSON fayliga yozadi.
        """
        pdf_name = os.path.basename(pdf_path)
        print(f"[*] {pdf_name} yuklanmoqda va tahlil qilinmoqda...")
        pages_data = self.extract_text_from_pdf(pdf_path, start_page, end_page)
        
        print(f"[*] {len(pages_data)} ta sahifa muvaffaqiyatli yuklandi.")
        chunks = self.semantic_chunking(pages_data, pdf_name)
        
        print(f"[+] Jami {len(chunks)} ta toza va to'liq kontekstli bo'laklar (chunks) yaratildi.")
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        print(f"[✔] Natija muvaffaqiyatli saqlandi: {output_json_path}")


if __name__ == "__main__":
    # Konsol orqali ishga tushirish uchun qulaylik yaratamiz
    parser = argparse.ArgumentParser(description="Kiberxavfsizlik kitoblarini dataset uchun qayta ishlash")
    parser.add_argument("--pdf", type=str, required=True, help="PDF faylga olib boruvchi yo'l")
    parser.add_argument("--start", type=int, default=1, help="Boshlang'ich sahifa")
    parser.add_argument("--end", type=int, default=9999, help="Yakuniy sahifa")
    parser.add_argument("--output", type=str, default="cleaned_chunks.json", help="Chiquvchi JSON fayl yo'li")
    
    args = parser.parse_args()
    
    builder = KiberDatasetBuilder()
    builder.process_book(args.pdf, args.start, args.end, args.output)
