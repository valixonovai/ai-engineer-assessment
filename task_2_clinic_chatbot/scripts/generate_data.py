"""Klinika test ma'lumotlarini generatsiya qiladi.

Barcha ma'lumotlar SUN'IY (xayoliy) — "Shifo Med" klinikasi uchun o'ylab topilgan
test bazasi. Haqiqiy klinika, shifokor yoki narx bilan bog'liq emas.

Ishlatish:
    python scripts/generate_data.py
Natija:
    data/doctors.json, data/services.json, data/rules.json, data/faqs.json
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLINIC = {
    "id": "shifo_med",
    "name": "Shifo Med klinikasi",
    "legal_name": '"SHIFO MED SERVIS" MChJ',
    "address": "Toshkent sh., Chilonzor tumani, Bunyodkor shoh ko'chasi 12-uy",
    "landmark": "Chilonzor metro bekati, 'Algoritm' savdo markazi ro'parasi",
    "phone": "+998 71 200 45 45",
    "short_number": "1145",
    "emergency_phone": "103",
    "email": "info@shifomed.uz",
    "website": "https://shifomed.uz",
    "work_hours": {
        "weekdays": "Dushanba–Shanba, 08:00–20:00",
        "sunday": "Yakshanba 09:00–15:00 (faqat navbatchi shifokor va laboratoriya)",
        "lab": "Laboratoriya: Dushanba–Shanba 07:30–19:00",
        "holidays": "Rasmiy bayramlarda 09:00–14:00 navbatchi xizmat",
    },
    "parking": "2 qavatli bepul avtoturargoh, 60 o'rin",
    "languages": ["o'zbek", "rus", "ingliz (cheklangan)"],
    "payment": ["Naqd", "Plastik karta (UzCard, Humo, Visa, Mastercard)", "Bank o'tkazmasi (yuridik shaxslar)"],
    "insurance_partners": ["Uzbekinvest", "Alskom", "Gross Insurance"],
    "note": "Ma'lumotlar test/demo maqsadida generatsiya qilingan va xayoliydir.",
}

# --------------------------------------------------------------------------
# Shifokorlar
# --------------------------------------------------------------------------
DOCTORS = [
    {
        "id": "d01",
        "full_name": "Aziz Karimov",
        "specialty": "Kardiolog",
        "degree": "Oliy toifali shifokor, tibbiyot fanlari nomzodi",
        "experience_years": 18,
        "room": "204-xona, 2-qavat",
        "schedule": {"days": ["Dushanba", "Seshanba", "Chorshanba", "Juma"], "time": "09:00–14:00"},
        "consultation_price_uzs": 250000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2007), kardiologiya ordinaturasi (2009)",
        "about": "Yurak-qon tomir kasalliklari, arterial gipertenziya, yurak ishemik kasalligi va "
                 "yurak yetishmovchiligini diagnostika qilish va kuzatish bilan shug'ullanadi. "
                 "EKG, ExoKG va 24 soatlik Xolter monitoringi natijalarini tahlil qiladi.",
        "procedures": ["Konsultatsiya", "EKG tahlili", "ExoKG (yurak UZI)", "Xolter monitoringi"],
    },
    {
        "id": "d02",
        "full_name": "Nilufar Rahimova",
        "specialty": "Nevrolog",
        "degree": "Birinchi toifali shifokor",
        "experience_years": 12,
        "room": "206-xona, 2-qavat",
        "schedule": {"days": ["Dushanba", "Chorshanba", "Payshanba", "Shanba"], "time": "10:00–16:00"},
        "consultation_price_uzs": 230000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent pediatriya tibbiyot instituti (2013), nevrologiya ordinaturasi (2015)",
        "about": "Bosh og'rig'i, migren, uyqu buzilishi, bel va bo'yin umurtqasi og'riqlari, "
                 "nevropatiya va stress bilan bog'liq holatlarni kuzatadi.",
        "procedures": ["Konsultatsiya", "Migren diagnostikasi", "Uyqu sifatini baholash"],
    },
    {
        "id": "d03",
        "full_name": "Bekzod Yusupov",
        "specialty": "Stomatolog-terapevt",
        "degree": "Oliy toifali shifokor",
        "experience_years": 15,
        "room": "108-xona, 1-qavat",
        "schedule": {"days": ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba"], "time": "08:00–13:00"},
        "consultation_price_uzs": 180000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus", "ingliz"],
        "education": "Toshkent davlat stomatologiya instituti (2010)",
        "about": "Kariesni davolash, kanal davolash, tish tozalash va estetik plombalash. "
                 "Og'riqsiz davolash (sedatsiya) bo'yicha malaka oshirgan.",
        "procedures": ["Konsultatsiya", "Kariesni davolash", "Kanal davolash", "Tish tozalash (ultratovush)", "Plombalash"],
    },
    {
        "id": "d04",
        "full_name": "Dilnoza Ergasheva",
        "specialty": "Pediatr",
        "degree": "Oliy toifali shifokor",
        "experience_years": 20,
        "room": "112-xona, 1-qavat",
        "schedule": {"days": ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma"], "time": "08:30–14:00"},
        "consultation_price_uzs": 200000,
        "accepts_children": True,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent pediatriya tibbiyot instituti (2005)",
        "about": "0 yoshdan 18 yoshgacha bolalarni kuzatish: o'sish va rivojlanish, emlash kalendari, "
                 "tez-tez kasallanish, ovqatlanish va vitamin yetishmovchiligi masalalari.",
        "procedures": ["Konsultatsiya", "Emlash rejasini tuzish", "O'sish-rivojlanish monitoringi"],
    },
    {
        "id": "d05",
        "full_name": "Sardor Toshpo'latov",
        "specialty": "Travmatolog-ortoped",
        "degree": "Birinchi toifali shifokor",
        "experience_years": 11,
        "room": "202-xona, 2-qavat",
        "schedule": {"days": ["Seshanba", "Payshanba", "Shanba"], "time": "11:00–17:00"},
        "consultation_price_uzs": 240000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2014), travmatologiya ordinaturasi (2016)",
        "about": "Suyak sinishi, cho'kish va burilishlar, bo'g'im og'riqlari, "
                 "umurtqa va tizza bo'g'imi muammolari, gips va immobilizatsiya.",
        "procedures": ["Konsultatsiya", "Gips qo'yish", "Bo'g'im blokadasi", "Rentgen tahlili"],
    },
    {
        "id": "d06",
        "full_name": "Kamola Ismoilova",
        "specialty": "Dermatolog",
        "degree": "Birinchi toifali shifokor",
        "experience_years": 9,
        "room": "110-xona, 1-qavat",
        "schedule": {"days": ["Dushanba", "Chorshanba", "Juma"], "time": "13:00–19:00"},
        "consultation_price_uzs": 220000,
        "accepts_children": True,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2016), dermatovenerologiya ordinaturasi (2018)",
        "about": "Teri toshmalari, akne, allergik dermatit, zamburug'li kasalliklar va "
                 "soch to'kilishi bilan shug'ullanadi.",
        "procedures": ["Konsultatsiya", "Dermoskopiya", "Zamburug' tahlili uchun namuna olish"],
    },
    {
        "id": "d07",
        "full_name": "Rustam Abdullayev",
        "specialty": "Urolog",
        "degree": "Oliy toifali shifokor",
        "experience_years": 22,
        "room": "208-xona, 2-qavat",
        "schedule": {"days": ["Seshanba", "Payshanba", "Shanba"], "time": "09:00–14:00"},
        "consultation_price_uzs": 260000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2003), urologiya ordinaturasi (2005)",
        "about": "Buyrak va siydik yo'li toshlari, prostata kasalliklari, "
                 "siydik chiqarish buzilishlari va infeksiyalari.",
        "procedures": ["Konsultatsiya", "Siydik tahlili talqini", "Buyrak UZI", "Kistoskopiya (yo'llanma bilan)"],
    },
    {
        "id": "d08",
        "full_name": "Zarina Mahmudova",
        "specialty": "Ginekolog",
        "degree": "Oliy toifali shifokor, tibbiyot fanlari nomzodi",
        "experience_years": 19,
        "room": "210-xona, 2-qavat",
        "schedule": {"days": ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma"], "time": "09:00–15:00"},
        "consultation_price_uzs": 250000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus", "ingliz"],
        "education": "Toshkent tibbiyot akademiyasi (2006), akusherlik-ginekologiya ordinaturasi (2008)",
        "about": "Ayollar salomatligi, homiladorlikni rejalashtirish va kuzatish, "
                 "ginekologik tekshiruvlar va profilaktika.",
        "procedures": ["Konsultatsiya", "Kolposkopiya", "Kichik chanoq UZI", "Homiladorlik monitoringi"],
    },
    {
        "id": "d09",
        "full_name": "Jasur Nazarov",
        "specialty": "LOR (otorinolaringolog)",
        "degree": "Birinchi toifali shifokor",
        "experience_years": 10,
        "room": "106-xona, 1-qavat",
        "schedule": {"days": ["Dushanba", "Seshanba", "Chorshanba", "Juma", "Shanba"], "time": "08:00–13:00"},
        "consultation_price_uzs": 190000,
        "accepts_children": True,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2015)",
        "about": "Quloq, tomoq va burun kasalliklari: gaymorit, otit, tonzillit, "
                 "burun bitishi va eshitish pasayishi.",
        "procedures": ["Konsultatsiya", "Endoskopik tekshiruv", "Burun yuvish", "Quloqni tozalash"],
    },
    {
        "id": "d10",
        "full_name": "Malika Sobirova",
        "specialty": "Endokrinolog",
        "degree": "Oliy toifali shifokor",
        "experience_years": 14,
        "room": "212-xona, 2-qavat",
        "schedule": {"days": ["Seshanba", "Payshanba", "Shanba"], "time": "10:00–16:00"},
        "consultation_price_uzs": 240000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2011), endokrinologiya ordinaturasi (2013)",
        "about": "Qalqonsimon bez kasalliklari, qandli diabet (2-tur), vazn va modda almashinuvi "
                 "buzilishlari, gormonal ko'rsatkichlarni talqin qilish.",
        "procedures": ["Konsultatsiya", "Qalqonsimon bez UZI", "Gormonal tahlil talqini"],
    },
    {
        "id": "d11",
        "full_name": "Otabek Xolmatov",
        "specialty": "Gastroenterolog",
        "degree": "Birinchi toifali shifokor",
        "experience_years": 13,
        "room": "214-xona, 2-qavat",
        "schedule": {"days": ["Dushanba", "Chorshanba", "Payshanba", "Shanba"], "time": "09:30–15:00"},
        "consultation_price_uzs": 235000,
        "accepts_children": False,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2012)",
        "about": "Oshqozon va ichak kasalliklari, gastrit, kolit, jigar va o't pufagi muammolari, "
                 "FGDS natijalarini talqin qilish.",
        "procedures": ["Konsultatsiya", "FGDS talqini", "Qorin bo'shlig'i UZI talqini"],
    },
    {
        "id": "d12",
        "full_name": "Sevara Qodirova",
        "specialty": "Oftalmolog",
        "degree": "Birinchi toifali shifokor",
        "experience_years": 8,
        "room": "104-xona, 1-qavat",
        "schedule": {"days": ["Seshanba", "Chorshanba", "Payshanba", "Juma"], "time": "11:00–17:00"},
        "consultation_price_uzs": 210000,
        "accepts_children": True,
        "languages": ["o'zbek", "rus"],
        "education": "Toshkent tibbiyot akademiyasi (2017)",
        "about": "Ko'z o'tkirligini tekshirish, ko'zoynak tanlash, kon'yunktivit, glaukoma va "
                 "katarakta shubhalarini baholash.",
        "procedures": ["Konsultatsiya", "Ko'rish o'tkirligini tekshirish", "Ko'z ichki bosimini o'lchash", "Ko'z tubini ko'rish"],
    },
]

# --------------------------------------------------------------------------
# Xizmatlar va narxlar (UZS)
# --------------------------------------------------------------------------
SERVICES = [
    # --- Konsultatsiyalar ---
    {"id": "s01", "name": "Birlamchi konsultatsiya (ixtisoslashgan shifokor)", "category": "Konsultatsiya",
     "price_uzs": 250000, "duration_min": 30, "note": "Shifokor ixtisosligiga qarab 180 000–260 000 so'm"},
    {"id": "s02", "name": "Takroriy konsultatsiya (30 kun ichida)", "category": "Konsultatsiya",
     "price_uzs": 150000, "duration_min": 20, "note": "O'sha shifokorga qayta murojaat"},
    {"id": "s03", "name": "Pediatr konsultatsiyasi", "category": "Konsultatsiya",
     "price_uzs": 200000, "duration_min": 30, "note": "0–18 yosh"},
    {"id": "s04", "name": "Onlayn konsultatsiya (video)", "category": "Konsultatsiya",
     "price_uzs": 180000, "duration_min": 20, "note": "Oldindan to'lov talab qilinadi"},
    {"id": "s05", "name": "Uyga shifokor chaqirish (Toshkent sh. ichida)", "category": "Konsultatsiya",
     "price_uzs": 400000, "duration_min": 45, "note": "Chilonzor, Yunusobod, Yakkasaroy tumanlari"},

    # --- Diagnostika ---
    {"id": "s10", "name": "Umumiy qon tahlili (KLA)", "category": "Laboratoriya",
     "price_uzs": 90000, "duration_min": 10, "preparation": "8–12 soat och qoringa (suv ichish mumkin)"},
    {"id": "s11", "name": "Qon shakari (glyukoza)", "category": "Laboratoriya",
     "price_uzs": 60000, "duration_min": 10, "preparation": "8–12 soat och qoringa"},
    {"id": "s12", "name": "Biokimyoviy qon tahlili (12 ko'rsatkich)", "category": "Laboratoriya",
     "price_uzs": 280000, "duration_min": 15, "preparation": "8–12 soat och qoringa"},
    {"id": "s13", "name": "Qalqonsimon bez gormonlari (TSH, T3, T4)", "category": "Laboratoriya",
     "price_uzs": 320000, "duration_min": 15, "preparation": "Ertalab, och qoringa"},
    {"id": "s14", "name": "Umumiy siydik tahlili", "category": "Laboratoriya",
     "price_uzs": 70000, "duration_min": 10, "preparation": "Ertalabki siydik, toza idishda"},
    {"id": "s15", "name": "EKG (12 kanalli)", "category": "Diagnostika",
     "price_uzs": 120000, "duration_min": 15},
    {"id": "s16", "name": "ExoKG (yurak UZI)", "category": "Diagnostika",
     "price_uzs": 300000, "duration_min": 30},
    {"id": "s17", "name": "Xolter monitoringi (24 soat)", "category": "Diagnostika",
     "price_uzs": 450000, "duration_min": 30, "note": "Qurilma 24 soatga o'rnatiladi, keyingi kuni olinadi"},
    {"id": "s18", "name": "Qorin bo'shlig'i UZI (kompleks)", "category": "Diagnostika",
     "price_uzs": 280000, "duration_min": 25, "preparation": "6–8 soat och qoringa"},
    {"id": "s19", "name": "Buyrak va siydik yo'llari UZI", "category": "Diagnostika",
     "price_uzs": 250000, "duration_min": 25, "preparation": "Siydik pufagi to'la bo'lishi kerak"},
    {"id": "s20", "name": "Kichik chanoq UZI", "category": "Diagnostika",
     "price_uzs": 260000, "duration_min": 25, "preparation": "Siydik pufagi to'la bo'lishi kerak"},
    {"id": "s21", "name": "Qalqonsimon bez UZI", "category": "Diagnostika",
     "price_uzs": 220000, "duration_min": 20},
    {"id": "s22", "name": "Rentgen (1 ta soha)", "category": "Diagnostika",
     "price_uzs": 150000, "duration_min": 15, "note": "Homiladorlikda taqiqlanadi"},
    {"id": "s23", "name": "FGDS (oshqozon-ichak endoskopiyasi)", "category": "Diagnostika",
     "price_uzs": 550000, "duration_min": 40, "preparation": "8–12 soat och qoringa; shifokor ko'rigidan keyin"},

    # --- Muolajalar ---
    {"id": "s30", "name": "Kariesni davolash (1 tish)", "category": "Stomatologiya",
     "price_uzs": 450000, "duration_min": 45, "note": "Plomba turiga qarab 450 000–900 000 so'm"},
    {"id": "s31", "name": "Kanal davolash (1 kanal)", "category": "Stomatologiya",
     "price_uzs": 700000, "duration_min": 60},
    {"id": "s32", "name": "Tish tozalash (ultratovush + polirovka)", "category": "Stomatologiya",
     "price_uzs": 420000, "duration_min": 45},
    {"id": "s33", "name": "Tish olish (oddiy)", "category": "Stomatologiya",
     "price_uzs": 300000, "duration_min": 30},
    {"id": "s34", "name": "Physiotherapy — magnit terapiya (1 seans)", "category": "Muolaja",
     "price_uzs": 120000, "duration_min": 20},
    {"id": "s35", "name": "Massaj (tibbiy, 1 seans)", "category": "Muolaja",
     "price_uzs": 180000, "duration_min": 40},
    {"id": "s36", "name": "Emizish/emlash (1 emlash)", "category": "Muolaja",
     "price_uzs": 130000, "duration_min": 20, "note": "Emlash kalendariga muvofiq"},
    {"id": "s37", "name": "Tomchilatib yuborish (kapelnitsa, shifokor ko'rigidan keyin)", "category": "Muolaja",
     "price_uzs": 200000, "duration_min": 60, "note": "Faqat shifokor ko'rigi va yo'llanmasi bilan"},
    {"id": "s38", "name": "Gips qo'yish", "category": "Travmatologiya",
     "price_uzs": 350000, "duration_min": 40},

    # --- Paketlar ---
    {"id": "s50", "name": "Yillik check-up paketi (Standart)", "category": "Paket",
     "price_uzs": 1500000, "duration_min": 180,
     "note": "Terapevt ko'rigi + KLA + biokimyo + siydik tahlili + EKG + qorin UZI"},
    {"id": "s51", "name": "Yillik check-up paketi (Premium)", "category": "Paket",
     "price_uzs": 2900000, "duration_min": 240,
     "note": "Standart paket + qalqonsimon bez gormonlari + ExoKG + 3 ixtisoslashgan konsultatsiya"},
    {"id": "s52", "name": "Ayollar check-up paketi", "category": "Paket",
     "price_uzs": 1950000, "duration_min": 210,
     "note": "Ginekolog ko'rigi + kichik chanoq UZI + laboratoriya tahlillari + mammologiya yo'llanmasi"},
]

# --------------------------------------------------------------------------
# Klinika qoidalari
# --------------------------------------------------------------------------
RULES = [
    {"id": "r01", "title": "Qabulga yozilish tartibi", "category": "Qabul",
     "text": "Qabul faqat oldindan yozilish orqali amalga oshiriladi. Yozilish dastur orqali, "
             "telefon +998 71 200 45 45 yoki '1145' qisqa raqami orqali yoki "
             "administratorga murojaat qilib amalga oshiriladi. Navbatsiz qabul qilinmaydi, "
             "faqat bo'sh oyna bo'lsa istisno qilinadi."},
    {"id": "r02", "title": "Kechikish qoidasi", "category": "Qabul",
     "text": "Bemor rejalashtirilgan vaqtdan 15 daqiqadan ko'proq kechiksa, qabul boshqa bemorga "
             "o'tkazilishi va yangi vaqt belgilanishi mumkin."},
    {"id": "r03", "title": "Bekor qilish", "category": "Qabul",
     "text": "Qabulni bekor qilish yoki ko'chirish qabul vaqtidan 3 soat oldin aytilishi kerak. "
             "Uch marta ogohlantirilmasdan kelmaslik holatida keyingi yozilishlar cheklanishi mumkin."},
    {"id": "r04", "title": "To'lov tartibi", "category": "To'lov",
     "text": "To'lov xizmat ko'rsatilgandan so'ng kassada amalga oshiriladi. Naqd, plastik karta "
             "(UzCard, Humo, Visa, Mastercard) va yuridik shaxslar uchun bank o'tkazmasi qabul qilinadi. "
             "Sug'urta kompaniyalari: Uzbekinvest, Alskom, Gross Insurance."},
    {"id": "r05", "title": "Chegirmalar", "category": "To'lov",
     "text": "Chegirmalar faqat klinika ma'muriyati tomonidan e'lon qilingan rasmiy aksiyalar doirasida "
             "beriladi. Chatbot, administrator yoki konsultant hech qanday holatda e'lon qilinmagan "
             "chegirma, bepul xizmat yoki narxni tushirishni taklif qilmaydi."},
    {"id": "r06", "title": "Hujjatlar", "category": "Hujjatlar",
     "text": "Birinchi murojaatda pasport yoki shaxsni tasdiqlovchi hujjat olib kelish talab qilinadi. "
             "Bolalar uchun tug'ilganlik haqidagi guvohnoma. Sug'urta polisi bo'lsa, polis nusxasi kerak."},
    {"id": "r07", "title": "Qabulni tayyorlash", "category": "Tayyorgarlik",
     "text": "Qon va ba'zi laboratoriya tahlillari 8–12 soat och qoringa topshiriladi (suv ichish mumkin). "
             "Iltimos, tahlil topshirishdan oldin aniq ko'rsatmalar bo'yicha administratorga aniqlashtiring."},
    {"id": "r08", "title": "Hujjatlarni qaytarish", "category": "Hujjatlar",
     "text": "Tekshiruv natijalari va xulosa (xulosa blanki) tayyor bo'lgandan keyin 3 ish kuni "
             "ichida klinikadan qog'oz shaklda yoki elektron pochta orqali olinadi."},
    {"id": "r09", "title": "Shaxsiy ma'lumotlar", "category": "Maxfiylik",
     "text": "Bemor haqidagi ma'lumotlar tibbiy maxfiylik hisoblanadi va faqat bemorning o'zi, "
             "qonuniy vakili yoki tibbiy yozuvlarni talab qilish huquqiga ega shaxslarga beriladi. "
             "Chatbot hech qanday holatda uchinchi shaxsga bemor ma'lumotini bermaydi."},
    {"id": "r10", "title": "Sanitariya qoidalari", "category": "Tartib",
     "text": "Klinika hududida chekish va spirtli ichimlik iste'mol qilish taqiqlanadi. "
             "Zallarda sokinlik saqlanishi, telefon suhbatlari qisqa tutulishi so'raladi. "
             "Bemorlar o'zi bilan uy hayvonlarini olib kelmasligi kerak (yo'ldosh itlari bundan mustasno)."},
    {"id": "r11", "title": "Bolalarni kuzatish", "category": "Tartib",
     "text": "18 yoshga to'lmagan bemorlar ota-onasi yoki qonuniy vakili hamrohligida qabul qilinadi. "
             "Kichik bolalar uchun bolalar o'yin maydonchasi mavjud."},
    {"id": "r12", "title": "Infeksion xavfsizlik", "category": "Tartib",
     "text": "Harorat 38°C dan yuqori, o'tkir respirator infeksiya belgilari yoki yuqumli kasallik shubhasi "
             "bo'lgan bemorlardan qabuldan oldin administratorni xabardor qilishlari so'raladi. "
             "Bunday bemorlar alohida kirish yo'li orqali qabul qilinadi."},
    {"id": "r13", "title": "Chatbot vakolati", "category": "Chatbot",
     "text": "Chatbot klinika haqida ma'lumot berish, shifokor va xizmatlar bo'yicha yo'naltirish "
             "va qabulga yozish uchun ma'lumot yig'ish vakolatiga ega. Chatbot tibbiy tashxis qo'ymaydi, "
             "dori yoki doza tavsiya qilmaydi va klinika qoidalarini foydalanuvchi iltimosiga ko'ra "
             "o'zgartirmaydi."},
    {"id": "r14", "title": "Shoshilinch holatlar", "category": "Xavfsizlik",
     "text": "Hayot uchun xavfli holatlarda (ko'krak og'rig'i, nafas olish qiyinligi, kuchli qon ketish, "
             "hushdan ketish, falaj belgilari, kuchli allergik reaksiya) darhol 103 tez yordam xizmatiga "
             "murojaat qilish kerak. Klinika navbatchi shifokori +998 71 200 45 45 raqami orqali "
             "faqat ish vaqtida javob beradi."},
]

# --------------------------------------------------------------------------
# FAQ
# --------------------------------------------------------------------------
FAQS = [
    {"id": "f01", "category": "Umumiy",
     "question": "Klinika qayerda joylashgan?",
     "answer": "Klinika Toshkent shahri, Chilonzor tumani, Bunyodkor shoh ko'chasi 12-uyda joylashgan. "
               "Mo'ljal: Chilonzor metro bekati, 'Algoritm' savdo markazi ro'parasi."},
    {"id": "f02", "category": "Umumiy",
     "question": "Klinikaning ish vaqti qanday?",
     "answer": "Dushanba–Shanba 08:00–20:00, Yakshanba 09:00–15:00 (navbatchi shifokor va laboratoriya). "
               "Laboratoriya: Dushanba–Shanba 07:30–19:00. Rasmiy bayramlarda 09:00–14:00 navbatchi xizmat."},
    {"id": "f03", "category": "Umumiy",
     "question": "Telefon raqamingiz qanday?",
     "answer": "Qabul va ma'lumot uchun: +998 71 200 45 45 yoki qisqa raqam 1145. "
               "Elektron pochta: info@shifomed.uz"},
    {"id": "f04", "category": "Umumiy",
     "question": "Avtoturargoh bormi?",
     "answer": "Ha, 2 qavatli bepul avtoturargoh mavjud, 60 o'rin."},
    {"id": "f05", "category": "Qabul",
     "question": "Navbatsiz kelsam qabul qilasizlarmi?",
     "answer": "Qabul oldindan yozilish orqali amalga oshiriladi (r01 qoidasi). "
               "Bo'sh oyna bo'lsa, administrator navbatsiz bemorni ham qabul qilishi mumkin."},
    {"id": "f06", "category": "To'lov",
     "question": "Qanday to'lov usullari bor?",
     "answer": "Naqd, plastik karta (UzCard, Humo, Visa, Mastercard) va yuridik shaxslar uchun "
               "bank o'tkazmasi. To'lov xizmat ko'rsatilgandan so'ng kassada amalga oshiriladi."},
    {"id": "f07", "category": "To'lov",
     "question": "Sug'urta bilan ishlaysizlarmi?",
     "answer": "Ha, quyidagi sug'urta kompaniyalari bilan hamkorlik qilamiz: Uzbekinvest, Alskom, "
               "Gross Insurance. Sug'urta orqali xizmat olish uchun polis va yo'llanma kerak bo'ladi."},
    {"id": "f08", "category": "Qabul",
     "question": "Onlayn konsultatsiya bormi?",
     "answer": "Ha, video orqali onlayn konsultatsiya mavjud — 180 000 so'm, 20 daqiqa "
               "(s04 xizmati). Onlayn konsultatsiya uchun oldindan to'lov talab qilinadi."},
    {"id": "f09", "category": "Xizmatlar",
     "question": "Tahlil natijalari qachon tayyor bo'ladi?",
     "answer": "Ko'p laboratoriya tahlillari o'sha kuni yoki keyingi ish kuni tayyor bo'ladi. "
               "Natijalar 3 ish kuni ichida qog'oz shaklda yoki elektron pochta orqali beriladi (r08)."},
    {"id": "f10", "category": "Xizmatlar",
     "question": "Uyga shifokor chaqirish mumkinmi?",
     "answer": "Ha, uyga shifokor chaqirish xizmati mavjud — 400 000 so'm (s05), Toshkent shahri "
               "Chilonzor, Yunusobod va Yakkasaroy tumanlari uchun."},
    {"id": "f11", "category": "Xizmatlar",
     "question": "Check-up paketlarida nimalar bor?",
     "answer": "Standart paket (1 500 000 so'm): terapevt ko'rigi, umumiy qon tahlili, biokimyo, "
               "siydik tahlili, EKG va qorin bo'shlig'i UZI. Premium paket (2 900 000 so'm) qo'shimcha "
               "qalqonsimon bez gormonlari, ExoKG va 3 ixtisoslashgan konsultatsiyani o'z ichiga oladi."},
    {"id": "f12", "category": "Hujjatlar",
     "question": "Kelmaganimda nima bo'ladi?",
     "answer": "Qabulni bekor qilish yoki ko'chirish qabul vaqtidan kamida 3 soat oldin aytilishi kerak. "
               "Uch marta ogohlantirilmasdan kelmaslik holatida keyingi yozilishlar cheklanishi mumkin (r03)."},
    {"id": "f13", "category": "Umumiy",
     "question": "Qanday tillarda xizmat ko'rsatiladi?",
     "answer": "Shifokorlarimiz o'zbek va rus tillarida xizmat ko'rsatadi, ingliz tili cheklangan "
               "darajada mavjud. Aniq shifokor bo'yicha tillar shifokor profilida ko'rsatilgan."},
    {"id": "f14", "category": "Tayyorgarlik",
     "question": "Qon tahlilidan oldin nima qilishim kerak?",
     "answer": "Ko'p qon tahlillari 8–12 soat och qoringa topshiriladi (suv ichish mumkin). "
               "Tahlil turiga qarab talab farq qiladi, aniq ko'rsatma uchun administratorga murojaat qiling."},
    {"id": "f15", "category": "Xavfsizlik",
     "question": "Holatim yomonlashsa nima qilishim kerak?",
     "answer": "Hayot uchun xavfli belgilarda (ko'krak og'rig'i, nafas olish qiyinligi, kuchli qon ketish, "
               "hushdan ketish, falaj belgilari, kuchli allergik reaksiya) darhol 103 raqamiga qo'ng'iroq qiling "
               "yoki eng yaqin shoshilinch tibbiy yordam bo'limiga murojaat qiling, kutib turmang."},
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "doctors.json": {"clinic": CLINIC, "count": len(DOCTORS), "doctors": DOCTORS},
        "services.json": {"clinic": CLINIC["name"], "currency": "UZS",
                          "price_note": "Narxlar 2026-yil uchun ko'rsatilgan va o'zgarishi mumkin. "
                                        "Aniq narx administrator orqali tasdiqlanadi.",
                          "count": len(SERVICES), "services": SERVICES},
        "rules.json": {"clinic": CLINIC["name"], "count": len(RULES), "rules": RULES},
        "faqs.json": {"clinic": CLINIC, "count": len(FAQS), "faqs": FAQS},
    }
    for filename, payload in payloads.items():
        path = DATA_DIR / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK  {path.relative_to(DATA_DIR.parent)}  ({payload['count']} ta yozuv)")

    print(f"\nJami: {len(DOCTORS)} shifokor, {len(SERVICES)} xizmat, "
          f"{len(RULES)} qoida, {len(FAQS)} FAQ")


if __name__ == "__main__":
    main()
