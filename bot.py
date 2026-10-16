import os
import sys
import re
import time
import random
import sqlite3
import asyncio
import logging
import warnings
from collections import deque
from dotenv import load_dotenv

# Kutubxonalardagi eskirish ogohlantirishlarini yashirish
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# 1. Muhit o'zgaruvchilarini (.env) yuklash
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    print("XATO: TELEGRAM_BOT_TOKEN topilmadi! Iltimos, .env faylini tekshiring.")
    sys.exit(1)

if not GEMINI_API_KEY:
    print("XATO: GEMINI_API_KEY topilmadi! Iltimos, .env faylini tekshiring.")
    sys.exit(1)

# 2. Log yozishni sozlash
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 3. Google Gemini AI ni sozlash
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """
Siz aqlli, har tomonlama yordam beruvchi, xushmuomala va kuchli sun'iy intellekt (AI) yordamchisisiz.

Qat'iy amal qilinishi shart bo'lgan asosiy qoidalar:
1. Tilga moslashish: Foydalanuvchi qaysi tilda murojaat qilsa (o'zbek, rus, ingliz va h.k.), aynan o'sha tilda mukammal, ravon va grammatik to'g'ri javob bering.
2. Har tomonlama bilim: Dasturlash (kod yozish va xatolarni tuzatish), matematika, fizika, tarjima, maqola/xat yozish, ilmiy va kundalik hayotiy maslahatlar bo'yicha aniq va mantiqiy javoblar bering.
3. Yaratuvchi va Admin haqida: Agar foydalanuvchi "seni kim yaratgan?", "muallifing kim?", "admin kim?", "admin bilan qanday bog'lansa bo'ladi?" deb so'rasa, "Meni O'ljaboyev Anvar yaratgan. Admin: @Oljaboyev_Anvar_2005 (https://t.me/Oljaboyev_Anvar_2005)" deb javob bering.
4. AI ekanlikni yashirmaslik: Siz sun'iy intellektsiz (AI). Buni aslo yashirmang va o'zingizni inson deb da'vo qilmang. O'ljaboyev Anvar tomonidan yaratilgan va sozlangan kuchli AI yordamchi ekanligingizni e'tirof eting.
5. Dizayn va Chiroyli ko'rinish (MUHIM):
   - Javoblarni Telegramda chiroyli, tartibli va ko'rkam qilib taqdim eting.
   - Ro'yxatlar tuzishda keraksiz yulduzcha (*) va belgilarni ko'paytirmang. Bandlarni chiroyli belgilardan (🔹, 📌, 1., 2.) foydalanib yozing.
   - Har qanday dasturlash kodini albatta alohida blok qilib (```dasturlash_tili\\nkod\\n```) ichida yozing. Kod boshqa matnlardan yaqqol ajralib, alohida chiroyli kod qutisi sifatida turishi shart.
"""

# Zaxira modellar ro'yxati (Biri band bo'lsa darhol ikkinchisiga o'tadi)
AVAILABLE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


async def ask_gemini_with_fallback(user_text: str, history: list) -> str:
    """
    So'rovni navbat bilan modellarga yuboradi.
    Agar bittasida limit tugasa, darhol ikkinchisiga o'tib javob qaytaradi!
    """
    last_error = None
    for model_name in AVAILABLE_MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(user_text)
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            logger.warning(f"Model {model_name} band yoki xato berdi ({e}), keyingisiga o'tilmoqda...")
            continue
    raise last_error


# ----------------- 4. BAZA VA CAPTCHA XAVFSIZLIK TIZIMI -----------------
DB_PATH = "bot_users.db"


def init_db():
    """Tasdiqlangan foydalanuvchilar jadvalini yaratish."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS verified_users (
                user_id INTEGER PRIMARY KEY,
                first_name TEXT,
                username TEXT,
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB yaratishda xato: {e}")


def load_verified_users() -> set:
    """Server qayta yonganda ham tasdiqlangan insonlarni xotiraga yuklash."""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM verified_users")
        rows = cursor.fetchall()
        conn.close()
        return set(r[0] for r in rows)
    except Exception as e:
        logger.error(f"Foydalanuvchilarni yuklashda xato: {e}")
        return set()


def save_verified_user(user_id: int, first_name: str = "", username: str = ""):
    """Yangi tasdiqlangan foydalanuvchini bazaga yozish."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO verified_users (user_id, first_name, username)
            VALUES (?, ?, ?)
        """,
            (user_id, first_name, username),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Foydalanuvchini saqlashda xato: {e}")


# Tasdiqlangan foydalanuvchilar to'plami (O(1) tezlikda tekshirish uchun)
verified_users = load_verified_users()
active_captchas = {}  # user_id -> to'g'ri javob

# ----------------- 5. ANTI-FLOOD & RATE LIMITING -----------------
user_request_times = {}  # user_id -> deque of timestamps
user_blacklist = {}  # user_id -> unban_timestamp (float)

RATE_LIMIT_SECONDS = 2.0
FLOOD_WINDOW_SECONDS = 3.0
FLOOD_MAX_REQUESTS = 5
BAN_DURATION_SECONDS = 600.0  # 10 daqiqa


def check_anti_flood(user_id: int) -> tuple[bool, str]:
    """
    Foydalanuvchi spambot yoki flood qilayotganini tekshiradi.
    Qaytaradi: (ruxsat_berilganmi, sabab)
    """
    now = time.time()

    # 1. Qora ro'yxatni tekshirish
    if user_id in user_blacklist:
        unban_time = user_blacklist[user_id]
        if now < unban_time:
            # Bloklangan: server resursini tejash uchun jim tashlab yuboramiz
            return False, "banned"
        else:
            del user_blacklist[user_id]

    # 2. Flood aniqlash (3 soniyada 5 tadan ko'p xabar)
    if user_id not in user_request_times:
        user_request_times[user_id] = deque(maxlen=10)

    times = user_request_times[user_id]
    while times and (now - times[0] > FLOOD_WINDOW_SECONDS):
        times.popleft()

    times.append(now)

    if len(times) >= FLOOD_MAX_REQUESTS:
        user_blacklist[user_id] = now + BAN_DURATION_SECONDS
        logger.warning(f"🚨 FLOOD ATTACK! User {user_id} 10 daqiqaga bloklandi.")
        return False, "flood_ban"

    # 3. Rate limiting (Har 2 soniyada maksimal 1 ta so'rov)
    if len(times) >= 2:
        prev_time = times[-2]
        if now - prev_time < RATE_LIMIT_SECONDS:
            return False, "rate_limit"

    return True, "ok"


def generate_captcha(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Dinamik matematik misol va 4 ta random inline tugma yasash."""
    num1 = random.randint(3, 15)
    num2 = random.randint(2, 12)
    correct_ans = num1 + num2
    active_captchas[user_id] = correct_ans

    wrong_answers = set()
    while len(wrong_answers) < 3:
        fake = correct_ans + random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
        if fake != correct_ans and fake > 0:
            wrong_answers.add(fake)

    options = list(wrong_answers) + [correct_ans]
    random.shuffle(options)

    buttons = [
        InlineKeyboardButton(
            text=str(opt),
            callback_data=f"cap:{user_id}:{opt}",
        )
        for opt in options
    ]

    keyboard = InlineKeyboardMarkup(
        [
            [buttons[0], buttons[1]],
            [buttons[2], buttons[3]],
        ]
    )

    text = (
        "🤖 <b>Xavfsizlik tekshiruvi (Anti-Spam Gate)</b>\n\n"
        "Botdan to'liq foydalanish uchun inson ekanligingizni tasdiqlang:\n"
        f"👉 <b>{num1} + {num2} = ?</b>\n\n"
        "<i>To'g'ri javobni pastdagi tugmalardan tanlang:</i>"
    )
    return text, keyboard


# ----------------- 6. XOTIRA VA TELEGRAM FORMATLASH -----------------
MAX_HISTORY_LENGTH = 20
user_chat_histories = {}


def get_user_history(user_id: int) -> deque:
    """Foydalanuvchining xotira navbatini qaytaradi."""
    if user_id not in user_chat_histories:
        user_chat_histories[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)
    return user_chat_histories[user_id]


def escape_tg_html(text: str) -> str:
    """Telegram HTML talab qiladigan asosiy belgilarni xavfsiz qiladi."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_to_telegram_html(text: str) -> str:
    """AI javobidagi Markdown belgilarini Telegram uchun chiroyli HTML ga aylantiradi."""
    code_blocks = []

    def save_code_block(match):
        lang = match.group(1).strip() if match.group(1) else ""
        code = match.group(2)
        escaped_code = escape_tg_html(code)
        if lang:
            tag = f'<pre><code class="language-{lang}">{escaped_code}</code></pre>'
        else:
            tag = f'<pre>{escaped_code}</pre>'
        code_blocks.append(tag)
        return f"@@@CODEBLOCK{len(code_blocks)-1}@@@"

    text = re.sub(r"```([a-zA-Z0-9_+-]*)\n?(.*?)```", save_code_block, text, flags=re.DOTALL)

    inline_codes = []

    def save_inline_code(match):
        code = match.group(1)
        escaped_code = escape_tg_html(code)
        inline_codes.append(f"<code>{escaped_code}</code>")
        return f"@@@INLINECODE{len(inline_codes)-1}@@@"

    text = re.sub(r"`([^`\n]+)`", save_inline_code, text)
    text = escape_tg_html(text)
    text = re.sub(r"^(?:#{1,6})\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"^[ \t]*[\*\-][ \t]+", "🔹 ", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

    for idx, tag in enumerate(inline_codes):
        text = text.replace(f"@@@INLINECODE{idx}@@@", tag)
    for idx, tag in enumerate(code_blocks):
        text = text.replace(f"@@@CODEBLOCK{idx}@@@", tag)

    return text


def split_message_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """Telegram xabar limiti (4096 belgi) dan oshib ketmasligi uchun bo'laklarga ajratadi."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_chunk_size:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            while len(line) > max_chunk_size:
                chunks.append(line[:max_chunk_size])
                line = line[max_chunk_size:]
            current_chunk = line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


async def keep_typing_action(chat_id: int, context: ContextTypes.DEFAULT_TYPE, stop_event: asyncio.Event):
    """AI javob tayyorlayotgan vaqtda 'yozmoqda...' holatini ko'rsatib turish."""
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# ----------------- 7. BUYRUQLAR (COMMANDS) -----------------
def get_welcome_text(first_name: str) -> str:
    return (
        f"Assalomu alaykum, <b>{first_name}</b>!\n\n"
        "Men har tomonlama yordam bera oladigan kuchli <b>AI yordamchiman</b>.\n\n"
        "Quyidagi sohalarda sizga yordam bera olaman:\n"
        "🔹 Dasturlash va kodlardagi xatolarni tuzatish\n"
        "🔹 Matematika va ilmiy masalalar\n"
        "🔹 Matnlar yaratish, tahrirlash va tarjima qilish\n"
        "🔹 Turli mavzularda maslahat va tahlillar\n\n"
        "Menga xohlagan tilingizda (o'zbek, rus, ingliz...) yozishingiz mumkin.\n\n"
        "📌 <b>Asosiy buyruqlar:</b>\n"
        "• /clear — Xotirani tozalash\n"
        "• /help — Bot haqida ma'lumot\n"
        "• /admin — Admin bilan bog'lanish\n\n"
        "Savolingizni yozavering, men tayyorman!"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi /start bosganda xavfsizlik va salomlashish."""
    user = update.effective_user
    if not user:
        return
    user_id = user.id

    # 1. Anti-flood tekshiruvi
    allowed, reason = check_anti_flood(user_id)
    if not allowed:
        if reason == "flood_ban":
            await update.message.reply_text("⚠️ Juda ko'p so'rov yubordingiz. Iltimos, 10 daqiqa kuting.")
        return

    # 2. CAPTCHA Gate (Agar yangi foydalanuvchi bo'lsa, avval tekshiruv!)
    if user_id not in verified_users:
        text, kb = generate_captcha(user_id)
        await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        return

    # 3. Tasdiqlangan foydalanuvchiga to'liq menyuni ochish
    first_name = escape_tg_html(user.first_name) if user.first_name else "Do'stim"
    await update.message.reply_text(get_welcome_text(first_name), parse_mode=ParseMode.HTML)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi suhbat tarixini tozalash."""
    user = update.effective_user
    if not user:
        return
    user_id = user.id

    allowed, reason = check_anti_flood(user_id)
    if not allowed:
        return

    if user_id not in verified_users:
        text, kb = generate_captcha(user_id)
        await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        return

    if user_id in user_chat_histories:
        user_chat_histories[user_id].clear()

    await update.message.reply_text(
        "🗑 <b>Suhbat xotirasi tozalandi!</b> Endi yangi mavzuni bemalol noldan boshlashimiz mumkin.",
        parse_mode=ParseMode.HTML,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Yordam bo'limi."""
    keyboard = [
        [InlineKeyboardButton("👨‍💻 Admin bilan bog'lanish", url="https://t.me/Oljaboyev_Anvar_2005")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    help_text = (
        "ℹ️ <b>Qo'llanma va Imkoniyatlar:</b>\n\n"
        "🔹 Bot suhbat davomida oxirgi 20 ta xabarni xotirada saqlab turadi.\n"
        "🔹 Mavzu o'zgarsa, /clear buyrug'i orqali yangi suhbat boshlang.\n"
        "🔹 Kodlar va matnlar alohida chiroyli dizaynda taqdim etiladi.\n\n"
        "👨‍💻 <b>Yaratuvchi:</b> O'ljaboyev Anvar\n"
        "👉 <b>Telegram:</b> @Oljaboyev_Anvar_2005\n"
        "⚡️ <b>AI Dvigateli:</b> Google Gemini AI"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin bilan bog'lanish buyrug'i."""
    keyboard = [
        [InlineKeyboardButton("💬 Anvarga xabar yozish (@Oljaboyev_Anvar_2005)", url="https://t.me/Oljaboyev_Anvar_2005")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    admin_text = (
        "👨‍💻 <b>Bot administratori va yaratuvchisi:</b>\n"
        "O'ljaboyev Anvar\n\n"
        "👉 Telegram: <b>@Oljaboyev_Anvar_2005</b>\n\n"
        "Savol, taklif yoki murojaatingiz bo'lsa, quyidagi tugmani bosib bemalol yozishingiz mumkin:"
    )
    await update.message.reply_text(admin_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)


# ----------------- 8. CAPTCHA TUGMALARI BOSILGANDA -----------------
async def captcha_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi CAPTCHA tugmasini bosganda tekshirish."""
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data.split(":")
    if len(data) != 3 or data[0] != "cap":
        return

    target_user_id = int(data[1])
    selected_ans = int(data[2])
    clicking_user_id = query.from_user.id

    # Birovning tekshiruvini boshqa birov bosa olmaydi
    if clicking_user_id != target_user_id:
        await query.answer("⚠️ Bu tekshiruv siz uchun emas!", show_alert=True)
        return

    correct_ans = active_captchas.get(target_user_id)
    if correct_ans is not None and selected_ans == correct_ans:
        # To'g'ri javob! Bazaga yozish va xotiraga qo'shish
        verified_users.add(target_user_id)
        save_verified_user(
            target_user_id,
            first_name=query.from_user.first_name or "",
            username=query.from_user.username or "",
        )
        active_captchas.pop(target_user_id, None)

        await query.answer("✅ To'g'ri javob!")
        await query.edit_message_text(
            "✅ <b>Tabriklaymiz, siz muvaffaqiyatli tasdiqlandingiz!</b>",
            parse_mode=ParseMode.HTML,
        )

        first_name = escape_tg_html(query.from_user.first_name) if query.from_user.first_name else "Do'stim"
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=get_welcome_text(first_name),
            parse_mode=ParseMode.HTML,
        )
        logger.info(f"Inson foydalanuvchi tasdiqlandi: {target_user_id} - {first_name}")
    else:
        # Noto'g'ri javob: yangi misol beriladi
        await query.answer("❌ Noto'g'ri javob! Yangi misolni yeching.", show_alert=True)
        text, kb = generate_captcha(target_user_id)
        try:
            await query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        except Exception:
            pass


# ----------------- 9. MATNLI XABARLARNI QAYTA ISHLASH -----------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    # 1. ANTI-FLOOD FILTRI (Spambotlar shu joyning o'zidayoq to'xtatiladi!)
    allowed, reason = check_anti_flood(user_id)
    if not allowed:
        if reason == "flood_ban":
            await update.message.reply_text("⚠️ Juda ko'p so'rov yubordingiz. Siz 10 daqiqaga cheklandingiz!")
        elif reason == "rate_limit":
            await update.message.reply_text("⚠️ Iltimos, shoshilmang! So'rovlar orasida 2 soniya kuting.")
        # Banned bo'lsa umuman javob qaytarmaydi (serverni ortiqcha yuklamaslik uchun)
        return

    # 2. CAPTCHA GATE FILTRI (Botlar bu to'siqdan o'ta olmaydi va Gemini ga yetib bormaydi!)
    if user_id not in verified_users:
        text, kb = generate_captcha(user_id)
        await update.message.reply_text(
            "🛑 <b>Kechirasiz! Botdan foydalanish uchun avval quyidagi tekshiruvdan o'ting:</b>\n\n" + text,
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )
        return

    # 3. FAQAT TASDIQLANGAN HAQIQIY INSONLAR UCHUN GEMINI AI ISHLAYDI:
    history = get_user_history(user_id)

    stop_typing_event = asyncio.Event()
    typing_task = asyncio.create_task(
        keep_typing_action(chat_id, context, stop_typing_event)
    )

    try:
        gemini_history = list(history)
        reply_text = await ask_gemini_with_fallback(user_text, gemini_history)

        if not reply_text:
            reply_text = "Kechirasiz, savolingizga javob shakllantirib bo'lmadi. Qaytadan urinib ko'ring."

        history.append({"role": "user", "parts": [user_text]})
        history.append({"role": "model", "parts": [reply_text]})

        chunks = split_message_text(reply_text)
        for chunk in chunks:
            html_chunk = markdown_to_telegram_html(chunk)
            try:
                await update.message.reply_text(html_chunk, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.warning(f"HTML yuborishda xatolik: {e}, oddiy matn ko'rinishida yuboriladi")
                await update.message.reply_text(chunk)

    except Exception as error:
        logger.error(f"Xatolik yuz berdi: {error}", exc_info=True)
        error_msg = str(error)

        if "429" in error_msg or "ResourceExhausted" in error_msg:
            user_friendly_error = (
                "⚠️ Hozirda AI serveriga so'rovlar ko'payib ketdi (limit tugadi). "
                "Iltimos, 1-2 daqiqa kuting va qayta yozib ko'ring."
            )
        elif "blocked" in error_msg.lower() or "safety" in error_msg.lower():
            user_friendly_error = (
                "⚠️ Kechirasiz, xavfsizlik filtrlari tufayli bu savolga javob berish cheklandi."
            )
        else:
            user_friendly_error = (
                "⚠️ Kechirasiz, texnik xatolik yuz berdi. "
                "Iltimos, qayta urinib ko'ring yoki /clear buyrug'i orqali xotirani tozalang."
            )

        await update.message.reply_text(user_friendly_error)

    finally:
        stop_typing_event.set()
        await typing_task


# ----------------- 10. XATOLIKLARNI USHLASH -----------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Telegram xatoligi yuz berdi:", exc_info=context.error)


async def post_init(application: Application) -> None:
    """Telegram menyusini o'rnatish."""
    commands = [
        BotCommand("start", "🚀 Botni boshlash"),
        BotCommand("clear", "🗑 Xotirani tozalash"),
        BotCommand("help", "ℹ️ Yordam"),
        BotCommand("admin", "👨‍💻 Admin bilan bog'lanish"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Telegram menyusi muvaffaqiyatli yangilandi!")


# ----------------- 11. 24/7 HEALTH CHECK SERVER -----------------
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Render.com 24/7 web-xizmati uchun salomatlik tekshiruvi (Health Check)."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Anvar AI Bot is running 24/7!")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def start_health_server():
    """Render.com serverida 24/7 ishlab turishi uchun fon rejimida mini-server ishga tushirish."""
    port_str = os.getenv("PORT")
    if port_str:
        try:
            port = int(port_str)
            server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            logger.info(f"Render uchun Health Check serveri {port}-portda ishga tushdi.")
        except Exception as e:
            logger.warning(f"Health serverni ishga tushirishda xatolik: {e}")


# ----------------- 12. ASOSIY ISHGA TUSHIRISH (MAIN) -----------------
def main():
    print("🛡️ Bot himoya tizimi (Anti-Spam CAPTCHA + Rate Limiter) bilan ishga tushirilmoqda...")

    # Baza jadvallarini tayyorlash
    init_db()

    # Render.com uchun fon serverini yoqish
    start_health_server()

    request_config = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request_config)
        .post_init(post_init)
        .build()
    )

    # Buyruqlarni ro'yxatdan o'tkazish
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("reset", clear_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("admin", admin_command))

    # CAPTCHA tugmalarini tekshirish handleri
    application.add_handler(CallbackQueryHandler(captcha_callback_handler, pattern=r"^cap:"))

    # Matnli xabarlarni tutib olish
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )

    # Xatoliklarni boshqarish
    application.add_error_handler(error_handler)

    print("✅ Bot muvaffaqiyatli ishga tushdi! Xabarlar kutilmoqda...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
