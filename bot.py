import os
import sys
import re
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
   - Har qanday dasturlash kodini albatta alohida blok qilib (```dasturlash_tili\nkod\n```) ichida yozing. Kod boshqa matnlardan yaqqol ajralib, alohida chiroyli kod qutisi sifatida turishi shart.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 3 ta eng kuchli va bepul zaxira modellar ro'yxati
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

# 4. Foydalanuvchilarning suhbat tarixini xotirada saqlash
# Har bir foydalanuvchi uchun oxirgi 20 ta xabar (10 ta savol + 10 ta javob)
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
    """
    AI javobidagi Markdown belgilarini Telegram uchun chiroyli HTML ga aylantiradi:
    - ```python ... ``` -> Kodni alohida ajralib turadigan, nusxalash tugmali blok qiladi
    - `inline kod` -> <code>inline kod</code>
    - **matn** -> <b>matn</b> (ortiqcha yulduzchalarni tozalaydi)
    - * yoki - ro'yxat -> chiroyli 🔹 belgilariga aylanadi
    """
    # 1. Kod bloklarini saqlash
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

    # 2. Inline kodlarni saqlash
    inline_codes = []
    def save_inline_code(match):
        code = match.group(1)
        escaped_code = escape_tg_html(code)
        inline_codes.append(f"<code>{escaped_code}</code>")
        return f"@@@INLINECODE{len(inline_codes)-1}@@@"

    text = re.sub(r"`([^`\n]+)`", save_inline_code, text)

    # 3. Asosiy matndagi xavfli belgilarni escape qilish (&, <, >)
    text = escape_tg_html(text)

    # 4. Sarlavhalar (#, ##, ###)
    text = re.sub(r"^(?:#{1,6})\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # 5. Qalin matn: **matn**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # 6. Ro'yxatlar: ro'yxat boshidagi yulduzcha va minuslarni chiroyli 🔹 ga aylantirish
    text = re.sub(r"^[ \t]*[\*\-][ \t]+", "🔹 ", text, flags=re.MULTILINE)

    # 7. Kursiv: *matn* yoki _matn_
    text = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

    # 8. Saqlangan kodlarni joyiga qaytarish
    for idx, tag in enumerate(inline_codes):
        text = text.replace(f"@@@INLINECODE{idx}@@@", tag)
    for idx, tag in enumerate(code_blocks):
        text = text.replace(f"@@@CODEBLOCK{idx}@@@", tag)

    return text


def split_message_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """
    Telegram xabar limiti (4096 belgi) dan oshib ketmasligi uchun
    matnni mantiqiy bo'laklarga ajratadi.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_chunk_size:
            current_chunk += (line + "\n")
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
    """
    AI javob tayyorlayotgan vaqtda Telegramda "yozmoqda..." (typing)
    holatini uzluksiz ko'rsatib turish uchun fon vazifasi.
    """
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# 5. Buyruqlar (Commands)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi /start bosganda salomlashish."""
    user = update.effective_user
    first_name = escape_tg_html(user.first_name) if user and user.first_name else "Do'stim"
    welcome_text = (
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
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi suhbat tarixini tozalash."""
    user_id = update.effective_user.id
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
        "⚡️ <b>AI Dvigateli:</b> Google Gemini 2.5 Flash"
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


# 6. Matnli xabarlarni qayta ishlash
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    history = get_user_history(user_id)

    # Typing holatini fon jarayonida ishga tushirish
    stop_typing_event = asyncio.Event()
    typing_task = asyncio.create_task(
        keep_typing_action(chat_id, context, stop_typing_event)
    )

    try:
        # Gemini uchun formatlangan xotira ro'yxati
        gemini_history = list(history)

        # Zaxira modellardan foydalangan holda javob olish
        reply_text = await ask_gemini_with_fallback(user_text, gemini_history)

        if not reply_text:
            reply_text = "Kechirasiz, savolingizga javob shakllantirib bo'lmadi. Qaytadan urinib ko'ring."

        # Xotirani yangilash (user savoli va bot javobi)
        history.append({"role": "user", "parts": [user_text]})
        history.append({"role": "model", "parts": [reply_text]})

        # Uzun javoblarni 4000 belgilik bo'laklarga ajratib yuborish
        chunks = split_message_text(reply_text)
        for chunk in chunks:
            html_chunk = markdown_to_telegram_html(chunk)
            try:
                # Telegram HTML formatida chiroyli qilib yuboramiz
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
        # Typing holatini to'xtatish
        stop_typing_event.set()
        await typing_task


# 7. Umumiy xatoliklarni ushlovchi handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Telegram xatoligi yuz berdi:", exc_info=context.error)


async def post_init(application: Application) -> None:
    """Telegram menyusidagi barcha keraksiz buyruqlarni tozalab, faqat keraklilarni o'rnatish."""
    commands = [
        BotCommand("start", "🚀 Botni boshlash"),
        BotCommand("clear", "🗑 Xotirani tozalash"),
        BotCommand("help", "📖 Yordam"),
        BotCommand("admin", "👨‍💻 Admin bilan bog'lanish"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Telegram menyusi muvaffaqiyatli yangilandi!")


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
        pass  # Ortiqcha log yozmaslik


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


def main():
    print("Bot ishga tushirilmoqda...")

    # Render.com uchun fon serverini yoqish
    start_health_server()

    # Telegram so'rovlari uchun barqaror timeoutlar (uzilishlarning oldini oladi)
    request_config = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    # Telegram bot ilovasini yig'ish
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

    # Matnli xabarlarni tutib olish
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )

    # Xatoliklarni boshqarish
    application.add_error_handler(error_handler)

    print(" Bot muvaffaqiyatli ishga tushdi! Xabarlar kutilmoqda...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
