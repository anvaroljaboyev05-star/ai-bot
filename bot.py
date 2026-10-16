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

# Kutubxonalardagi ogohlantirishlarni yashirish
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from groq import AsyncGroq
from telegram import (
    Update,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
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

# ----------------- 1. MUHIT O'ZGARUVCHILARI -----------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    print("XATO: TELEGRAM_BOT_TOKEN topilmadi! Iltimos, .env faylini tekshiring.")
    sys.exit(1)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ----------------- 2. AI DUAL-ENGINE SOZLAMALARI -----------------
# 1-Dvigatel: Groq (Ultra-tezkor Llama & GPT-OSS modellar)
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        logger.info("Groq AI mijozi muvaffaqiyatli ishga tushirildi.")
    except Exception as e:
        logger.warning(f"Groq mijozini yaratishda xato: {e}")

GROQ_MODELS = [
    "openai/gpt-oss-120b",  # 120-Billion parameter kuchli aql va tezlik
    "openai/gpt-oss-20b",   # Ultra-tezkor zaxira
    "qwen/qwen3.8-27b",     # Qo'shimcha zaxira
]

# 2-Dvigatel: Google Gemini (Zaxira dvigatel)
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Google Gemini AI muvaffaqiyatli sozlandi.")
    except Exception as e:
        logger.warning(f"Gemini sozlashda xato: {e}")

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

SYSTEM_INSTRUCTION = """
Siz aqlli, har tomonlama yordam beruvchi, xushmuomala va dunyodagi eng ilg'or sun'iy intellekt (AI) yordamchisisiz.

Qat'iy amal qilinishi shart bo'lgan qoidalar:
1. Tilga moslashish: Foydalanuvchi qaysi tilda murojaat qilsa (o'zbek, rus, ingliz va h.k.), aynan o'sha tilda mukammal, ravon va grammatik to'g'ri javob bering.
2. Har tomonlama bilim: Dasturlash (barcha tillarda toza kod yozish va xatolarni tushuntirib to'g'irlash), matematika, fizika, tarjima, maqola/xat yozish, ilmiy va kundalik hayotiy maslahatlar bo'yicha aniq va mantiqiy javoblar bering.
3. Yaratuvchi va Admin haqida: Agar foydalanuvchi "seni kim yaratgan?", "muallifing kim?", "admin kim?", "admin bilan qanday bog'lansa bo'ladi?" deb so'rasa, "Meni O'ljaboyev Anvar yaratgan. Admin: @Oljaboyev_Anvar_2005 (https://t.me/Oljaboyev_Anvar_2005)" deb javob bering.
4. AI ekanlikni yashirmaslik: Siz sun'iy intellektsiz (AI). Buni aslo yashirmang. O'ljaboyev Anvar tomonidan yaratilgan va sozlangan kuchli AI yordamchi ekanligingizni e'tirof eting.
5. Dizayn va Ko'rinish (MUHIM):
   - Javoblarni Telegramda chiroyli, tartibli va ko'rkam qilib taqdim eting.
   - Ro'yxatlar tuzishda keraksiz yulduzcha (*) va panjaralarni (#) ko'paytirmang. Bandlarni chiroyli belgilardan (🔹, 📌, 1., 2.) foydalanib yozing.
   - Har qanday dasturlash kodini albatta alohida blok qilib (```dasturlash_tili\nkod\n```) ichida yozing. Kod boshqa matnlardan yaqqol ajralib, alohida chiroyli kod qutisi sifatida turishi shart.
"""


async def ask_groq_ai(user_text: str, history: list) -> str:
    """1-Dvigatel: Groq orqali ultra-tezkor javob olish."""
    if not groq_client:
        raise ValueError("Groq mijozi sozlanmagan")

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for msg in history:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("parts", [""])[0] if "parts" in msg else msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})

    last_err = None
    for model_name in GROQ_MODELS:
        try:
            resp = await groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
            )
            ans = resp.choices[0].message.content
            if ans and ans.strip():
                return ans.strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Groq {model_name} xato berdi ({e}), keyingisiga o'tilmoqda...")
            continue
    raise last_err or Exception("Groq modellari javob bera olmadi")


async def ask_gemini_ai(user_text: str, history: list) -> str:
    """2-Dvigatel (Zaxira): Google Gemini AI."""
    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(user_text)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Gemini {model_name} xato berdi ({e}), keyingisiga o'tilmoqda...")
            continue
    raise last_err or Exception("Gemini modellari javob bera olmadi")


async def ask_dual_engine(user_text: str, history: list) -> tuple[str, str]:
    """
    Kombinatsiya: Avval Groq (0.3 soniyada javob), agar muammo bo'lsa Gemini!
    Qaytaradi: (javob_matni, ishlatilgan_dvigatel_nomi)
    """
    # 1. Avval Groq AI orqali harakat qilamiz
    if groq_client:
        try:
            ans = await ask_groq_ai(user_text, history)
            return ans, "Groq (GPT-120B) ⚡️"
        except Exception as e:
            logger.warning(f"Groq ishlamadi ({e}), zaxira Gemini dvigateliga o'tilmoqda...")

    # 2. Agar Groq bo'lmasa yoki xato bersa, Gemini ga o'tamiz
    ans = await ask_gemini_ai(user_text, history)
    return ans, "Google Gemini ♊️"


# ----------------- 3. BAZA VA CAPTCHA XAVFSIZLIK TIZIMI -----------------
DB_PATH = "bot_users.db"


def init_db():
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


verified_users = load_verified_users()
active_captchas = {}  # user_id -> to'g'ri javob

# ----------------- 4. ANTI-FLOOD & RATE LIMITING -----------------
user_request_times = {}  # user_id -> deque of timestamps
user_blacklist = {}      # user_id -> unban_timestamp (float)

RATE_LIMIT_SECONDS = 1.5
FLOOD_WINDOW_SECONDS = 3.0
FLOOD_MAX_REQUESTS = 5
BAN_DURATION_SECONDS = 600.0  # 10 daqiqa


def check_anti_flood(user_id: int) -> tuple[bool, str]:
    now = time.time()

    if user_id in user_blacklist:
        unban_time = user_blacklist[user_id]
        if now < unban_time:
            return False, "banned"
        else:
            del user_blacklist[user_id]

    if user_id not in user_request_times:
        user_request_times[user_id] = deque(maxlen=10)

    times = user_request_times[user_id]
    while times and (now - times[0] > FLOOD_WINDOW_SECONDS):
        times.popleft()

    times.append(now)

    if len(times) >= FLOOD_MAX_REQUESTS:
        user_blacklist[user_id] = now + BAN_DURATION_SECONDS
        logger.warning(f"FLOOD ATTACK! User {user_id} 10 daqiqaga bloklandi.")
        return False, "flood_ban"

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
            text=f"👉 {opt}",
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
        "🛡️ <b>Xavfsizlik tekshiruvi (Anti-Spam Gate)</b>\n\n"
        "Botdan foydalanish uchun inson ekanligingizni tasdiqlang:\n"
        f"👉 <b>{num1} + {num2} = ?</b>\n\n"
        "<i>To'g'ri javobni pastdagi tugmalardan tanlang:</i>"
    )
    return text, keyboard


# ----------------- 5. XOTIRA VA TELEGRAM FORMATLASH -----------------
MAX_HISTORY_LENGTH = 20
user_chat_histories = {}
user_last_prompts = {}  # user_id -> oxirgi savoli (qayta yozish uchun)


def get_user_history(user_id: int) -> deque:
    if user_id not in user_chat_histories:
        user_chat_histories[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)
    return user_chat_histories[user_id]


def escape_tg_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def convert_markdown_tables(text: str) -> str:
    """Markdown jadvallarni Telegram uchun chiroyli ro'yxat formatiga aylantirish."""
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Jadval qatorini aniqlash: | ... | ... | formatida
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 3:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            # Jadval sarlavhasi va ma'lumotlarini ajratib olish
            headers = []
            data_rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                # Ajratuvchi qatorni tashlab yuborish (--- | --- | ---)
                if all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                if not headers:
                    headers = cells
                else:
                    data_rows.append(cells)

            if headers and data_rows:
                # Chiroyli formatda chiqarish
                for idx, row in enumerate(data_rows, 1):
                    entry_parts = []
                    for h, val in zip(headers, row):
                        h_clean = h.strip()
                        val_clean = val.strip()
                        if h_clean and val_clean and h_clean.lower() not in ("№", "#", "no"):
                            entry_parts.append(f"<b>{h_clean}:</b> {val_clean}")
                        elif h_clean.lower() in ("№", "#", "no"):
                            continue
                        elif val_clean:
                            entry_parts.append(val_clean)
                    if entry_parts:
                        result.append(f"📌 <b>{idx}.</b> " + "\n     ".join(entry_parts))
                        result.append("")
            elif headers:
                # Faqat sarlavha bor, ma'lumot yo'q
                result.append(" | ".join(headers))
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def markdown_to_telegram_html(text: str) -> str:
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

    # Jadvallarni chiroyli formatga aylantirish (escape_dan OLDIN)
    text = convert_markdown_tables(text)

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
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# ----------------- 6. YANGI ELITA DIZAYN (KLAVIATURALAR) -----------------
def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Doimiy pastki chiroyli menyu tugmalari."""
    keyboard = [
        [KeyboardButton("💬 Yangi mavzu (/clear)"), KeyboardButton("📊 Mening profilim")],
        [KeyboardButton("ℹ️ Bot haqida"), KeyboardButton("👨‍💻 Admin bilan aloqa")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)


def get_response_inline_keyboard() -> InlineKeyboardMarkup:
    """Har bir javob ostidagi tezkor amallar paneli."""
    keyboard = [
        [
            InlineKeyboardButton("🔄 Qayta yozish", callback_data="action:regenerate"),
            InlineKeyboardButton("🗑 Tozalash", callback_data="action:clear"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_welcome_text(first_name: str) -> str:
    return (
        f"Assalomu alaykum, <b>{first_name}</b>!\n\n"
        "Men sizning eng aqlli, tezkor va ko'p qirrali <b>AI Yordamchingizman</b>.\n\n"
        "🚀 <b>Mening asosiy kuchim:</b>\n"
        "⚡ <b>Tezkorlik:</b> Ultra-tezkor sun'iy intellekt texnologiyalari\n"
        "🔹 <b>Dasturlash:</b> Python, C++, JS, HTML va barcha tillarda kod yozish va xatolarni tuzatish\n"
        "🔹 <b>Aniq fanlar:</b> Matematika, fizika, mantiqiy masalalar\n"
        "🔹 <b>Ijod va Tarjima:</b> Maqolalar, insholar, xatlar va professional tarjimalar\n\n"
        "💡 <i>Menga xohlagan tilingizda yozing. Pastdagi tugmalar orqali xotirani tozalashingiz yoki profilingizni ko'rishingiz mumkin!</i>\n\n"
        "Savolingizni yozavering, men tayyorman! 👇"
    )


# ----------------- 7. BUYRUQLAR (COMMANDS) -----------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    user_id = user.id

    allowed, reason = check_anti_flood(user_id)
    if not allowed:
        if reason == "flood_ban":
            await update.message.reply_text("⚠️ Juda ko'p so'rov yubordingiz. Iltimos, 10 daqiqa kuting.")
        return

    # CAPTCHA tekshiruvi
    if user_id not in verified_users:
        text, kb = generate_captcha(user_id)
        await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        return

    first_name = escape_tg_html(user.first_name) if user.first_name else "Do'stim"
    await update.message.reply_text(
        get_welcome_text(first_name),
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(),
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    user_id = user.id

    allowed, reason = check_anti_flood(user_id)
    if not allowed:
        return

    if user_id in user_chat_histories:
        user_chat_histories[user_id].clear()
    user_last_prompts.pop(user_id, None)

    await update.message.reply_text(
        "🗑 <b>Suhbat xotirasi tozalandi!</b>\n"
        "Endi yangi mavzuni bemalol noldan boshlashimiz mumkin.",
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(),
    )


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi profili va tizim holati."""
    user = update.effective_user
    if not user:
        return
    user_id = user.id

    history = get_user_history(user_id)
    msg_count = len(history) // 2

    profile_text = (
        "📊 <b>Sizning profilingiz:</b>\n\n"
        f"👤 <b>Ism:</b> {escape_tg_html(user.full_name)}\n"
        f"🆔 <b>Telegram ID:</b> <code>{user_id}</code>\n"
        f"🛡 <b>Holat:</b> {'✅ Tasdiqlangan (Inson)' if user_id in verified_users else '❌ Tasdiqlanmagan'}\n"
        f"💬 <b>Xotiradagi xabarlar:</b> {msg_count} / 10 ta suhbat\n\n"
        "⚡ <b>AI Tizimi:</b>\n"
        "• <b>Ultra-tezkor sun'iy intellekt</b> ⚡️\n"
        "• Server: <b>24/7 Cloud Engine</b> 🌐\n\n"
        "👨‍💻 <b>Yaratuvchi:</b> O'ljaboyev Anvar (@Oljaboyev_Anvar_2005)"
    )
    await update.message.reply_text(
        profile_text, parse_mode=ParseMode.HTML, reply_markup=get_main_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "ℹ️ <b>Qo'llanma va Imkoniyatlar:</b>\n\n"
        "🔹 <b>Ikki karra quvvat:</b> Bot eng zamonaviy sun'iy intellekt texnologiyalari bilan chaqmoqdek tez ishlaydi.\n"
        "🔹 <b>Xotira:</b> Bot oldingi 20 ta savol-javobni eslab qoladi va mantiqiy suhbat quradi.\n"
        "🔹 <b>Dasturlash:</b> Kodlarni chiroyli qutilarda taqdim etadi, nusxalash oson.\n"
        "🔹 <b>Mavzuni o'zgartirish:</b> '💬 Yangi mavzu' tugmasi orqali xotirani tozalashingiz mumkin.\n\n"
        "👨‍💻 <b>Muallif:</b> O'ljaboyev Anvar\n"
        "👉 <b>Telegram:</b> @Oljaboyev_Anvar_2005"
    )
    await update.message.reply_text(
        help_text, parse_mode=ParseMode.HTML, reply_markup=get_main_keyboard()
    )


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_text = (
        "👨‍💻 <b>Bot administratori va yaratuvchisi:</b>\n"
        "O'ljaboyev Anvar\n\n"
        "👉 Telegram: <b>@Oljaboyev_Anvar_2005</b>\n\n"
        "Savol, taklif yoki hamkorlik bo'lsa, bemalol yozishingiz mumkin:"
    )
    keyboard = [
        [InlineKeyboardButton("💬 Anvarga yozish (@Oljaboyev_Anvar_2005)", url="https://t.me/Oljaboyev_Anvar_2005")]
    ]
    await update.message.reply_text(
        admin_text,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ----------------- 8. CAPTCHA VA INLINE AMALLAR -----------------
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    user_id = query.from_user.id

    # CAPTCHA tekshiruvi
    if data.startswith("cap:"):
        parts = data.split(":")
        if len(parts) == 3:
            target_user_id = int(parts[1])
            selected_ans = int(parts[2])

            if user_id != target_user_id:
                await query.answer("⚠️ Bu tekshiruv siz uchun emas!", show_alert=True)
                return

            correct_ans = active_captchas.get(target_user_id)
            if correct_ans is not None and selected_ans == correct_ans:
                verified_users.add(target_user_id)
                save_verified_user(
                    target_user_id,
                    first_name=query.from_user.first_name or "",
                    username=query.from_user.username or "",
                )
                active_captchas.pop(target_user_id, None)

                await query.answer("✅ To'g'ri javob!")
                await query.edit_message_text(
                    "✅ <b>Tabriklaymiz, siz inson ekanligingiz muvaffaqiyatli tasdiqlandi!</b>",
                    parse_mode=ParseMode.HTML,
                )

                first_name = escape_tg_html(query.from_user.first_name) if query.from_user.first_name else "Do'stim"
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=get_welcome_text(first_name),
                    parse_mode=ParseMode.HTML,
                    reply_markup=get_main_keyboard(),
                )
                logger.info(f"Foydalanuvchi tasdiqlandi: {target_user_id}")
            else:
                await query.answer("❌ Noto'g'ri javob! Yangi misolni yeching.", show_alert=True)
                text, kb = generate_captcha(target_user_id)
                try:
                    await query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
                except Exception:
                    pass
        return

    # Xabar ostidagi tugmalar (action:...)
    if data == "action:clear":
        if user_id in user_chat_histories:
            user_chat_histories[user_id].clear()
        user_last_prompts.pop(user_id, None)
        await query.answer("🗑 Suhbat xotirasi tozalandi!")
        await query.message.reply_text(
            "🗑 <b>Xotira tozalandi!</b> Yangi savolingizni yozavering.",
            parse_mode=ParseMode.HTML,
            reply_markup=get_main_keyboard(),
        )

    elif data == "action:regenerate":
        last_prompt = user_last_prompts.get(user_id)
        if not last_prompt:
            await query.answer("Oxirgi savol topilmadi. Yangidan yozing.", show_alert=True)
            return

        await query.answer("🔄 Yangi javob tayyorlanmoqda...")
        history = get_user_history(user_id)

        # Oxirgi javobni xotiradan olib tashlaymiz
        if len(history) >= 2:
            history.pop()  # model
            history.pop()  # user

        # Qayta javob olish
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            keep_typing_action(query.message.chat_id, context, stop_typing)
        )
        try:
            reply_text, engine = await ask_dual_engine(last_prompt, list(history))
            history.append({"role": "user", "parts": [last_prompt]})
            history.append({"role": "model", "parts": [reply_text]})

            formatted = markdown_to_telegram_html(reply_text)
            await query.message.reply_text(
                formatted,
                parse_mode=ParseMode.HTML,
                reply_markup=get_response_inline_keyboard(),
            )
        except Exception as e:
            logger.error(f"Regenerate xatosi: {e}")
            await query.message.reply_text("Kechirasiz, qayta javob shakllantirib bo'lmadi.")
        finally:
            stop_typing.set()
            await typing_task


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

    # Pastki menyu tugmalari bosilganda:
    if user_text == "💬 Yangi mavzu (/clear)":
        return await clear_command(update, context)
    elif user_text == "📊 Mening profilim":
        return await profile_command(update, context)
    elif user_text == "ℹ️ Bot haqida":
        return await help_command(update, context)
    elif user_text == "👨‍💻 Admin bilan aloqa":
        return await admin_command(update, context)

    # 1. ANTI-FLOOD FILTRI
    allowed, reason = check_anti_flood(user_id)
    if not allowed:
        if reason == "flood_ban":
            await update.message.reply_text("⚠️ Juda ko'p so'rov yubordingiz. Siz 10 daqiqaga cheklandingiz!")
        elif reason == "rate_limit":
            await update.message.reply_text("⚠️ Iltimos, shoshilmang! So'rovlar orasida biroz kuting.")
        return

    # 2. CAPTCHA GATE FILTRI (Spambotlar bu to'siqdan o'ta olmaydi)
    if user_id not in verified_users:
        text, kb = generate_captcha(user_id)
        await update.message.reply_text(
            "🛑 <b>Kechirasiz! Botdan foydalanish uchun avval quyidagi tekshiruvdan o'ting:</b>\n\n" + text,
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )
        return

    # 3. AI DUAL-ENGINE GA YUBORISH (GROQ ⚡️ + GEMINI ♊️)
    user_last_prompts[user_id] = user_text
    history = get_user_history(user_id)

    stop_typing_event = asyncio.Event()
    typing_task = asyncio.create_task(
        keep_typing_action(chat_id, context, stop_typing_event)
    )

    try:
        reply_text, engine_used = await ask_dual_engine(user_text, list(history))

        if not reply_text:
            reply_text = "Kechirasiz, savolingizga javob shakllantirib bo'lmadi. Qaytadan urinib ko'ring."

        history.append({"role": "user", "parts": [user_text]})
        history.append({"role": "model", "parts": [reply_text]})

        chunks = split_message_text(reply_text)
        for i, chunk in enumerate(chunks):
            html_chunk = markdown_to_telegram_html(chunk)

            # Oxirgi bo'lak ostiga tezkor tugmalarni qo'yamiz
            if i == len(chunks) - 1:
                reply_markup = get_response_inline_keyboard()
            else:
                reply_markup = None

            try:
                await update.message.reply_text(
                    html_chunk, parse_mode=ParseMode.HTML, reply_markup=reply_markup
                )
            except Exception as e:
                logger.warning(f"HTML xatosi: {e}, oddiy matn ko'rinishida yuboriladi")
                await update.message.reply_text(chunk, reply_markup=reply_markup)

    except Exception as error:
        logger.error(f"Xatolik yuz berdi: {error}", exc_info=True)
        error_msg = str(error)

        if "429" in error_msg or "ResourceExhausted" in error_msg or "rate" in error_msg.lower():
            user_friendly_error = (
                "⚠️ Hozirda AI serveriga so'rovlar ko'payib ketdi. "
                "Iltimos, 1 daqiqa kuting va qayta yozib ko'ring."
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
    commands = [
        BotCommand("start", "🚀 Botni boshlash"),
        BotCommand("clear", "🗑 Xotirani tozalash"),
        BotCommand("profile", "📊 Mening profilim"),
        BotCommand("help", "ℹ️ Qo'llanma"),
        BotCommand("admin", "👨‍💻 Admin bilan bog'lanish"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Telegram menyusi muvaffaqiyatli yangilandi!")


# ----------------- 11. 24/7 HEALTH CHECK SERVER -----------------
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class HealthCheckHandler(BaseHTTPRequestHandler):
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
    port_str = os.getenv("PORT", "10000")
    try:
        port = int(port_str)
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Render Health Check serveri {port}-portda ishga tushdi.")
    except Exception as e:
        logger.warning(f"Health server xatosi: {e}")


# ----------------- 12. ASOSIY ISHGA TUSHIRISH (MAIN) -----------------
def main():
    # 1. Render Health Check o'tishi uchun darhol fon serverini yoqish
    start_health_server()

    # 2. Baza jadvallarini tayyorlash
    init_db()

    logger.info("Bot Dual-Engine (Groq + Gemini) va Elita Dizayn bilan ishga tushirilmoqda...")

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
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("admin", admin_command))

    # Tugmalar (CAPTCHA va Action tugmalari)
    application.add_handler(CallbackQueryHandler(callback_handler))

    # Matnli xabarlarni tutib olish
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )

    # Xatoliklarni boshqarish
    application.add_error_handler(error_handler)

    logger.info("Bot muvaffaqiyatli ishga tushdi! Xabarlar kutilmoqda...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
