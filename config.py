"""
Configuration settings for Call Analyzer
"""
import os

# Load .env file if exists
from dotenv import load_dotenv
load_dotenv()

# OpenAI API - set via environment variable or .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_PATH = os.path.join(DATA_DIR, "knowledge.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Transcription source folder (on server)
TRANSCRIPTION_FOLDER = "/var/www/whisper/outputs"

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
DEBUG = False

# Watcher settings
WATCH_INTERVAL_SECONDS = 300  # 5 minutes

# Document types
DOC_TYPES = {
    "transcription": "Transcription",
    "manual_faq": "FAQ Document",
    "manual_instruction": "Instruction",
    "manual_knowledge": "Knowledge Base"
}

# Fact categories
FACT_CATEGORIES = [
    "contact",
    "question",
    "answer",
    "problem",
    "solution",
    "agreement",
    "sentiment"
]

# Analysis prompt for OpenAI
ANALYSIS_PROMPT = """Проанализируй транскрипцию телефонного разговора службы поддержки.

Извлеки:
1. КОНТАКТЫ: имена, телефоны, email, адреса (кто звонил, кому)
2. ВОПРОС КЛИЕНТА: главный вопрос или проблема
3. РЕШЕНИЕ: как оператор решил или предложил решить
4. ДОГОВОРЁННОСТИ: что обещали, сроки
5. ИТОГ: доволен ли клиент (positive/neutral/negative)

Ответ ТОЛЬКО JSON без markdown:
{
  "contacts": [{"type": "имя/телефон/email", "value": "...", "role": "клиент/оператор"}],
  "question": "главный вопрос клиента",
  "answer": "ответ/решение оператора",
  "problem": "описание проблемы если есть",
  "solution": "как решили",
  "agreements": "что договорились",
  "sentiment": "positive/neutral/negative",
  "summary": "краткое резюме в 1-2 предложения"
}

Если данных нет - используй пустую строку "" или пустой массив [].

ТРАНСКРИПЦИЯ:
"""
