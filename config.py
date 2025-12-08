"""
Configuration settings for Call Analyzer
"""
import os

# OpenAI API - set via environment variable or .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "knowledge.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Transcription source folder (on server)
TRANSCRIPTION_FOLDER = "/var/www/whisper/completed"

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
    "product"
]

# Analysis prompt for OpenAI
ANALYSIS_PROMPT = """Проанализируй транскрипцию телефонного разговора службы поддержки.

Извлеки и структурируй следующую информацию:

1. КОНТАКТЫ: имена, телефоны, email, адреса упомянутые в разговоре
2. ВОПРОСЫ: что спрашивал клиент (дословно или суть)
3. ОТВЕТЫ: что отвечал оператор на каждый вопрос
4. ПРОБЛЕМЫ: с чем столкнулся клиент
5. РЕШЕНИЯ: как решили или предложили решить проблему
6. ДОГОВОРЁННОСТИ: что обещали, сроки, следующие шаги
7. ПРОДУКТЫ/УСЛУГИ: что обсуждали, заказывали
8. НАСТРОЕНИЕ: доволен ли клиент (positive/neutral/negative)

Верни результат ТОЛЬКО в формате JSON (без markdown, без ```):
{
  "contacts": [{"type": "phone/email/name/address", "value": "...", "person": "client/operator"}],
  "questions": [{"text": "текст вопроса", "topic": "тема"}],
  "answers": [{"question": "вопрос", "answer": "ответ оператора"}],
  "problems": [{"description": "описание проблемы", "severity": "low/medium/high"}],
  "solutions": [{"problem": "проблема", "solution": "предложенное решение"}],
  "agreements": [{"what": "что договорились", "when": "когда/срок", "who": "кто ответственный"}],
  "products": [{"name": "название", "action": "discussed/ordered/returned/etc"}],
  "sentiment": "positive/neutral/negative",
  "summary": "краткое резюме разговора в 2-3 предложения на русском"
}

Если какая-то категория пустая, верни пустой массив [].
Извлекай только то, что явно упомянуто в разговоре.

ТРАНСКРИПЦИЯ:
"""
