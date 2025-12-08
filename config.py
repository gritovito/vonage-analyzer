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
ANALYSIS_PROMPT = """Analyze this phone call transcription from a customer support center.

Extract:
1. CONTACTS: names, phone numbers, emails, addresses mentioned
2. CUSTOMER QUESTION: main question or issue
3. SOLUTION: how the operator resolved or offered to resolve
4. AGREEMENTS: what was promised, deadlines
5. SENTIMENT: customer satisfaction (positive/neutral/negative)

Return ONLY valid JSON without markdown:
{
  "contacts": [{"type": "name/phone/email", "value": "...", "role": "customer/operator"}],
  "question": "main customer question in English",
  "answer": "operator's response/solution in English",
  "problem": "problem description if any",
  "solution": "how it was resolved",
  "agreements": "what was agreed",
  "sentiment": "positive/neutral/negative",
  "summary": "brief 1-2 sentence summary in English"
}

If data is not available, use empty string "" or empty array [].

TRANSCRIPTION:
"""
