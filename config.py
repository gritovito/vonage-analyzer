"""
Configuration settings for Call Analyzer v2.0
"""
import os

# Load .env file if exists
from dotenv import load_dotenv
load_dotenv()

# OpenAI API - set via environment variable or .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Semantic matching threshold (cosine similarity)
SIMILARITY_THRESHOLD = 0.82

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
    "transcription": "Call Transcription",
    "knowledge_base": "Knowledge Base",
    "manual_faq": "FAQ Document",
    "manual_instruction": "Instruction"
}

# Topics (categories) for questions
TOPIC_CATEGORIES = [
    "Device Issues",
    "Software & Apps",
    "Account & Billing",
    "Connectivity",
    "Store & Pickup",
    "General Inquiry"
]

# Analysis prompt for OpenAI - extracts structured data from transcriptions
ANALYSIS_PROMPT = """Analyze this customer support call transcription.

Extract the following information:

1. TOPIC: Main category of the issue. Choose one:
   - Device Issues (hardware, screen, battery, buttons, physical damage)
   - Software & Apps (updates, settings, applications, OS issues)
   - Account & Billing (payments, subscriptions, account management)
   - Connectivity (network, WiFi, Bluetooth, calls, data issues)
   - Store & Pickup (orders, delivery, store pickup, returns)
   - General Inquiry (other questions)

2. PROBLEM: What issue did the customer have? Generalize it without specific personal details.

3. SOLUTION: How did the operator try to solve it? What advice or actions were taken?

4. RESOLUTION: Was the problem resolved?
   - resolved: Problem was fully solved
   - partial: Problem was partially addressed
   - unresolved: Problem was not solved
   - unknown: Cannot determine from the call

5. SATISFACTION: Customer's apparent satisfaction level
   - positive: Customer seemed happy/satisfied
   - neutral: Customer was neutral
   - negative: Customer was unhappy/frustrated

Return ONLY valid JSON without markdown formatting:
{
  "topic": "category name from list above",
  "problem": "generalized problem description without personal details",
  "problem_keywords": ["keyword1", "keyword2", "keyword3"],
  "solution": "what the operator suggested or did to help",
  "resolution": "resolved/partial/unresolved/unknown",
  "satisfaction": "positive/neutral/negative",
  "summary": "1-2 sentence summary of the call in English"
}

IMPORTANT:
- Generalize the problem (no specific names, phone numbers, order IDs, dates)
- Focus on the TYPE of problem, not specific details
- Solution should be actionable advice that could help others
- All text should be in English

TRANSCRIPTION:
"""
