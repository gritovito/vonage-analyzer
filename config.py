"""
Knowledge Hub Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Semantic matching threshold
SIMILARITY_THRESHOLD = 0.82

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_PATH = os.path.join(DATA_DIR, "knowledge_hub.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Transcription source folder
TRANSCRIPTION_FOLDER = "/var/www/whisper/outputs"

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
DEBUG = False

# Watcher settings
WATCH_INTERVAL_SECONDS = 300

# Clusters
CLUSTERS = [
    "Messaging",
    "Calls & Voice",
    "Data & Internet",
    "Device Issues",
    "Apps & Software",
    "Account & Billing",
    "Store & Service",
    "General Inquiry"
]

# Analysis prompt for GPT
ANALYSIS_PROMPT = """Analyze this customer support call transcription.

Determine:

1. CLUSTER: Which category best fits this issue?
   - Messaging (SMS, MMS, messages delayed, not sending)
   - Calls & Voice (call quality, can't make calls, voicemail)
   - Data & Internet (slow data, no internet, WiFi issues)
   - Device Issues (screen, battery, charging, buttons)
   - Apps & Software (app crashes, updates, settings)
   - Account & Billing (payments, plans, account issues)
   - Store & Service (pickup, repair, warranty)
   - General Inquiry (other questions)

2. QUESTION: What is the customer's main question/problem?
   - Write a GENERALIZED version (no specific names, numbers, dates)
   - Example: "Messages arriving late" not "My messages from John came 2 hours late yesterday"

3. ANSWER: What solution did the operator provide?
   - Write the actionable advice given
   - Example: "Clear the storage for your message app in Settings"

4. RESOLUTION: Was the problem resolved?
   - resolved: Problem was fully solved
   - unresolved: Problem was not solved
   - partial: Partially addressed
   - unknown: Cannot determine

5. SATISFACTION: Customer sentiment at the end
   - positive: Customer seemed satisfied
   - neutral: Customer was neutral
   - negative: Customer was frustrated

Return JSON only (no markdown):
{
  "cluster": "category name from list above",
  "question": "generalized question/problem",
  "answer": "solution provided by operator",
  "resolution": "resolved/unresolved/partial/unknown",
  "satisfaction": "positive/neutral/negative"
}

IMPORTANT:
- Generalize the question (no names, phone numbers, dates, order IDs)
- Focus on the TYPE of problem
- Answer should be actionable advice
- All text in English

TRANSCRIPTION:
"""
