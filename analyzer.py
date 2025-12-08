"""
Analyzer module - processes documents using OpenAI GPT-4o-mini
Extracts structured information from call transcriptions
"""
import json
import logging
import time
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, ANALYSIS_PROMPT
import database as db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OpenAI API key not set! Analysis will not work.")


def analyze_transcription(content):
    """
    Analyze transcription content using OpenAI
    Returns structured JSON with extracted information
    """
    if not client:
        logger.error("OpenAI client not initialized - API key missing")
        return None

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing customer support phone call transcriptions. Extract structured information from transcriptions. Respond only with valid JSON without markdown formatting. Always respond in English."
                },
                {
                    "role": "user",
                    "content": ANALYSIS_PROMPT + "\n\n" + content
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()

        # Clean up response if it has markdown code blocks
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(lines).strip()

        # Parse JSON
        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response as JSON: {e}")
        logger.error(f"Response was: {result_text[:500]}")
        return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


def process_document(doc_id):
    """
    Process a single document: analyze and extract facts
    """
    doc = db.get_document(doc_id)
    if not doc:
        logger.error(f"Document {doc_id} not found")
        return False

    if doc['status'] == 'processed':
        logger.info(f"Document {doc_id} already processed, skipping")
        return True

    content = doc['content']
    if not content or not content.strip():
        logger.error(f"Document {doc_id} has no content")
        db.update_document_status(doc_id, 'error', 'No content')
        return False

    logger.info(f"Processing document {doc_id}: {doc['filename']}")

    # Analyze with OpenAI
    analysis = analyze_transcription(content)
    if not analysis:
        db.update_document_status(doc_id, 'error', 'OpenAI analysis failed')
        return False

    # Extract and store facts
    facts_added = 0

    # Process contacts
    for contact in analysis.get('contacts', []):
        if contact.get('value'):
            db.add_fact(
                doc_id,
                'contact',
                contact.get('type', 'unknown'),
                f"{contact.get('value')} ({contact.get('role', 'unknown')})"
            )
            facts_added += 1

    # Process main question
    question = analysis.get('question', '')
    if question:
        db.add_fact(doc_id, 'question', 'main', question)
        facts_added += 1

    # Process answer
    answer = analysis.get('answer', '')
    if answer:
        db.add_fact(doc_id, 'answer', 'main', answer)
        facts_added += 1

        # Add to FAQ if we have both question and answer
        if question:
            db.add_or_update_faq(question, answer, doc_id)

    # Process problem
    problem = analysis.get('problem', '')
    if problem:
        db.add_fact(doc_id, 'problem', 'description', problem)
        facts_added += 1

    # Process solution
    solution = analysis.get('solution', '')
    if solution:
        db.add_fact(doc_id, 'solution', 'description', solution)
        facts_added += 1

    # Process agreements
    agreements = analysis.get('agreements', '')
    if agreements:
        db.add_fact(doc_id, 'agreement', 'details', agreements)
        facts_added += 1

    # Process sentiment
    sentiment = analysis.get('sentiment', '')
    if sentiment:
        db.add_fact(doc_id, 'sentiment', 'customer', sentiment)
        facts_added += 1

    # Process summary
    summary = analysis.get('summary', '')
    if summary:
        db.add_fact(doc_id, 'summary', 'call', summary)
        facts_added += 1

    # Update document status
    db.update_document_status(doc_id, 'processed')

    # Update daily summary
    db.update_daily_summary(calls=1, facts=facts_added)

    logger.info(f"Document {doc_id} processed successfully: {facts_added} facts extracted")
    return True


def process_pending_documents():
    """
    Process all pending documents one by one
    Returns tuple (processed_count, error_count, total_pending)
    """
    pending = db.get_pending_documents(limit=100)
    total = len(pending)

    if total == 0:
        logger.info("No pending documents to process")
        return 0, 0, 0

    logger.info(f"Starting processing of {total} pending documents...")

    processed = 0
    errors = 0

    for idx, doc in enumerate(pending, 1):
        logger.info(f"Processing {idx} of {total} documents: {doc['filename']}")

        try:
            if process_document(doc['id']):
                processed += 1
            else:
                errors += 1
        except Exception as e:
            logger.error(f"Error processing document {doc['id']}: {e}")
            db.update_document_status(doc['id'], 'error', str(e))
            errors += 1

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    logger.info(f"Processing complete: {processed} processed, {errors} errors out of {total} total")
    return processed, errors, total


def analyze_manual_document(doc_id, doc_type):
    """
    Process manually uploaded documents (instructions, FAQ, etc.)
    """
    doc = db.get_document(doc_id)
    if not doc:
        return False

    content = doc['content']
    if not content:
        db.update_document_status(doc_id, 'error', 'No content')
        return False

    if doc_type == 'manual_faq':
        return process_faq_document(doc_id, content)
    elif doc_type == 'manual_instruction':
        return process_instruction_document(doc_id, content)
    else:
        # For other knowledge, just mark as processed
        db.update_document_status(doc_id, 'processed')
        return True


def process_faq_document(doc_id, content):
    """
    Process uploaded FAQ document
    Try to extract Q&A pairs using OpenAI
    """
    if not client:
        db.update_document_status(doc_id, 'error', 'OpenAI API key not configured')
        return False

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Extract all question-answer pairs from the document. Return only valid JSON. Always respond in English."
                },
                {
                    "role": "user",
                    "content": f"""Extract all questions and answers from this FAQ document.

Return JSON in format (without markdown):
{{"faq": [{{"question": "question text", "answer": "answer text"}}]}}

Document:
{content}"""
                }
            ],
            temperature=0.1,
            max_tokens=3000
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(lines).strip()

        result = json.loads(result_text)

        for item in result.get('faq', []):
            if item.get('question') and item.get('answer'):
                db.add_or_update_faq(item['question'], item['answer'], doc_id)
                db.add_fact(doc_id, 'question', 'faq', item['question'])
                db.add_fact(doc_id, 'answer', 'faq', item['answer'])

        db.update_document_status(doc_id, 'processed')
        return True

    except Exception as e:
        logger.error(f"Error processing FAQ document: {e}")
        db.update_document_status(doc_id, 'error', str(e))
        return False


def process_instruction_document(doc_id, content):
    """
    Process instruction document
    Store key points as facts
    """
    if not client:
        db.update_document_status(doc_id, 'error', 'OpenAI API key not configured')
        return False

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Extract key instructions and rules from the document. Return only valid JSON. Always respond in English."
                },
                {
                    "role": "user",
                    "content": f"""Extract key instructions, rules and recommendations from this document.

Return JSON (without markdown):
{{"instructions": [{{"topic": "topic", "instruction": "instruction text"}}]}}

Document:
{content}"""
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(lines).strip()

        result = json.loads(result_text)

        for item in result.get('instructions', []):
            db.add_fact(
                doc_id,
                'instruction',
                item.get('topic', 'general'),
                item.get('instruction', '')
            )

        db.update_document_status(doc_id, 'processed')
        return True

    except Exception as e:
        logger.error(f"Error processing instruction document: {e}")
        db.update_document_status(doc_id, 'error', str(e))
        return False
