"""
Knowledge Hub Analyzer Module
Processes call transcriptions using OpenAI GPT-4o-mini
Extracts cluster, question, answer with semantic matching
"""
import json
import logging
import time
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, ANALYSIS_PROMPT, CLUSTERS
import database as db
import embeddings

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
    Returns structured JSON with cluster, question, answer, resolution, satisfaction
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
    Process a single document: analyze and extract questions/answers
    Uses semantic matching to merge similar questions into clusters
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

    # Store analysis result
    db.update_document_status(doc_id, 'processing', analysis_result=json.dumps(analysis))

    # Extract data from analysis (new format)
    cluster_name = analysis.get('cluster', 'General Inquiry')
    question_text = analysis.get('question', '')
    answer_text = analysis.get('answer', '')
    resolution = analysis.get('resolution', 'unknown')
    satisfaction = analysis.get('satisfaction', 'neutral')

    if not question_text or not answer_text:
        logger.warning(f"Document {doc_id}: No question or answer extracted")
        db.update_document_status(doc_id, 'processed', analysis_result=json.dumps(analysis))
        db.update_daily_summary(calls=1)
        return True

    # Validate cluster name
    if cluster_name not in CLUSTERS:
        cluster_name = 'General Inquiry'

    # Get cluster ID
    cluster = db.get_cluster_by_name(cluster_name)
    if not cluster:
        # Fallback to General Inquiry
        cluster = db.get_cluster_by_name('General Inquiry')
    cluster_id = cluster['id'] if cluster else 1

    # Find similar existing question using semantic matching
    match_result = embeddings.find_similar_question(question_text)

    new_question = False
    resolved = 1 if resolution == 'resolved' else 0
    unresolved = 1 if resolution == 'unresolved' else 0

    if match_result and match_result.get('question_id'):
        # Found similar question - add variant and new answer
        question_id = match_result['question_id']
        similarity = match_result['similarity']

        logger.info(f"Found similar question (id={question_id}, similarity={similarity:.2%})")

        # Add this question as a variant phrasing
        db.add_question_variant(question_id, question_text, doc_id)

        # Increment times_asked
        db.increment_question_asked(question_id)

    else:
        # No similar question found - create new one
        embedding = match_result.get('embedding') if match_result else None

        question_id = db.add_question(cluster_id, question_text, embedding)
        new_question = True

        logger.info(f"Created new question (id={question_id}) in cluster '{cluster_name}'")

    # Add the answer with effectiveness tracking
    answer_id = db.add_answer(
        question_id=question_id,
        answer_text=answer_text,
        source_document_id=doc_id,
        resolution_status=resolution,
        customer_satisfaction=satisfaction
    )

    logger.info(f"Added answer (id={answer_id}) for question {question_id}")

    # Update document status
    db.update_document_status(doc_id, 'processed', analysis_result=json.dumps(analysis))

    # Update daily summary
    db.update_daily_summary(
        calls=1,
        questions=1 if new_question else 0,
        answers=1,
        resolved=resolved,
        unresolved=unresolved
    )

    logger.info(f"Document {doc_id} processed successfully")
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
    Process manually uploaded documents (FAQ, instructions)
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
    else:
        # For other types, just mark as processed
        db.update_document_status(doc_id, 'processed')
        return True


def process_faq_document(doc_id, content):
    """
    Process uploaded FAQ document
    Extract Q&A pairs and add them as questions/answers
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
                    "content": "Extract all question-answer pairs from the document. Return only valid JSON without markdown. Always respond in English."
                },
                {
                    "role": "user",
                    "content": f"""Extract all questions and answers from this FAQ document.
For each Q&A pair, also determine the cluster category:
- Messaging (SMS, MMS, messages)
- Calls & Voice (call quality, voicemail)
- Data & Internet (slow data, WiFi issues)
- Device Issues (screen, battery, charging)
- Apps & Software (app crashes, updates)
- Account & Billing (payments, plans)
- Store & Service (pickup, repair, warranty)
- General Inquiry (other questions)

Return JSON (without markdown):
{{"faq": [{{"question": "question text", "answer": "answer text", "cluster": "category"}}]}}

Document:
{content}"""
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(lines).strip()

        result = json.loads(result_text)

        for item in result.get('faq', []):
            question_text = item.get('question', '').strip()
            answer_text = item.get('answer', '').strip()
            cluster_name = item.get('cluster', 'General Inquiry')

            if question_text and answer_text:
                # Validate cluster name
                if cluster_name not in CLUSTERS:
                    cluster_name = 'General Inquiry'

                # Get cluster ID
                cluster = db.get_cluster_by_name(cluster_name)
                if not cluster:
                    cluster = db.get_cluster_by_name('General Inquiry')
                cluster_id = cluster['id'] if cluster else 1

                # Check for similar existing question
                match_result = embeddings.find_similar_question(question_text)

                if match_result and match_result.get('question_id'):
                    question_id = match_result['question_id']
                    db.add_question_variant(question_id, question_text, doc_id)
                    db.increment_question_asked(question_id)
                else:
                    embedding = match_result.get('embedding') if match_result else None
                    question_id = db.add_question(cluster_id, question_text, embedding)

                # Add answer (from FAQ, assume it's a good answer)
                db.add_answer(
                    question_id=question_id,
                    answer_text=answer_text,
                    source_document_id=doc_id,
                    resolution_status='resolved',
                    customer_satisfaction='positive'
                )

        db.update_document_status(doc_id, 'processed')
        return True

    except Exception as e:
        logger.error(f"Error processing FAQ document: {e}")
        db.update_document_status(doc_id, 'error', str(e))
        return False


def reprocess_all_documents():
    """
    Reprocess all documents with new analysis logic
    Used for migration or after system updates
    """
    logger.info("Starting reprocessing of all documents...")

    with db.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE documents SET status = 'pending'
            WHERE type = 'transcription' AND content IS NOT NULL
        """)
        updated = cursor.rowcount

    logger.info(f"Reset {updated} documents to pending status")

    # Now process them
    return process_pending_documents()
