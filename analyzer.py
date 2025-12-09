"""
Knowledge Hub Analyzer Module v3 - Response Scripts Library
Processes call transcriptions using OpenAI GPT-4o-mini
Two-stage processing: Classification + Script Extraction
"""
import json
import logging
import time
from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, CLUSTERS,
    CLASSIFICATION_PROMPT, SCRIPT_EXTRACTION_PROMPT, ANALYSIS_PROMPT
)
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


def _clean_json_response(result_text):
    """Clean up JSON response from OpenAI"""
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        result_text = "\n".join(lines).strip()
    return result_text


def analyze_classification(content):
    """
    Stage 1: Classify the transcript - extract cluster and question
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
                    "content": "You are an expert at analyzing customer support phone call transcriptions. Extract structured information. Respond only with valid JSON without markdown formatting. Always respond in English."
                },
                {
                    "role": "user",
                    "content": CLASSIFICATION_PROMPT + "\n\n" + content
                }
            ],
            temperature=0.1,
            max_tokens=500
        )

        result_text = _clean_json_response(response.choices[0].message.content.strip())
        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Classification: Failed to parse JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Classification API error: {e}")
        return None


def extract_scripts(content):
    """
    Stage 2: Extract actual operator scripts from the transcript
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
                    "content": "You are an expert at extracting operator responses from customer support call transcriptions. Extract the EXACT phrases operators use - ready for copy-paste reuse. Respond only with valid JSON without markdown. Always respond in English."
                },
                {
                    "role": "user",
                    "content": SCRIPT_EXTRACTION_PROMPT + "\n\n" + content
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = _clean_json_response(response.choices[0].message.content.strip())
        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Script extraction: Failed to parse JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Script extraction API error: {e}")
        return None


def analyze_transcription(content):
    """
    Legacy single-pass analysis (kept for compatibility)
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

        result_text = _clean_json_response(response.choices[0].message.content.strip())
        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


def process_document(doc_id):
    """
    Process a single document with two-stage analysis:
    Stage 1: Classification (cluster + subcategory + question)
    Stage 2: Script extraction (actual operator responses)
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

    # Stage 1: Classification
    classification = analyze_classification(content)
    if not classification:
        db.update_document_status(doc_id, 'error', 'Classification failed')
        return False

    cluster_name = classification.get('cluster', 'General Inquiry')
    subcategory_name = classification.get('subcategory', 'Other')
    question_text = classification.get('question', '')

    if not question_text:
        logger.warning(f"Document {doc_id}: No question extracted")
        db.update_document_status(doc_id, 'processed', analysis_result=json.dumps(classification))
        db.update_daily_summary(calls=1)
        return True

    # Validate cluster name
    if cluster_name not in CLUSTERS:
        cluster_name = 'General Inquiry'

    # Get cluster ID
    cluster = db.get_cluster_by_name(cluster_name)
    if not cluster:
        cluster = db.get_cluster_by_name('General Inquiry')
    cluster_id = cluster['id'] if cluster else 1

    # Get or create subcategory
    subcategory_id = db.get_or_create_subcategory(cluster_id, subcategory_name)

    # Find or create question using semantic matching
    match_result = embeddings.find_similar_question(question_text)
    new_question = False

    if match_result and match_result.get('question_id'):
        question_id = match_result['question_id']
        similarity = match_result['similarity']
        logger.info(f"Found similar question (id={question_id}, similarity={similarity:.2%})")
        db.add_question_variant(question_id, question_text, doc_id)
        db.increment_question_asked(question_id)
    else:
        embedding = match_result.get('embedding') if match_result else None
        question_id = db.add_question(cluster_id, question_text, embedding, subcategory_id=subcategory_id)
        new_question = True
        logger.info(f"Created new question (id={question_id}) in cluster '{cluster_name}' / subcategory '{subcategory_name}'")

    # Stage 2: Script extraction
    extraction = extract_scripts(content)
    scripts_added = 0

    if extraction and extraction.get('scripts'):
        customer_satisfied = extraction.get('customer_satisfied', False)

        for script_data in extraction['scripts']:
            script_text = script_data.get('text', '').strip()
            if not script_text or len(script_text) < 10:
                continue  # Skip empty or too short scripts

            script_type = script_data.get('type', 'instruction')
            has_steps = script_data.get('has_steps', False)
            resolved = script_data.get('resolved_issue', customer_satisfied)

            # Check for duplicate script
            existing_script = db.find_similar_script(question_id, script_text)

            if existing_script:
                # Update existing script's count
                db.update_script_count(existing_script, resolved)
                logger.info(f"Updated existing script (id={existing_script})")
            else:
                # Add new script
                script_id = db.add_script(
                    question_id=question_id,
                    script_text=script_text,
                    script_type=script_type,
                    has_steps=has_steps,
                    resolved=resolved,
                    source_doc_id=doc_id
                )
                scripts_added += 1
                logger.info(f"Added new script (id={script_id}) for question {question_id}")

        # Recalculate best script
        db.recalculate_best_script(question_id)

    # Store combined analysis result
    combined_analysis = {
        'classification': classification,
        'extraction': extraction,
        'scripts_added': scripts_added
    }
    db.update_document_status(doc_id, 'processed', analysis_result=json.dumps(combined_analysis))

    # Update daily summary
    resolved_count = 1 if extraction and extraction.get('customer_satisfied') else 0
    db.update_daily_summary(
        calls=1,
        questions=1 if new_question else 0,
        scripts=scripts_added,
        resolved=resolved_count,
        unresolved=1 - resolved_count
    )

    logger.info(f"Document {doc_id} processed: {scripts_added} scripts extracted")
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
