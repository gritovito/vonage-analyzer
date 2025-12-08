"""
Analyzer module - processes documents using OpenAI GPT-4o-mini
Extracts structured information from call transcriptions
"""
import json
import logging
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, ANALYSIS_PROMPT
import database as db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_transcription(content):
    """
    Analyze transcription content using OpenAI
    Returns structured JSON with extracted information
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Ты - эксперт по анализу телефонных разговоров службы поддержки. Извлекай структурированную информацию из транскрипций. Отвечай только валидным JSON."
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
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        # Parse JSON
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
    Process a document: analyze and extract facts
    """
    doc = db.get_document(doc_id)
    if not doc:
        logger.error(f"Document {doc_id} not found")
        return False

    if doc['status'] == 'processed':
        logger.info(f"Document {doc_id} already processed")
        return True

    content = doc['content']
    if not content:
        logger.error(f"Document {doc_id} has no content")
        db.update_document_status(doc_id, 'error')
        return False

    logger.info(f"Processing document {doc_id}: {doc['filename']}")

    # Analyze with OpenAI
    analysis = analyze_transcription(content)
    if not analysis:
        db.update_document_status(doc_id, 'error')
        return False

    # Extract and store facts
    facts_added = 0

    # Process contacts
    for contact in analysis.get('contacts', []):
        db.add_fact(
            doc_id,
            'contact',
            contact.get('type', 'unknown'),
            json.dumps(contact, ensure_ascii=False)
        )
        facts_added += 1

    # Process questions and answers - also add to FAQ
    questions = analysis.get('questions', [])
    answers = analysis.get('answers', [])

    for q in questions:
        db.add_fact(
            doc_id,
            'question',
            q.get('topic', 'general'),
            q.get('text', '')
        )
        facts_added += 1

    for qa in answers:
        db.add_fact(
            doc_id,
            'answer',
            qa.get('question', '')[:100],
            qa.get('answer', '')
        )
        facts_added += 1

        # Add to FAQ
        if qa.get('question') and qa.get('answer'):
            db.add_or_update_faq(qa['question'], qa['answer'], doc_id)

    # Process problems
    for problem in analysis.get('problems', []):
        db.add_fact(
            doc_id,
            'problem',
            problem.get('severity', 'medium'),
            problem.get('description', '')
        )
        facts_added += 1

    # Process solutions
    for solution in analysis.get('solutions', []):
        db.add_fact(
            doc_id,
            'solution',
            solution.get('problem', '')[:100],
            solution.get('solution', '')
        )
        facts_added += 1

    # Process agreements
    for agreement in analysis.get('agreements', []):
        db.add_fact(
            doc_id,
            'agreement',
            agreement.get('when', 'unspecified'),
            json.dumps(agreement, ensure_ascii=False)
        )
        facts_added += 1

    # Process products
    for product in analysis.get('products', []):
        db.add_fact(
            doc_id,
            'product',
            product.get('action', 'discussed'),
            product.get('name', '')
        )
        facts_added += 1

    # Store summary and sentiment as facts
    if analysis.get('summary'):
        db.add_fact(doc_id, 'summary', 'call_summary', analysis['summary'])
        facts_added += 1

    if analysis.get('sentiment'):
        db.add_fact(doc_id, 'sentiment', 'customer_sentiment', analysis['sentiment'])
        facts_added += 1

    # Update document status
    db.update_document_status(doc_id, 'processed')

    # Update daily summary
    db.update_daily_summary(calls=1, facts=facts_added)

    logger.info(f"Document {doc_id} processed: {facts_added} facts extracted")
    return True


def process_pending_documents():
    """
    Process all pending documents
    """
    pending = db.get_documents(status='pending', limit=50)
    processed = 0
    errors = 0

    for doc in pending:
        try:
            if process_document(doc['id']):
                processed += 1
            else:
                errors += 1
        except Exception as e:
            logger.error(f"Error processing document {doc['id']}: {e}")
            errors += 1

    logger.info(f"Batch processing complete: {processed} processed, {errors} errors")
    return processed, errors


def analyze_manual_document(doc_id, doc_type):
    """
    Process manually uploaded documents (instructions, FAQ, etc.)
    Extract relevant information based on document type
    """
    doc = db.get_document(doc_id)
    if not doc:
        return False

    content = doc['content']
    if not content:
        db.update_document_status(doc_id, 'error')
        return False

    if doc_type == 'manual_faq':
        # For FAQ documents, try to extract Q&A pairs
        return process_faq_document(doc_id, content)
    elif doc_type == 'manual_instruction':
        # For instructions, store as knowledge
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
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Извлеки все пары вопрос-ответ из документа. Верни JSON массив."
                },
                {
                    "role": "user",
                    "content": f"""Извлеки все вопросы и ответы из этого FAQ документа.

Верни JSON в формате:
{{"faq": [{{"question": "вопрос", "answer": "ответ"}}]}}

Документ:
{content}"""
                }
            ],
            temperature=0.1,
            max_tokens=3000
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        result = json.loads(result_text)

        for item in result.get('faq', []):
            if item.get('question') and item.get('answer'):
                db.add_or_update_faq(item['question'], item['answer'], doc_id)
                db.add_fact(doc_id, 'question', 'faq', item['question'])
                db.add_fact(doc_id, 'answer', item['question'][:100], item['answer'])

        db.update_document_status(doc_id, 'processed')
        return True

    except Exception as e:
        logger.error(f"Error processing FAQ document: {e}")
        db.update_document_status(doc_id, 'error')
        return False


def process_instruction_document(doc_id, content):
    """
    Process instruction document
    Store key points as facts
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Извлеки ключевые инструкции и правила из документа."
                },
                {
                    "role": "user",
                    "content": f"""Извлеки ключевые инструкции, правила и рекомендации из этого документа.

Верни JSON:
{{"instructions": [{{"topic": "тема", "instruction": "инструкция"}}]}}

Документ:
{content}"""
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

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
        db.update_document_status(doc_id, 'error')
        return False
