"""
Knowledge Hub Embeddings Module
Semantic matching using OpenAI text-embedding-3-small
"""
import logging
import math
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, SIMILARITY_THRESHOLD
import database as db

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OpenAI API key not set!")


def get_embedding(text):
    """Get embedding vector for text"""
    if not client:
        logger.error("OpenAI client not initialized")
        return None

    if not text or not text.strip():
        return None

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.strip()
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def find_similar_question(question_text, threshold=None):
    """
    Find existing question similar to the given text

    Returns:
        dict with question_id, similarity, cluster_id, embedding
        question_id is None if no match found
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    new_embedding = get_embedding(question_text)
    if not new_embedding:
        return {'question_id': None, 'similarity': 0, 'cluster_id': None, 'embedding': None}

    existing_questions = db.get_all_questions_with_embeddings()

    if not existing_questions:
        return {'question_id': None, 'similarity': 0, 'cluster_id': None, 'embedding': new_embedding}

    best_match = None
    best_similarity = 0.0

    for question in existing_questions:
        if question['embedding']:
            similarity = cosine_similarity(new_embedding, question['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = question

    if best_match and best_similarity >= threshold:
        return {
            'question_id': best_match['id'],
            'similarity': best_similarity,
            'cluster_id': best_match['cluster_id'],
            'canonical_text': best_match['canonical_text'],
            'embedding': new_embedding
        }

    return {
        'question_id': None,
        'similarity': best_similarity,
        'cluster_id': None,
        'canonical_text': None,
        'embedding': new_embedding
    }


def semantic_search(query_text, limit=10, threshold=0.3):
    """
    Search for similar questions using semantic matching

    Returns:
        List of questions with similarity scores
    """
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []

    existing_questions = db.get_all_questions_with_embeddings()
    if not existing_questions:
        return []

    results = []
    for question in existing_questions:
        if question['embedding']:
            similarity = cosine_similarity(query_embedding, question['embedding'])

            if similarity >= threshold:
                full_question = db.get_question(question['id'])
                if full_question:
                    results.append({
                        'id': question['id'],
                        'canonical_text': question['canonical_text'],
                        'similarity': round(similarity * 100, 1),
                        'cluster_name': full_question['cluster_name'],
                        'cluster_icon': full_question['cluster_icon'],
                        'cluster_color': full_question['cluster_color'],
                        'status': full_question['status'],
                        'times_asked': full_question['times_asked'],
                        'answer_count': full_question['answer_count']
                    })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:limit]


def update_all_embeddings():
    """Update embeddings for questions without one"""
    if not client:
        logger.error("OpenAI client not initialized")
        return 0

    from database import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, canonical_text FROM questions WHERE embedding IS NULL")
        questions = cursor.fetchall()

    updated = 0
    for question in questions:
        embedding = get_embedding(question['canonical_text'])
        if embedding:
            db.update_question_embedding(question['id'], embedding)
            updated += 1
            logger.info(f"Updated embedding for question {question['id']}")

    logger.info(f"Updated {updated} embeddings")
    return updated
