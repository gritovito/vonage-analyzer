"""
Embeddings module for Call Analyzer v2.0
Handles semantic matching using OpenAI text-embedding-3-small
"""
import logging
import math
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, SIMILARITY_THRESHOLD
import database as db

# Setup logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OpenAI API key not set! Embeddings will not work.")


def get_embedding(text):
    """
    Get embedding vector for a text using OpenAI API
    Returns list of floats or None if error
    """
    if not client:
        logger.error("OpenAI client not initialized - API key missing")
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
    """
    Calculate cosine similarity between two vectors
    Returns float between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def find_similar_question(problem_text, threshold=None):
    """
    Find existing question that is semantically similar to the given problem text

    Args:
        problem_text: The problem/question text to match
        threshold: Similarity threshold (default from config)

    Returns:
        dict with 'question_id', 'similarity', 'canonical_text' if found, None otherwise
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    # Get embedding for the new problem
    new_embedding = get_embedding(problem_text)
    if not new_embedding:
        logger.warning("Could not get embedding for problem text")
        return None

    # Get all existing questions with embeddings
    existing_questions = db.get_all_questions_with_embeddings()

    if not existing_questions:
        return None

    best_match = None
    best_similarity = 0.0

    for question in existing_questions:
        if question['embedding']:
            similarity = cosine_similarity(new_embedding, question['embedding'])

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = question

    # Check if best match exceeds threshold
    if best_match and best_similarity >= threshold:
        return {
            'question_id': best_match['id'],
            'similarity': best_similarity,
            'canonical_text': best_match['canonical_text'],
            'embedding': new_embedding  # Return for potential storage
        }

    # No match found, return the embedding for new question creation
    return {
        'question_id': None,
        'similarity': best_similarity if best_match else 0.0,
        'canonical_text': None,
        'embedding': new_embedding
    }


def semantic_search(query_text, limit=10, threshold=0.5):
    """
    Search for questions semantically similar to the query

    Args:
        query_text: Search query
        limit: Maximum number of results
        threshold: Minimum similarity threshold

    Returns:
        List of dicts with question data and similarity scores
    """
    # Get embedding for query
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []

    # Get all questions with embeddings
    existing_questions = db.get_all_questions_with_embeddings()

    if not existing_questions:
        return []

    # Calculate similarities and sort
    results = []
    for question in existing_questions:
        if question['embedding']:
            similarity = cosine_similarity(query_embedding, question['embedding'])

            if similarity >= threshold:
                # Get full question details
                full_question = db.get_question(question['id'])
                if full_question:
                    results.append({
                        'id': question['id'],
                        'canonical_text': question['canonical_text'],
                        'similarity': round(similarity * 100, 1),  # Convert to percentage
                        'topic_name': full_question['topic_name'],
                        'topic_icon': full_question['topic_icon'],
                        'topic_color': full_question['topic_color'],
                        'times_asked': full_question['times_asked']
                    })

    # Sort by similarity descending
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results[:limit]


def update_all_embeddings():
    """
    Update embeddings for all questions that don't have one
    Useful for batch processing or migration
    """
    if not client:
        logger.error("Cannot update embeddings - OpenAI client not initialized")
        return 0

    # Get questions without embeddings
    questions = db.get_all_questions_with_embeddings()
    questions_without_embedding = [q for q in questions if not q.get('embedding')]

    # Also get questions that have None embeddings from DB
    from database import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, canonical_text FROM questions WHERE embedding IS NULL")
        null_embedding_questions = cursor.fetchall()

    updated = 0

    for question in null_embedding_questions:
        embedding = get_embedding(question['canonical_text'])
        if embedding:
            db.update_question_embedding(question['id'], embedding)
            updated += 1
            logger.info(f"Updated embedding for question {question['id']}")

    logger.info(f"Updated {updated} question embeddings")
    return updated


def batch_get_embeddings(texts):
    """
    Get embeddings for multiple texts in a single API call
    More efficient for processing many items

    Args:
        texts: List of text strings

    Returns:
        List of embedding vectors (same order as input)
    """
    if not client:
        logger.error("OpenAI client not initialized")
        return [None] * len(texts)

    if not texts:
        return []

    # Filter out empty texts, keeping track of indices
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)

    if not valid_texts:
        return [None] * len(texts)

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=valid_texts
        )

        # Build result list with None for invalid texts
        results = [None] * len(texts)
        for i, embedding_data in enumerate(response.data):
            original_index = valid_indices[i]
            results[original_index] = embedding_data.embedding

        return results

    except Exception as e:
        logger.error(f"Error getting batch embeddings: {e}")
        return [None] * len(texts)
