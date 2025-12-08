"""
File watcher module
Monitors transcription folder for new files and processes them
"""
import os
import logging
from pathlib import Path
from config import TRANSCRIPTION_FOLDER
import database as db
import analyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_transcription_files():
    """
    Get list of transcription files from the watched folder
    Returns list of (filename, filepath) tuples
    """
    files = []
    folder = Path(TRANSCRIPTION_FOLDER)

    if not folder.exists():
        logger.warning(f"Transcription folder does not exist: {TRANSCRIPTION_FOLDER}")
        return files

    # Look for text files (transcriptions)
    for ext in ['*.txt', '*.json']:
        for filepath in folder.glob(ext):
            files.append((filepath.name, str(filepath)))

    logger.info(f"Found {len(files)} files in {TRANSCRIPTION_FOLDER}")
    return files


def scan_for_new_files():
    """
    Scan transcription folder and add new files to database
    Returns number of new files found
    """
    new_files = 0
    files = get_transcription_files()

    for filename, filepath in files:
        # Check if already in database
        existing = db.get_document_by_filename(filename)
        if existing:
            continue

        # Read file content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(filepath, 'r', encoding='cp1251') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {e}")
                continue
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            continue

        # Skip empty files
        if not content or not content.strip():
            logger.warning(f"Skipping empty file: {filename}")
            continue

        # Add to database
        doc_id = db.add_document(
            filename=filename,
            doc_type='transcription',
            content=content,
            status='pending'
        )

        if doc_id:
            logger.info(f"New transcription added: {filename} (id: {doc_id})")
            new_files += 1

    return new_files


def process_new_transcriptions():
    """
    Main function to scan and process new transcriptions
    Called by scheduler every 5 minutes
    """
    logger.info("=" * 50)
    logger.info("Starting scheduled transcription scan...")

    # Scan for new files
    new_count = scan_for_new_files()
    logger.info(f"Found {new_count} new transcriptions")

    # Get pending count before processing
    pending_count = db.get_documents_count(status='pending')

    if pending_count > 0:
        logger.info(f"Processing {pending_count} pending documents...")
        processed, errors, total = analyzer.process_pending_documents()
        logger.info(f"Batch complete: {processed} processed, {errors} errors")
    else:
        logger.info("No pending documents to process")

    logger.info("Scheduled scan complete")
    logger.info("=" * 50)

    return new_count


def run_initial_scan():
    """
    Run initial scan on startup
    Process any existing unprocessed files
    """
    logger.info("=" * 50)
    logger.info("Running initial transcription scan...")

    # Scan for new files first
    new_count = scan_for_new_files()
    logger.info(f"Initial scan found {new_count} new files")

    # Get count of pending documents
    pending_count = db.get_documents_count(status='pending')

    if pending_count > 0:
        logger.info(f"Found {pending_count} pending documents from previous runs")
        logger.info("Processing will happen on next scheduled run or manual trigger")

    logger.info("Initial scan complete")
    logger.info("=" * 50)

    return new_count


if __name__ == "__main__":
    # Test watcher
    print("Testing file watcher...")
    print(f"Watching folder: {TRANSCRIPTION_FOLDER}")
    new_count = process_new_transcriptions()
    print(f"Found and queued {new_count} new files")
