"""
Call Analyzer - Flask Web Application
Self-learning Help Desk system for analyzing call transcriptions
"""
import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from config import (
    FLASK_HOST, FLASK_PORT, DEBUG,
    UPLOAD_FOLDER, DOC_TYPES, FACT_CATEGORIES,
    WATCH_INTERVAL_SECONDS, DATA_DIR
)
import database as db
import analyzer
import watcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'call-analyzer-secret-key-2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'json', 'md', 'pdf', 'doc', 'docx'}

# Processing status for progress tracking
processing_status = {
    'is_processing': False,
    'current': 0,
    'total': 0,
    'current_file': '',
    'last_run': None
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_background_processing():
    """Background job to scan and process files"""
    global processing_status

    if processing_status['is_processing']:
        logger.info("Processing already in progress, skipping scheduled run")
        return

    processing_status['is_processing'] = True
    processing_status['last_run'] = datetime.now().isoformat()

    try:
        watcher.process_new_transcriptions()
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
    finally:
        processing_status['is_processing'] = False
        processing_status['current'] = 0
        processing_status['total'] = 0
        processing_status['current_file'] = ''


# Initialize scheduler for background tasks
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=run_background_processing,
    trigger="interval",
    seconds=WATCH_INTERVAL_SECONDS,
    id='transcription_watcher',
    name='Watch for new transcriptions',
    max_instances=1
)
scheduler.start()
logger.info(f"Scheduler started - running every {WATCH_INTERVAL_SECONDS} seconds")

# Shut down scheduler when app exits
atexit.register(lambda: scheduler.shutdown())


# Template filters
@app.template_filter('datetime')
def format_datetime(value):
    if value is None:
        return ''
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except:
            return value
    return value.strftime('%d.%m.%Y %H:%M')


@app.template_filter('truncate_text')
def truncate_text(text, length=100):
    if not text:
        return ''
    if len(text) <= length:
        return text
    return text[:length] + '...'


# Routes
@app.route('/')
def index():
    """Dashboard - main page with statistics"""
    stats = db.get_total_stats()
    recent_docs = db.get_documents(limit=10)
    facts_by_category = db.get_facts_by_category()
    top_faq = db.get_faq(limit=5)
    summary = db.get_summary(days=7)

    return render_template('index.html',
                           stats=stats,
                           recent_docs=recent_docs,
                           facts_by_category=facts_by_category,
                           top_faq=top_faq,
                           summary=summary,
                           doc_types=DOC_TYPES,
                           processing_status=processing_status)


@app.route('/documents')
def documents():
    """List all documents"""
    doc_type = request.args.get('type')
    status = request.args.get('status')
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page

    docs = db.get_documents(doc_type=doc_type, status=status, limit=per_page, offset=offset)
    total = db.get_documents_count(doc_type=doc_type, status=status)
    total_pages = (total + per_page - 1) // per_page

    return render_template('documents.html',
                           documents=docs,
                           doc_types=DOC_TYPES,
                           current_type=doc_type,
                           current_status=status,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/document/<int:doc_id>')
def document_detail(doc_id):
    """View single document with its facts"""
    doc = db.get_document(doc_id)
    if not doc:
        flash('Document not found', 'error')
        return redirect(url_for('documents'))

    facts = db.get_facts(document_id=doc_id, limit=100)
    return render_template('document_detail.html',
                           document=doc,
                           facts=facts,
                           doc_types=DOC_TYPES)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload new documents"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        doc_type = request.form.get('doc_type', 'manual_knowledge')

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Check if file already exists
            if db.get_document_by_filename(filename):
                flash(f'File {filename} already exists', 'error')
                return redirect(request.url)

            # Read content
            try:
                content = file.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    file.seek(0)
                    content = file.read().decode('cp1251')
                except:
                    flash('Could not read file. Please upload a text file.', 'error')
                    return redirect(request.url)

            # Save to database
            doc_id = db.add_document(
                filename=filename,
                doc_type=doc_type,
                content=content,
                status='pending'
            )

            if doc_id:
                # Process immediately
                analyzer.analyze_manual_document(doc_id, doc_type)
                flash(f'File {filename} uploaded and processed successfully', 'success')
            else:
                flash('Error saving document', 'error')

            return redirect(url_for('documents'))

        flash('Invalid file type', 'error')
        return redirect(request.url)

    return render_template('upload.html', doc_types=DOC_TYPES)


@app.route('/faq')
def faq():
    """FAQ page"""
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page
    sort = request.args.get('sort', 'times_asked')

    faq_items = db.get_faq(limit=per_page, offset=offset, sort_by=sort)
    total = db.get_faq_count()
    total_pages = (total + per_page - 1) // per_page

    return render_template('faq.html',
                           faq_items=faq_items,
                           page=page,
                           total_pages=total_pages,
                           total=total,
                           sort=sort)


@app.route('/facts')
def facts():
    """View all facts by category"""
    category = request.args.get('category')
    page = int(request.args.get('page', 1))
    per_page = 50
    offset = (page - 1) * per_page

    facts_list = db.get_facts(category=category, limit=per_page, offset=offset)
    total = db.get_facts_count(category=category)
    total_pages = (total + per_page - 1) // per_page
    facts_by_category = db.get_facts_by_category()

    return render_template('facts.html',
                           facts=facts_list,
                           categories=FACT_CATEGORIES,
                           facts_by_category=facts_by_category,
                           current_category=category,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/search')
def search():
    """Search page"""
    query = request.args.get('q', '')
    results = None

    if query:
        results = db.search_all(query)

    return render_template('search.html', query=query, results=results)


@app.route('/analytics')
def analytics():
    """Analytics page with charts"""
    stats = db.get_total_stats()
    summary = db.get_summary(days=30)
    facts_by_category = db.get_facts_by_category()

    # Prepare chart data
    chart_data = {
        'dates': [],
        'calls': [],
        'facts': []
    }
    for s in reversed(list(summary)):
        chart_data['dates'].append(s['date'])
        chart_data['calls'].append(s['total_calls'])
        chart_data['facts'].append(s['new_facts'])

    return render_template('analytics.html',
                           stats=stats,
                           summary=summary,
                           facts_by_category=facts_by_category,
                           chart_data=json.dumps(chart_data))


# API endpoints
@app.route('/api/stats')
def api_stats():
    """Get current statistics"""
    stats = db.get_total_stats()
    stats['processing'] = processing_status
    return jsonify(stats)


@app.route('/api/process/<int:doc_id>', methods=['POST'])
def api_process_document(doc_id):
    """Manually trigger document processing"""
    doc = db.get_document(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    success = analyzer.process_document(doc_id)
    return jsonify({'success': success})


@app.route('/api/scan', methods=['POST'])
def api_scan():
    """Manually trigger folder scan"""
    global processing_status

    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress', 'status': processing_status})

    # Run scan in current thread (blocking but gives immediate feedback)
    new_count = watcher.scan_for_new_files()

    return jsonify({
        'new_files': new_count,
        'pending': db.get_documents_count(status='pending')
    })


@app.route('/api/process-all', methods=['POST'])
def api_process_all():
    """Manually trigger processing of all pending documents"""
    global processing_status

    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress', 'status': processing_status})

    # First scan for new files
    new_count = watcher.scan_for_new_files()

    # Then process pending
    pending_count = db.get_documents_count(status='pending')

    if pending_count == 0:
        return jsonify({
            'message': 'No pending documents to process',
            'new_files': new_count,
            'pending': 0
        })

    processing_status['is_processing'] = True
    processing_status['total'] = pending_count

    try:
        processed, errors, total = analyzer.process_pending_documents()
        return jsonify({
            'processed': processed,
            'errors': errors,
            'total': total,
            'new_files': new_count
        })
    finally:
        processing_status['is_processing'] = False
        processing_status['current'] = 0
        processing_status['total'] = 0


@app.route('/api/search')
def api_search():
    """Search API endpoint"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query required'}), 400

    results = db.search_all(query)

    # Convert Row objects to dicts
    response = {
        'documents': [dict(d) for d in results['documents']],
        'facts': [dict(f) for f in results['facts']],
        'faq': [dict(f) for f in results['faq']]
    }
    return jsonify(response)


@app.route('/api/faq/search')
def api_faq_search():
    """Search FAQ endpoint"""
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    results = db.search_faq(query)
    return jsonify([dict(r) for r in results])


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Server error'), 500


# Run initial scan on startup
logger.info("Starting Call Analyzer...")
watcher.run_initial_scan()


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
