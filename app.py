"""
Knowledge Hub - Flask Web Application
Clustered Q&A system with semantic matching and answer effectiveness tracking
"""
import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from config import FLASK_HOST, FLASK_PORT, DEBUG, UPLOAD_FOLDER, DATA_DIR
import database as db
import analyzer
import watcher
import embeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'knowledge-hub-secret-key-2024'
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
    seconds=300,
    id='transcription_watcher',
    name='Watch for new transcriptions',
    max_instances=1
)
scheduler.start()
logger.info("Scheduler started - running every 5 minutes")

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


# ==================== MAIN ROUTES ====================

@app.route('/')
def index():
    """Dashboard - main page with statistics"""
    stats = db.get_stats()
    clusters = db.get_clusters()
    questions_by_cluster = db.get_questions_by_cluster()
    top_questions = db.get_top_questions(limit=5)
    needs_work = db.get_needs_work_questions(limit=5)
    summary = db.get_summary(days=7)
    recent_docs = db.get_documents(limit=10)

    return render_template('index.html',
                           stats=stats,
                           clusters=clusters,
                           questions_by_cluster=questions_by_cluster,
                           top_questions=top_questions,
                           needs_work=needs_work,
                           summary=summary,
                           recent_docs=recent_docs,
                           processing_status=processing_status)


@app.route('/clusters')
def clusters():
    """All clusters view"""
    clusters_list = db.get_clusters()
    stats = db.get_stats()

    return render_template('clusters.html',
                           clusters=clusters_list,
                           stats=stats)


@app.route('/cluster/<int:cluster_id>')
def cluster_detail(cluster_id):
    """View cluster with subcategories and top questions"""
    cluster_data = db.get_cluster_with_subcategories(cluster_id)
    if not cluster_data:
        flash('Cluster not found', 'error')
        return redirect(url_for('clusters'))

    return render_template('cluster_detail.html',
                           cluster=cluster_data)


@app.route('/subcategory/<int:subcategory_id>')
def subcategory_detail(subcategory_id):
    """View subcategory with all questions"""
    page = int(request.args.get('page', 1))
    sort = request.args.get('sort', 'times_asked')
    per_page = 20
    offset = (page - 1) * per_page

    subcategory = db.get_subcategory(subcategory_id)
    if not subcategory:
        flash('Subcategory not found', 'error')
        return redirect(url_for('clusters'))

    questions = db.get_questions(subcategory_id=subcategory_id, limit=per_page, offset=offset, sort_by=sort)
    total = db.get_questions_count(subcategory_id=subcategory_id)
    total_pages = (total + per_page - 1) // per_page

    return render_template('subcategory_detail.html',
                           subcategory=subcategory,
                           questions=questions,
                           page=page,
                           total_pages=total_pages,
                           total=total,
                           sort=sort)


@app.route('/question/<int:question_id>')
def question_detail(question_id):
    """View single question with all scripts"""
    question_data = db.get_question_detail(question_id)
    if not question_data:
        flash('Question not found', 'error')
        return redirect(url_for('index'))

    return render_template('question_detail.html',
                           question=question_data,
                           best_script=question_data.get('best_script'),
                           other_scripts=question_data.get('other_scripts', []),
                           variants=question_data.get('variants', []))


@app.route('/search')
def search():
    """Search page with semantic search"""
    query = request.args.get('q', '')
    search_type = request.args.get('type', 'semantic')
    results = None

    if query:
        if search_type == 'semantic':
            results = {
                'questions': embeddings.semantic_search(query, limit=20, threshold=0.3),
                'is_semantic': True
            }
        else:
            results = {
                'questions': [dict(q) for q in db.search_questions_text(query)],
                'is_semantic': False
            }

    return render_template('search.html',
                           query=query,
                           results=results,
                           search_type=search_type)


@app.route('/needs-work')
def needs_work():
    """Questions that need work (low effectiveness)"""
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page

    questions = db.get_questions(status='needs_work', limit=per_page, offset=offset, sort_by='times_asked')
    total = db.get_questions_count(status='needs_work')
    total_pages = (total + per_page - 1) // per_page

    return render_template('needs_work.html',
                           questions=questions,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/documents')
def documents():
    """List all documents"""
    status = request.args.get('status')
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page

    docs = db.get_documents(status=status, limit=per_page, offset=offset)
    total = db.get_documents_count(status=status)
    total_pages = (total + per_page - 1) // per_page

    return render_template('documents.html',
                           documents=docs,
                           current_status=status,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/document/<int:doc_id>')
def document_detail(doc_id):
    """View single document"""
    doc = db.get_document(doc_id)
    if not doc:
        flash('Document not found', 'error')
        return redirect(url_for('documents'))

    # Parse analysis result if available
    analysis = None
    if doc['analysis_result']:
        try:
            analysis = json.loads(doc['analysis_result'])
        except:
            pass

    return render_template('document_detail.html',
                           document=doc,
                           analysis=analysis)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload new documents"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        doc_type = request.form.get('doc_type', 'manual_faq')

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

    return render_template('upload.html')


@app.route('/analytics')
def analytics():
    """Analytics page with charts"""
    stats = db.get_stats()
    summary = db.get_summary(days=30)
    questions_by_cluster = db.get_questions_by_cluster()

    # Prepare chart data
    chart_data = {
        'dates': [],
        'calls': [],
        'questions': [],
        'resolved': [],
        'unresolved': []
    }
    for s in reversed(list(summary)):
        chart_data['dates'].append(s['date'])
        chart_data['calls'].append(s['total_calls'])
        chart_data['questions'].append(s['new_questions'])
        chart_data['resolved'].append(s['resolved_count'])
        chart_data['unresolved'].append(s['unresolved_count'])

    return render_template('analytics.html',
                           stats=stats,
                           summary=summary,
                           questions_by_cluster=questions_by_cluster,
                           chart_data=json.dumps(chart_data))


# ==================== API ENDPOINTS ====================

@app.route('/api/stats')
def api_stats():
    """Get current statistics"""
    stats = db.get_stats()
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
    search_type = request.args.get('type', 'semantic')

    if not query:
        return jsonify({'error': 'Query required'}), 400

    if search_type == 'semantic':
        results = embeddings.semantic_search(query, limit=20, threshold=0.3)
        return jsonify({'questions': results, 'is_semantic': True})
    else:
        questions = [dict(q) for q in db.search_questions_text(query)]
        return jsonify({'questions': questions, 'is_semantic': False})


@app.route('/api/autocomplete')
def api_autocomplete():
    """Autocomplete suggestions for search"""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])

    suggestions = db.get_autocomplete_suggestions(query, limit=5)
    return jsonify(suggestions)


@app.route('/api/question/<int:question_id>/answers')
def api_question_answers(question_id):
    """Get answers for a question"""
    answers = db.get_answers(question_id)
    return jsonify([dict(a) for a in answers])


@app.route('/api/answer/<int:answer_id>/feedback', methods=['POST'])
def api_answer_feedback(answer_id):
    """Submit feedback for an answer (legacy)"""
    data = request.get_json()
    helpful = data.get('helpful', False)

    db.update_answer_feedback(answer_id, helpful)

    return jsonify({'success': True})


# ==================== SCRIPT API ENDPOINTS (v3) ====================

@app.route('/api/script/<int:script_id>/feedback', methods=['POST'])
def api_script_feedback(script_id):
    """Submit feedback for a script"""
    data = request.get_json()
    helpful = data.get('helpful', False)

    db.update_script_feedback(script_id, helpful)

    return jsonify({'success': True})


@app.route('/api/script/<int:script_id>/copy')
def api_script_copy(script_id):
    """Get script text for copying (also tracks analytics)"""
    script = db.get_script(script_id)
    if not script:
        return jsonify({'error': 'Script not found'}), 404

    return jsonify({
        'text': script['script_text'],
        'type': script['script_type'],
        'has_steps': bool(script['has_steps'])
    })


@app.route('/api/question/<int:question_id>/scripts')
def api_question_scripts(question_id):
    """Get all scripts for a question"""
    scripts = db.get_scripts(question_id)
    return jsonify([dict(s) for s in scripts])


@app.route('/api/reprocess', methods=['POST'])
def api_reprocess():
    """Reprocess all documents"""
    global processing_status

    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress'})

    processing_status['is_processing'] = True

    try:
        processed, errors, total = analyzer.reprocess_all_documents()
        return jsonify({
            'success': True,
            'processed': processed,
            'errors': errors,
            'total': total
        })
    finally:
        processing_status['is_processing'] = False


@app.route('/api/update-embeddings', methods=['POST'])
def api_update_embeddings():
    """Update embeddings for all questions"""
    try:
        updated = embeddings.update_all_embeddings()
        return jsonify({'success': True, 'updated': updated})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Server error'), 500


# Run initial scan on startup
logger.info("Starting Knowledge Hub...")
watcher.run_initial_scan()


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
