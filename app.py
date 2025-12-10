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


# ==================== ADMIN ROUTES ====================

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard with statistics"""
    stats = db.get_admin_stats()
    pending_questions = db.get_questions_for_moderation('pending', limit=10)
    recent_log = db.get_moderation_log(limit=10)

    return render_template('admin/dashboard.html',
                           stats=stats,
                           pending_questions=pending_questions,
                           recent_log=recent_log)


@app.route('/admin/moderation')
def admin_moderation():
    """Moderation page for reviewing questions"""
    status_filter = request.args.get('status', 'pending')
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page

    questions = db.get_questions_for_moderation(status_filter, limit=per_page, offset=offset)
    counts = db.get_moderation_counts()
    total = counts.get(status_filter, 0)
    total_pages = (total + per_page - 1) // per_page

    return render_template('admin/moderation.html',
                           questions=questions,
                           counts=counts,
                           status_filter=status_filter,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/admin/rules')
def admin_rules():
    """Filter rules management page"""
    rules = db.get_filter_rules()
    return render_template('admin/rules.html', rules=rules)


@app.route('/admin/merge')
def admin_merge():
    """Question merge tool"""
    source_id = request.args.get('source')
    source_question = None
    similar_questions = []

    if source_id:
        source_question = db.get_question_detail(int(source_id))
        if source_question:
            similar_questions = db.get_similar_questions_for_merge(int(source_id))

    return render_template('admin/merge.html',
                           source_question=source_question,
                           similar_questions=similar_questions)


@app.route('/admin/question/<int:question_id>/edit')
def admin_question_edit(question_id):
    """Question edit page"""
    question = db.get_question_detail(question_id)
    if not question:
        flash('Question not found', 'error')
        return redirect(url_for('admin_moderation'))

    clusters = db.get_clusters()
    subcategories = {}
    for c in clusters:
        subcategories[c['id']] = db.get_subcategories_for_cluster(c['id'])

    return render_template('admin/question_edit.html',
                           question=question,
                           clusters=clusters,
                           subcategories=subcategories)


@app.route('/admin/log')
def admin_log():
    """Moderation log page"""
    page = int(request.args.get('page', 1))
    question_id = request.args.get('question_id')
    per_page = 50
    offset = (page - 1) * per_page

    if question_id:
        question_id = int(question_id)
        log_entries = db.get_moderation_log(limit=per_page, offset=offset, question_id=question_id)
        total = db.get_moderation_log_count(question_id=question_id)
    else:
        log_entries = db.get_moderation_log(limit=per_page, offset=offset)
        total = db.get_moderation_log_count()

    total_pages = (total + per_page - 1) // per_page

    return render_template('admin/log.html',
                           log_entries=log_entries,
                           page=page,
                           total_pages=total_pages,
                           total=total,
                           question_id=question_id)


# ==================== ADMIN API ROUTES ====================

@app.route('/api/admin/question/<int:question_id>/approve', methods=['POST'])
def api_admin_approve_question(question_id):
    """Approve a question"""
    data = request.get_json() or {}
    reason = data.get('reason', '')

    success = db.set_question_moderation_status(question_id, 'approved', reason)
    return jsonify({'success': success})


@app.route('/api/admin/question/<int:question_id>/reject', methods=['POST'])
def api_admin_reject_question(question_id):
    """Reject a question"""
    data = request.get_json() or {}
    reason = data.get('reason', '')

    success = db.set_question_moderation_status(question_id, 'rejected', reason)
    return jsonify({'success': success})


@app.route('/api/admin/question/<int:question_id>/pending', methods=['POST'])
def api_admin_pending_question(question_id):
    """Set question to pending"""
    success = db.set_question_moderation_status(question_id, 'pending')
    return jsonify({'success': success})


@app.route('/api/admin/question/<int:question_id>/delete', methods=['POST'])
def api_admin_delete_question(question_id):
    """Delete a question"""
    data = request.get_json() or {}
    reason = data.get('reason', '')

    success = db.delete_question(question_id, reason)
    return jsonify({'success': success})


@app.route('/api/admin/questions/bulk', methods=['POST'])
def api_admin_bulk_action():
    """Bulk action on multiple questions"""
    data = request.get_json() or {}
    question_ids = data.get('question_ids', [])
    action = data.get('action', '')
    reason = data.get('reason', '')

    if not question_ids:
        return jsonify({'error': 'No questions selected'}), 400

    if action == 'approve':
        count = db.bulk_set_moderation_status(question_ids, 'approved', reason)
    elif action == 'reject':
        count = db.bulk_set_moderation_status(question_ids, 'rejected', reason)
    elif action == 'delete':
        count = db.bulk_delete_questions(question_ids, reason)
    else:
        return jsonify({'error': 'Invalid action'}), 400

    return jsonify({'success': True, 'count': count})


@app.route('/api/admin/question/<int:question_id>/update', methods=['POST'])
def api_admin_update_question(question_id):
    """Update question text and category"""
    data = request.get_json() or {}
    canonical_text = data.get('canonical_text')
    cluster_id = data.get('cluster_id')
    subcategory_id = data.get('subcategory_id')
    moderation_status = data.get('moderation_status')

    if canonical_text:
        db.update_question_text(question_id, canonical_text)

    if cluster_id:
        db.update_question_cluster(question_id, cluster_id, subcategory_id)

    if moderation_status:
        db.set_question_moderation_status(question_id, moderation_status)

    return jsonify({'success': True})


@app.route('/api/admin/script/<int:script_id>/update', methods=['POST'])
def api_admin_update_script(script_id):
    """Update script text"""
    data = request.get_json() or {}
    script_text = data.get('script_text')

    if script_text:
        success = db.update_script_text(script_id, script_text)
        return jsonify({'success': success})
    return jsonify({'error': 'No text provided'}), 400


@app.route('/api/admin/script/<int:script_id>/delete', methods=['POST'])
def api_admin_delete_script(script_id):
    """Delete a script"""
    success = db.delete_script(script_id)
    return jsonify({'success': success})


@app.route('/api/admin/script/<int:script_id>/set-best', methods=['POST'])
def api_admin_set_best_script(script_id):
    """Set script as best"""
    data = request.get_json() or {}
    question_id = data.get('question_id')

    if not question_id:
        return jsonify({'error': 'Question ID required'}), 400

    success = db.set_best_script(question_id, script_id)
    return jsonify({'success': success})


@app.route('/api/admin/variant/<int:variant_id>/delete', methods=['POST'])
def api_admin_delete_variant(variant_id):
    """Delete a question variant"""
    success = db.delete_variant(variant_id)
    return jsonify({'success': success})


@app.route('/api/admin/merge', methods=['POST'])
def api_admin_merge_questions():
    """Merge two questions"""
    data = request.get_json() or {}
    source_id = data.get('source_id')
    target_id = data.get('target_id')

    if not source_id or not target_id:
        return jsonify({'error': 'Both source and target required'}), 400

    success, message = db.merge_questions(source_id, target_id)
    return jsonify({'success': success, 'message': message})


@app.route('/api/admin/rule', methods=['POST'])
def api_admin_add_rule():
    """Add a new filter rule"""
    data = request.get_json() or {}
    rule_type = data.get('rule_type')
    condition_value = data.get('condition_value')
    action = data.get('action', 'auto_reject')
    description = data.get('description', '')

    if not rule_type or not condition_value:
        return jsonify({'error': 'Rule type and condition value required'}), 400

    rule_id = db.add_filter_rule(rule_type, condition_value, action, description)
    return jsonify({'success': True, 'rule_id': rule_id})


@app.route('/api/admin/rule/<int:rule_id>/toggle', methods=['POST'])
def api_admin_toggle_rule(rule_id):
    """Toggle filter rule active status"""
    success = db.toggle_filter_rule(rule_id)
    return jsonify({'success': success})


@app.route('/api/admin/rule/<int:rule_id>/delete', methods=['POST'])
def api_admin_delete_rule(rule_id):
    """Delete a filter rule"""
    success = db.delete_filter_rule(rule_id)
    return jsonify({'success': success})


@app.route('/api/admin/rules/apply', methods=['POST'])
def api_admin_apply_rules():
    """Apply filter rules to all existing questions"""
    updated = db.apply_rules_to_existing_questions()
    return jsonify({'success': True, 'updated': updated})


@app.route('/api/admin/search-questions')
def api_admin_search_questions():
    """Search questions for merge (includes all moderation statuses)"""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])

    results = db.search_questions_text(query, limit=10, approved_only=False)
    return jsonify([{
        'id': r['id'],
        'canonical_text': r['canonical_text'],
        'cluster_name': r['cluster_name'],
        'times_asked': r['times_asked']
    } for r in results])


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
