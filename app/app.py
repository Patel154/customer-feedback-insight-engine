"""
Flask Application Entrypoint for Customer Feedback Sentiment & Insight Platform
"""

import os
import sys
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predictor import SentimentInsightPipeline
from src.generate_dataset import create_feedback_dataset
from src.feedback_files import read_feedback_file

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = SentimentInsightPipeline(models_dir=os.path.join(BASE_DIR, "models"))
    return pipeline

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or not isinstance(data.get('text'), str):
        return jsonify({'error': 'Please enter a review as text.'}), 400
    text = data['text'].strip()
    try:
        result = get_pipeline().predict_single(text)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({'error': 'File is too large. The upload limit is 16 MB.'}), 413

@app.route('/api/preview_upload', methods=['POST'])
def api_preview_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Choose a feedback file first.'}), 400
    try:
        _, metadata = read_feedback_file(request.files['file'], request.form.get('sheet') or None)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': f'Could not read the file: {e}'}), 400

@app.route("/api/batch_upload", methods=["POST"])
def api_batch_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        df, metadata = read_feedback_file(file, request.form.get('sheet') or None)
        text_col = request.form.get('text_column') or metadata['suggested_column']
        if text_col is None:
            return jsonify({'error': 'Please choose which column contains the customer feedback.', 'columns': metadata['columns']}), 400
        pipe = get_pipeline()
        enriched_df, analytics = pipe.predict_batch(df, text_column=text_col)
        return jsonify({
            "analytics": analytics,
            "reviews": json.loads(enriched_df.to_json(orient='records', date_format='iso')),
            'text_column': text_col,
            'sheet': metadata['sheet']
        })
    except (ValueError, pd.errors.ParserError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process feedback: {str(e)}"}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, filename in [('real_review_evaluation', 'real_review_evaluation.json'), ('baseline_real_review_evaluation', 'baseline_real_review_evaluation.json')]:
            evaluation_path = os.path.join(BASE_DIR, 'models', filename)
            if os.path.exists(evaluation_path):
                with open(evaluation_path, encoding='utf-8') as f:
                    data[key] = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics not found. Please train models first."}), 404

@app.route("/api/download_sample", methods=["GET"])
def api_download_sample():
    sample_path = os.path.join(BASE_DIR, "data", "raw", "customer_feedback.csv")
    if not os.path.exists(sample_path):
        create_feedback_dataset(sample_path)
    return send_file(sample_path, as_attachment=True, download_name="customer_feedback_sample.csv")

@app.route("/static/img/<path:filename>")
def serve_model_images(filename):
    models_dir = os.path.join(BASE_DIR, "models")
    return send_from_directory(models_dir, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting InsightPulse ML Server at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
