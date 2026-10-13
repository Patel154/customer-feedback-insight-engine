# InsightPulse — Customer Feedback Intelligence

**5th semester Artificial Intelligence project**

An English customer-feedback analysis dashboard built with Flask, scikit-learn,
NLTK, pandas and local ONNX inference. Analyze a review or upload a feedback file to classify sentiment,
inspect aspect-specific evidence, discover topics and summarize business issues.

## Run the dashboard

On this workspace, a working environment is already available:

```powershell
.\run.ps1
```

Open **http://127.0.0.1:5050**. Keep the terminal running while using the dashboard.
Alternatively: `.\.venv-codex\Scripts\python.exe app\app.py`.

For a new machine, use Python **3.12** and install the locked dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe src\download_context_model.py
.\.venv\Scripts\python.exe app\app.py
```

The locally trained SVM artifacts are included. The pretrained BERT weights need a
one-time download (approximately 168 MB); they are already installed in this workspace.
Predictions run locally without sending uploaded reviews to a service. Retraining the SVM is optional. Pickled scikit-learn
models should be loaded with the same version used to train them; the lock file
records the working environment. Only load model artifacts you trust.

## What the project does

- Positive, Neutral and Negative classification with class probabilities.
- Five predefined business aspects: Product Quality, Customer Support,
  Delivery & Packaging, Pricing & Value, and Usability & UI.
- Clause-level aspect sentiment, including separate evidence for mixed opinions.
- LDA topic discovery and heuristic business recommendations.
- CSV, TSV and Excel `.xlsx` uploads, with worksheet and feedback-column selection.
- Input preview, skipped-row reporting and flags for uncertain predictions.
- Search, sentiment/status filters, pagination and complete CSV export.
- Responsive light/dark dashboard with local fonts, icons and charts. No CDN is
  needed for inference or dashboard rendering.

## Uploading your data

1. Open **Batch insights** and choose a CSV, TSV or `.xlsx` file.
2. For Excel, select the worksheet containing the feedback.
3. Inspect the preview and choose the column with the actual review text.
4. Select **Analyze feedback**.
5. Inspect uncertain/skipped rows and export the complete report if needed.

Column names such as `review_text`, `Review Text`, `Feedback` or `Comments` are
recognized. When multiple review-like columns exist, an explicit choice is
required. Customer names and numeric ratings are never silently chosen instead.
CSV supports UTF-8, UTF-16 and Windows-1252, with comma, tab, semicolon or pipe
delimiters. Upload limits: **16 MB**, **10,000 rows**, and **10,000 characters per
review**. Unsupported or blank rows remain in the exported report and are
excluded from sentiment percentages. Repeated reviews are counted and reported.
Only English is supported; sarcasm and new domains may need manual review.

## Active model and evaluation

The active engine combines a pretrained **BERT product-review classifier** with
the locally trained calibrated Linear SVM. BERT preserves word order, punctuation
and context. Its five star probabilities are mapped to Negative (1-2), Neutral (3)
and Positive (4-5). The SVM supplies specialist evidence for short factual text,
mild criticism and uncertain contextual predictions. Disagreements are flagged.
These combined confidence scores are **uncalibrated estimates**, not accuracy.

BERT is an existing pretrained model, not a model trained from scratch by this
project. Source: [NLP Town model card](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
(MIT); the pinned quantized ONNX conversion is downloaded from Xenova.
Long reviews use overlapping 512-token windows instead of silently truncating
at the first window. Duplicate text is cached locally in memory.

| Fixed UCI benchmark | Correct / total | Accuracy |
| --- | ---: | ---: |
| Local TF-IDF + calibrated SVM | 503 / 594 | 84.7% |
| Active BERT + SVM pipeline | 538 / 594 | 90.6% |

This benchmark contains only positive/negative English Amazon, Yelp and IMDb
reviews. They were excluded from local SVM training, but benchmark errors were
consulted during refinement of the hybrid. Overlap with upstream BERT training
is also **unknown**. Thus 90.6% is a development benchmark result, not an
independent generalization estimate. It does not measure real neutral accuracy
or establish accuracy on arbitrary uploads.
Development examples and user-file corrections are regression checks, not an
independent accuracy estimate. A fresh labeled target-domain test is still needed.
Reproducible results and caveats are in `models/context_evaluation.json`.

The academic baseline compares Naive Bayes, Logistic Regression, calibrated
Linear SVM and Random Forest using 5-fold training-only CV with TF-IDF fitted
inside each fold. The selected SVM scored 0.868 CV macro F1 and 87.5% internal
accuracy. Internal data includes synthetic teaching examples. These baseline
results remain separately displayed on the Model evaluation page.

### Understanding uploaded sheets

The reader detects recognized headers within the first 20 rows and can skip an
Excel cover sheet. It recognizes `Feedback / Comments`, preserves blank rows
and original row numbers, rejects duplicate headers, and skips repeated headers
inside the data. Metadata such as ratings, customer names and business-aspect
labels is not silently selected as review text. Ambiguous columns require a choice.

Aspect detection still uses five predefined English vocabularies with contextual
guards. Tracking belongings, lost keys and airport luggage are no longer treated
as seller delivery problems just because of generic words. Related opinion
clauses retain their subject and contrast. Uncertain aspect predictions also
flag the review. Read the displayed evidence before acting on a recommendation;
aspect coverage and recommendations remain heuristic, not causal findings.

## Reproduce training and tests

```powershell
.\.venv-codex\Scripts\python.exe src\train.py
.\.venv-codex\Scripts\python.exe src\evaluate_context_model.py
.\.venv-codex\Scripts\python.exe -m unittest discover -s tests -v
```

The included `data/raw/training_feedback.csv` and fixed
`data/processed/real_review_holdout.csv` make training independent of network
access. `src/prepare_training_data.py` documents the deterministic source split
and augmentation. It can rebuild these files from the included UCI CSV.
The academic notebook uses the same production modules and saved evaluation
results; it does not maintain a second training implementation.
To open it interactively, install `requirements-notebook.txt` in your environment
and run `python -m jupyterlab notebooks/sentiment_analysis_and_insights.ipynb`.

## Files

| Location | Purpose |
| --- | --- |
| `app/app.py` | Flask prediction, upload-preview, analysis and metrics APIs |
| `app/templates/index.html` | Dashboard structure |
| `app/static/` | Local CSS, JavaScript and vector icon assets |
| `src/feedback_files.py` | CSV/TSV/Excel parsing and column selection |
| `src/preprocess.py` | Offline, consistent English preprocessing |
| `src/predictor.py` | Shared single/batch inference and input-quality checks |
| `src/insights.py` | Clause-level aspects, topics and recommendations |
| `src/train.py` | Leakage-aware model benchmarking and model export |
| `src/evaluate_context_model.py` | Fixed benchmark comparison with explicit pretrained-model limits |
| `src/context_sentiment.py` | Local contextual inference, windowing and caching |
| `src/download_context_model.py` | One-time pinned model download |
| `tests/test_project.py` | Regression tests for analysis and upload behavior |
| `notebooks/` | Reproducible academic walkthrough |

## Dataset attribution

Kotzias, D. (2015). **Sentiment Labelled Sentences**. UCI Machine Learning
Repository. DOI: [10.24432/C57604](https://doi.org/10.24432/C57604).
License: **CC BY 4.0**. Paper: Kotzias, Denil, de Freitas and Smyth,
*From Group to Individual Labels Using Deep Features*, KDD 2015.
See `data/raw/DATA_SOURCES.md` for source, label coverage and evaluation limits.

This is an academic project; Flask's local development server is used for the demo.
