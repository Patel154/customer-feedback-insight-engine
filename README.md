# InsightPulse — Customer Feedback Intelligence

**5th semester Artificial Intelligence project**

An English customer-feedback analysis dashboard built with Flask, scikit-learn,
NLTK and pandas. Analyze a review or upload a feedback file to classify sentiment,
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
.\.venv\Scripts\python.exe app\app.py
```

The trained models are included. Retraining is optional. Pickled scikit-learn
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

## Accuracy and evaluation

The previous classifier relied on a small generated dataset. The revised corpus
adds real English reviews from UCI, short sentiment/negation teaching examples
and explicit neutral examples. Text is normalized with a consistent offline
Snowball stemmer; negations and context words are retained. Features combine
word TF-IDF (1–3 grams) and character TF-IDF (3–5 grams).

The same **594 held-out real reviews**, excluded from revised training and model
selection, were used to compare both versions:

| Evaluation | Previous model | Revised model |
| --- | ---: | ---: |
| Real-review holdout accuracy | 52.0% | 84.7% |
| Amazon subset (196 reviews) | 59.7% | 88.8% |
| Yelp subset (199 reviews) | 46.7% | 87.4% |
| IMDb subset (199 reviews) | 49.7% | 80.9% |

These are measured results for this fixed test corpus, **not a promise of accuracy
on every uploaded file**. UCI labels are positive/negative only, so this external
test does not validate real neutral feedback. Neutral training examples are
generated or curated. The internal benchmark also contains shared synthetic
templates and is reported separately. Prediction confidence is not accuracy.

Four models are compared with 5-fold training-only cross-validation: Naive Bayes,
Logistic Regression, calibrated Linear SVM and Random Forest. TF-IDF is fitted
inside every CV fold. Selection uses training CV macro F1; neither test set is
used to choose the winner. Exact normalized duplicates and conflicting labels
are removed. Internal train/test text and the external holdout do not overlap.

Selected classifier: **calibrated Linear SVM**. Internal test accuracy: **87.5%**;
training CV macro F1: **0.868**. Current values come from
`models/model_metrics.json`, `models/baseline_real_review_evaluation.json` and
`models/real_review_evaluation.json` and are displayed in **Model evaluation**.

Recommendations summarize observed complaint proportions; they are suggestions,
not causal evidence or guarantees of revenue, retention or cost improvements.
Aspect detection uses a predefined English keyword vocabulary; LDA topics are
unsupervised hints, not exact aspect labels.

### Aspect coverage and contextual feedback fixes

The aspect matcher now matches normalized whole words and phrases, including
arrival inflections, damage, instructions and product functionality. The chart
shows recognized-aspect coverage and a separate general/unrecognized count;
zero matches are explained instead of displaying five empty bars. Missing
matches do not establish that a business area has no problems. Reviews may
contribute to more than one aspect, and complaint counts determine display order.

Contrastive teaching examples distinguish defects from undamaged deliveries,
factual arrival statements, and criticism such as "could be better" from praise
such as "could not be better". Comparative "better" is retained during preprocessing.
These fixes pass the regression suite. The fixed real-review benchmark changed
from 85.7% before this refinement to 84.7% afterwards (503/594 correct), while
training CV macro F1 increased from 0.841 to 0.868. This is a domain-specific
improvement with a small external regression, not a universal accuracy gain.
Regression examples are development checks, not an independent accuracy estimate.

## Reproduce training and tests

```powershell
.\.venv-codex\Scripts\python.exe src\train.py
.\.venv-codex\Scripts\python.exe src\evaluate_real_reviews.py
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
| `src/evaluate_real_reviews.py` | Fixed external holdout evaluation |
| `tests/test_project.py` | Regression tests for analysis and upload behavior |
| `notebooks/` | Reproducible academic walkthrough |

## Dataset attribution

Kotzias, D. (2015). **Sentiment Labelled Sentences**. UCI Machine Learning
Repository. DOI: [10.24432/C57604](https://doi.org/10.24432/C57604).
License: **CC BY 4.0**. Paper: Kotzias, Denil, de Freitas and Smyth,
*From Group to Individual Labels Using Deep Features*, KDD 2015.
See `data/raw/DATA_SOURCES.md` for source, label coverage and evaluation limits.

This is an academic project; Flask's local development server is used for the demo.
