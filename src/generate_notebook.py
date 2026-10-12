"""Build an academic walkthrough that reuses the deployed project modules."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate_academic_notebook(output_path=None):
    cells = []
    def md(text):
        cells.append({'cell_type':'markdown', 'metadata':{}, 'source':text.strip().splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type':'code', 'metadata':{}, 'source':text.strip().splitlines(keepends=True), 'execution_count':None, 'outputs':[]})
    md('''# InsightPulse: Sentiment Analysis and Insight Extraction
## 5th semester Artificial Intelligence project
Classify English customer feedback, identify aspect-specific opinions, discover
LDA topics and suggest operational actions. This notebook uses the actual app
modules and saved metrics so the academic explanation matches the live demo.''')
    code('''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.preprocess import clean_text
from src.predictor import SentimentInsightPipeline
from src.train import load_and_clean_data, train_and_evaluate_models
from src.evaluate_real_reviews import evaluate
pipe = SentimentInsightPipeline(ROOT / 'models')
metrics = json.loads((ROOT / 'models/model_metrics.json').read_text(encoding='utf8'))
print('Deployed classifier:', metrics['best_model_name'])''')
    md('''## 1. Data sources and label coverage
UCI Sentiment Labelled Sentences (Kotzias, 2015, DOI 10.24432/C57604,
CC BY 4.0) provides real Amazon, Yelp and IMDb positive/negative sentences.
There are no neutral UCI labels. Neutral examples come from generated customer
reviews and curated teaching examples. The fixed external holdout is excluded
before training, including after text normalization.''')
    code('''data = load_and_clean_data()
display(data.groupby(['source', 'sentiment']).size().unstack(fill_value=0))
print('Deduplicated training/evaluation corpus:', len(data))
display(data[['review_text', 'sentiment', 'source']].sample(8, random_state=42))''')
    md('''## 2. Preprocessing and feature extraction
Expand contractions, retain negation and context words, remove URLs/HTML and
normalize with the same offline Snowball stemmer on every machine. Word TF-IDF
(1–3 grams) captures phrases; character TF-IDF (3–5 grams) captures spelling
patterns. We do not switch preprocessing based on installed WordNet data.''')
    code('''for text in ['The product is good.', 'The product is not good.', 'The app isn’t reliable.']:
    print(text, '->', pipe.preprocess(text))
print(metrics['feature_extraction'])''')
    md('''## 3. Training without test-set selection
Exact normalized duplicates and conflicting labels are removed. The remaining
corpus is split 80/20 with a fixed seed and class stratification. Four algorithms
use the same 5-fold training-only cross-validation. TF-IDF is fitted inside each
fold, and the highest CV macro F1 chooses the classifier. Neither internal nor
external test scores choose the winner. Shared generated templates can still
make internal evaluation easier than real reviews.

The cell below displays saved results. Retraining is optional and should be run
with `python src/train.py`, followed by `python src/evaluate_real_reviews.py`.''')
    code('''comparison = pd.DataFrame(metrics['comparison']).T
display(comparison[['cv_f1_macro_mean', 'cv_f1_macro_std', 'test_accuracy', 'test_f1_macro', 'test_precision_macro', 'test_recall_macro']])
display(Image(filename=str(ROOT / 'models/model_comparison.png')))
display(Image(filename=str(ROOT / 'models/confusion_matrix.png')))''')
    md('''## 4. Separate real-review evaluation
The same 594 excluded real reviews compare the original and revised models.
This test covers positive and negative reviews only and does not establish
accuracy on real neutral feedback. Confidence is a model estimate, not accuracy.
Do not infer that any test-corpus score guarantees performance on all uploads.''')
    code('''baseline = json.loads((ROOT / 'models/baseline_real_review_evaluation.json').read_text())
real = json.loads((ROOT / 'models/real_review_evaluation.json').read_text())
display(pd.DataFrame({'Version':['Previous','Revised'], 'Real holdout accuracy':[baseline['accuracy'],real['accuracy']], 'Samples':[baseline['samples'],real['samples']]}))
display(pd.DataFrame(real['by_source']).T)
print(metrics['limitations'])''')
    md('''## 5. Negation and aspect-level evidence
The classifier predicts overall Positive/Neutral/Negative sentiment. Business
aspects use keyword detection with word boundaries and classify their relevant
clauses. Positive and negative aspects are shown as mixed opinions separately;
a single overall label does not erase the aspect evidence.''')
    code('''for text in ['The product is not good.', 'The delivery was excellent but the app is terrible.']:
    result = pipe.predict_single(text)
    print(text)
    print(result['sentiment'], result['confidence'], 'Mixed aspects:', result['is_mixed'])
    display(pd.DataFrame(result['aspects']).T)''')
    md('''## 6. Batch input quality and recommendations
Single and batch inference share one classifier path. Missing, numeric-only
and unsupported text are retained as skipped rows. Uncertain predictions are
flagged. Percentages use analyzed rows only. Recommendations are heuristic
suggestions, not causal findings or guaranteed business improvements.''')
    code('''sample = pd.DataFrame({'Customer':['A','B','C','D','E'], 'Feedback':[
    'The product is excellent.', 'The package arrived broken and damaged.',
    'The product is average.', None, 'The delivery was excellent but the app is terrible.']})
enriched, analytics = pipe.predict_batch(sample, text_column='Feedback')
display(enriched[['source_row','analyzed_text','predicted_sentiment','confidence','analysis_status','detected_aspects']])
print(json.dumps(analytics, indent=2))''')
    md('''## 7. LDA topic discovery and limitations
LDA learns co-occurring terms from the training corpus. Its topics are hints,
not predefined aspects. No topic is assigned when there are no known topic
terms. English only; sarcasm, unseen domains, ambiguous neutral text and keyword
aspect coverage remain limitations. Further improvement needs independently
labeled feedback from the target application, especially real neutral examples.''')
    code('''display(pd.DataFrame(metrics['lda_topics']))
print('Run the app: python app/app.py')
print('Regression checks: python -m unittest discover -s tests -v')''')
    notebook = {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    output = Path(output_path or ROOT / 'notebooks/sentiment_analysis_and_insights.ipynb')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(f'Notebook generated: {output}')


if __name__ == '__main__':
    generate_academic_notebook()
