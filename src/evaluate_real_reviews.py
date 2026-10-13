"""Evaluate the deployed model on the fixed, excluded real-review holdout."""
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predictor import SentimentInsightPipeline


def evaluate(output):
    data = pd.read_csv(ROOT / 'data/processed/real_review_holdout.csv')
    pipe = SentimentInsightPipeline(str(ROOT / 'models'))
    predictions = np.array([result['sentiment'] for result in pipe._classify_many(data.review_text.tolist())])
    metrics = {'samples': len(data), 'accuracy': accuracy_score(data.sentiment, predictions),
               'macro_f1': f1_score(data.sentiment, predictions, labels=['Negative', 'Positive'], average='macro'),
               'classification_report': classification_report(data.sentiment, predictions, zero_division=0, output_dict=True),
               'by_source': {source: {'samples': int((data.source == source).sum()),
                   'accuracy': accuracy_score(data.loc[data.source == source, 'sentiment'], predictions[data.source == source])} for source in data.source.unique()},
               'engine': pipe.engine_name,
               'note': 'Fixed UCI positive/negative development benchmark, excluded from local SVM training but consulted during hybrid refinement. Upstream pretrained BERT training overlap is unknown. No real neutral labels; this does not establish independent three-class or target-domain accuracy.'}
    Path(output).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k: metrics[k] for k in ('samples', 'accuracy', 'macro_f1', 'by_source')}, indent=2))


if __name__ == '__main__':
    evaluate(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'models/real_review_evaluation.json')
