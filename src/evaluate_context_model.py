"""Compare the deployed engine with the frozen SVM; never fit on this benchmark."""
import json
import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predictor import SentimentInsightPipeline


def evaluate():
    data = pd.read_csv(ROOT / 'data/processed/real_review_holdout.csv')
    report = {'samples': len(data), 'results': {},
              'note': 'Fixed UCI positive/negative development benchmark, excluded from local SVM training but consulted during hybrid refinement. '
                      'Upstream pretrained BERT training overlap is unknown. No real neutral labels. '
                      'This does not establish independent three-class or target-domain accuracy.'}
    for engine in ['classic', 'context']:
        pipe = SentimentInsightPipeline(ROOT / 'models', engine=engine)
        start = time.monotonic()
        predicted = pd.Series([p['sentiment'] for p in pipe._classify_many(data.review_text.tolist())])
        result = {'accuracy': accuracy_score(data.sentiment, predicted),
                  'macro_f1_binary_labels': f1_score(data.sentiment, predicted, labels=['Negative', 'Positive'], average='macro'),
                  'correct': int((data.sentiment == predicted).sum()),
                  'classification_report': classification_report(data.sentiment, predicted, output_dict=True, zero_division=0),
                  'by_source': {source: {'samples': int((data.source == source).sum()),
                      'accuracy': accuracy_score(data.loc[data.source == source, 'sentiment'], predicted[data.source == source])}
                      for source in data.source.unique()},
                  'elapsed_seconds': round(time.monotonic() - start, 2)}
        report['results'][engine] = result
        print(engine, result['accuracy'], result['correct'], result['elapsed_seconds'], flush=True)
    (ROOT / 'models/context_evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    deployed = {**report['results']['context'], 'samples': len(data), 'note': report['note']}
    (ROOT / 'models/real_review_evaluation.json').write_text(json.dumps(deployed, indent=2), encoding='utf-8')


if __name__ == '__main__':
    evaluate()
