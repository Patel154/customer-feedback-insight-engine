"""Reproducible sentiment training with fold-local features and honest holdouts."""
import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import clean_text
from src.insights import TopicModeler
CLASSES = ['Negative', 'Neutral', 'Positive']

def make_features():
    return FeatureUnion([
        ('words', TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=18000, sublinear_tf=True)),
        ('characters', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=14000, sublinear_tf=True))
    ], transformer_weights={'words': 1.0, 'characters': .45})

def load_and_clean_data(data_path=None):
    df = pd.read_csv(data_path or ROOT / 'data/raw/training_feedback.csv').dropna(subset=['review_text', 'sentiment'])
    df = df[df.sentiment.isin(CLASSES)].copy()
    df['cleaned_text'] = df.review_text.map(lambda t: clean_text(t, remove_stopwords=False))
    df = df[df.cleaned_text.str.len() > 0]
    conflicts = df.groupby('cleaned_text').sentiment.nunique()
    df = df[~df.cleaned_text.isin(conflicts[conflicts > 1].index)].drop_duplicates('cleaned_text')
    external = pd.read_csv(ROOT / 'data/processed/real_review_holdout.csv')
    excluded = set(external.review_text.map(lambda t: clean_text(t, remove_stopwords=False)))
    return df[~df.cleaned_text.isin(excluded)].reset_index(drop=True)

def train_and_evaluate_models(data_path=None, output_dir=None):
    output = Path(output_dir or ROOT / 'models')
    output.mkdir(exist_ok=True)
    df = load_and_clean_data(data_path)
    train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df.sentiment)
    for name, split in [('train', train), ('test', test)]:
        split.to_csv(ROOT / f'data/processed/{name}_data.csv', index=False)
    x_train, y_train = train.cleaned_text.tolist(), train.sentiment.to_numpy()
    x_test, y_test = test.cleaned_text.tolist(), test.sentiment.to_numpy()
    print(f'Deduplicated corpus: {len(df)} | training: {len(train)} | internal test: {len(test)}', flush=True)
    candidates = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=.5),
        'Logistic Regression': LogisticRegression(C=4, max_iter=1500, class_weight='balanced', random_state=42),
        'Linear Support Vector Machine': CalibratedClassifierCV(LinearSVC(C=1, class_weight='balanced', random_state=42, max_iter=5000), cv=3),
        'Random Forest': RandomForestClassifier(n_estimators=80, max_depth=32, class_weight='balanced', random_state=42, n_jobs=1)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, fitted = {}, {}
    for name, classifier in candidates.items():
        print(f'Cross-validating {name} (features fitted inside each fold)...', flush=True)
        pipeline = Pipeline([('features', make_features()), ('classifier', classifier)])
        started = time.perf_counter()
        scores = cross_validate(pipeline, x_train, y_train, cv=cv,
                                scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], n_jobs=1)
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        duration = time.perf_counter() - started
        prediction_started = time.perf_counter()
        predicted = pipeline.predict(x_test)
        latency = (time.perf_counter() - prediction_started) * 1000 / len(test)
        results[name] = {
            'cv_accuracy_mean': float(scores['test_accuracy'].mean()), 'cv_accuracy_std': float(scores['test_accuracy'].std()),
            'cv_f1_macro_mean': float(scores['test_f1_macro'].mean()), 'cv_f1_macro_std': float(scores['test_f1_macro'].std()),
            'test_accuracy': accuracy_score(y_test, predicted), 'test_balanced_accuracy': balanced_accuracy_score(y_test, predicted),
            'test_f1_macro': f1_score(y_test, predicted, average='macro'),
            'test_precision_macro': precision_score(y_test, predicted, average='macro', zero_division=0),
            'test_recall_macro': recall_score(y_test, predicted, average='macro', zero_division=0),
            'training_time_sec': round(duration, 3), 'inference_latency_ms': round(latency, 3),
            'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predicted, labels=CLASSES).tolist()
        }
        print(f"  CV F1: {results[name]['cv_f1_macro_mean']:.3f}; internal test accuracy: {results[name]['test_accuracy']:.3f}", flush=True)
    best_name = max(results, key=lambda name: results[name]['cv_f1_macro_mean'])
    winner = fitted[best_name]
    joblib.dump(winner.named_steps['features'], output / 'tfidf_vectorizer.joblib')
    joblib.dump(winner.named_steps['classifier'], output / 'best_sentiment_model.joblib')
    topic = TopicModeler(n_topics=5)
    topic.fit(x_train)
    topic.save(output / 'lda_topic_model.joblib')
    metrics = {
        'best_model_name': best_name, 'classes': CLASSES, 'comparison': results, 'lda_topics': topic.get_topics(),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'preprocessing': 'offline_stemmer_keep_stopwords_v2',
        'feature_extraction': 'Word TF-IDF (1–3 grams) + character TF-IDF (3–5 grams)',
        'selection_method': 'Highest training-only 5-fold macro F1; TF-IDF fitted inside every CV fold.',
        'dataset': {'samples': len(df), 'train_samples': len(train), 'test_samples': len(test),
                    'sources': df.source.value_counts().to_dict(), 'class_counts': df.sentiment.value_counts().to_dict()},
        'limitations': 'Internal evaluation includes generated and curated examples with shared templates. External UCI holdout is real positive/negative English reviews only. Confidence is an estimate, not accuracy. Neutral, sarcasm and new domains need further validation.'
    }
    (output / 'model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(6, 5))
    sns.heatmap(results[best_name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Internal holdout · ' + best_name)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    plt.savefig(output / 'confusion_matrix.png', dpi=180); plt.close()
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(results))
    ax.bar(positions - .18, [r['test_accuracy'] for r in results.values()], .36, label='Internal accuracy', color='#246cdd')
    ax.bar(positions + .18, [r['cv_f1_macro_mean'] for r in results.values()], .36, label='Training CV macro F1', color='#21a59b')
    ax.set_xticks(positions, ['Naive Bayes', 'Logistic Regression', 'Linear SVM', 'Random Forest'])
    ax.set_ylim(0, 1); ax.legend(); fig.tight_layout()
    fig.savefig(output / 'model_comparison.png', dpi=180); plt.close(fig)
    print(f'Saved model: {best_name}', flush=True)
    return results

if __name__ == '__main__':
    train_and_evaluate_models()
