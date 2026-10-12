"""Shared, vectorized single/batch sentiment engine with explicit input quality."""
import json
import os
import re
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import clean_text
from src.insights import AspectSentimentAnalyzer, TopicModeler, generate_actionable_recommendations

CLASSES = ['Negative', 'Neutral', 'Positive']


class SentimentInsightPipeline:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.model = self.vectorizer = self.topic_modeler = None
        self.aspect_analyzer = AspectSentimentAnalyzer()
        self.keep_stopwords = False
        self._load_resources()

    def _load_resources(self):
        model, vectorizer = self.models_dir / 'best_sentiment_model.joblib', self.models_dir / 'tfidf_vectorizer.joblib'
        if model.exists() and vectorizer.exists():
            self.model, self.vectorizer = joblib.load(model), joblib.load(vectorizer)
        metadata = self.models_dir / 'model_metrics.json'
        if metadata.exists():
            self.keep_stopwords = json.loads(metadata.read_text(encoding='utf-8')).get('preprocessing') == 'offline_stemmer_keep_stopwords_v2'
        lda = self.models_dir / 'lda_topic_model.joblib'
        if lda.exists():
            self.topic_modeler = TopicModeler.load(lda)

    def is_ready(self):
        return self.model is not None and self.vectorizer is not None

    def preprocess(self, text):
        return clean_text(text, remove_stopwords=not self.keep_stopwords)

    @staticmethod
    def input_issue(value):
        if not isinstance(value, str) or not value.strip():
            return 'Missing or non-text feedback'
        if len(value) > 10000:
            return 'Feedback exceeds 10,000 characters'
        letters = [c for c in value if c.isalpha()]
        if sum(c.isascii() for c in letters) < 2:
            return 'No usable English text'
        if letters and sum(c.isascii() for c in letters) / len(letters) < .7:
            return 'English feedback is required'
        if not clean_text(value, remove_stopwords=False, normalize=False):
            return 'No usable feedback after removing URLs and formatting'
        return None

    def _classify_many(self, texts):
        if not texts:
            return []
        if not self.is_ready():
            raise RuntimeError('Models are missing. Run python src/train.py first.')
        cleaned = [self.preprocess(text) for text in texts]
        features = self.vectorizer.transform(cleaned)
        probabilities = self.model.predict_proba(features)
        labels = self.model.classes_
        # Check known whole words, not character fragments, for unfamiliar input.
        word_features = self.vectorizer.transformer_list[0][1].transform(cleaned) if hasattr(self.vectorizer, 'transformer_list') else features
        results = []
        for index, (text, clean, probs) in enumerate(zip(texts, cleaned, probabilities)):
            best = int(np.argmax(probs))
            ordered = sorted(probs, reverse=True)
            confidence = float(probs[best])
            unfamiliar = word_features[index].nnz == 0
            needs_review = bool(unfamiliar or confidence < .60 or ordered[0] - ordered[1] < .15)
            reason = 'No known review words; check this prediction.' if unfamiliar else 'The model is uncertain; review this prediction.' if needs_review else ''
            results.append({'raw_text': text, 'cleaned_text': clean, 'sentiment': str(labels[best]),
                            'confidence': round(confidence, 4),
                            'probabilities': {str(label): round(float(p), 4) for label, p in zip(labels, probs)},
                            'requires_review': needs_review, 'review_reason': reason})
        return results

    def predict_single(self, raw_text):
        issue = self.input_issue(raw_text)
        if issue:
            raise ValueError(issue)
        result = self._classify_many([raw_text])[0]
        result['aspects'] = self.aspect_analyzer.analyze_many([raw_text], self._classify_many)[0]
        sentiments = {a['sentiment'] for a in result['aspects'].values()}
        result['is_mixed'] = 'Mixed' in sentiments or {'Positive', 'Negative'}.issubset(sentiments)
        result['topic'] = self.topic_modeler.predict_topic(result['cleaned_text']) if self.topic_modeler else None
        return result

    def predict_batch(self, df, text_column='review_text'):
        if text_column not in df.columns:
            raise ValueError('Choose a feedback column that exists in the uploaded file.')
        metadata_columns = {'status', 'rating', 'ratings', 'stars', 'score', 'date', 'email', 'emailaddress',
                            'customername', 'customerid', 'reviewid', 'id', 'assigneddepartment'}
        if re.sub(r'[^a-z0-9]', '', str(text_column).lower()) in metadata_columns:
            raise ValueError(f'"{text_column}" contains metadata, not review text. Choose the Feedback, Review or Comments column.')
        if df.empty:
            raise ValueError('This file has no rows.')
        result_df = df.copy().reset_index(drop=True)
        raw = result_df[text_column].tolist()
        issues = [self.input_issue(t) for t in raw]
        valid_indices = [i for i, issue in enumerate(issues) if issue is None]
        if not valid_indices:
            raise ValueError('The selected column contains no usable English feedback. Choose another column.')
        texts = [raw[i].strip() for i in valid_indices]
        predictions = self._classify_many(texts)
        aspects = self.aspect_analyzer.analyze_many(texts, self._classify_many)
        topics = [None] * len(texts)
        if self.topic_modeler:
            dtm = self.topic_modeler.count_vectorizer.transform([p['cleaned_text'] for p in predictions])
            distributions = self.topic_modeler.lda_model.transform(dtm)
            for index, distribution in enumerate(distributions):
                if dtm[index].nnz:
                    topics[index] = self.topic_modeler.topic_names[int(distribution.argmax())]['label']
        rows = []
        valid = dict(zip(valid_indices, zip(predictions, aspects, topics)))
        counts = {a: 0 for a in self.aspect_analyzer.aspect_lexicon}
        aspect_counts = {label: counts.copy() for label in ['Positive', 'Negative', 'Neutral', 'Mixed']}
        for index, value in enumerate(raw):
            row = {'source_row': index + 2, 'analyzed_text': value if isinstance(value, str) else None,
                   'analysis_status': 'Skipped', 'analysis_note': issues[index], 'predicted_sentiment': None,
                   'confidence': None, 'cleaned_text': None, 'detected_aspects': [], 'aspect_sentiments': {},
                   'dominant_topic': None, 'is_mixed': False}
            if index in valid:
                prediction, aspect, topic = valid[index]
                sentiments = {a['sentiment'] for a in aspect.values()}
                row.update({'analysis_status': 'Needs review' if prediction['requires_review'] else 'Analyzed',
                            'analysis_note': prediction['review_reason'], 'predicted_sentiment': prediction['sentiment'],
                            'confidence': prediction['confidence'], 'cleaned_text': prediction['cleaned_text'],
                            'detected_aspects': list(aspect), 'aspect_sentiments': aspect, 'dominant_topic': topic or 'General',
                            'is_mixed': 'Mixed' in sentiments or {'Positive', 'Negative'}.issubset(sentiments)})
                row.update({f'prob_{label}': prediction['probabilities'][label] for label in CLASSES})
                for name, detail in aspect.items():
                    counts[name] += 1
                    aspect_counts[detail['sentiment']][name] += 1
            rows.append(row)
        # Original columns are retained; derived columns have well-defined names.
        reserved = set(rows[0]) | {f'prob_{label}' for label in CLASSES}
        occupied = set(result_df.columns) | reserved
        renamed = {}
        for column in result_df.columns:
            if column in reserved:
                replacement = f'original_{column}'
                while replacement in occupied:
                    replacement = f'original_{replacement}'
                renamed[column] = replacement
                occupied.add(replacement)
        result_df = result_df.rename(columns=renamed)
        result_df = pd.concat([result_df, pd.DataFrame(rows)], axis=1)
        analyzed = result_df[result_df.analysis_status != 'Skipped']
        sentiment_counts = {label: int((analyzed.predicted_sentiment == label).sum()) for label in CLASSES}
        analytics = {
            'total_rows': len(df), 'total_samples': len(analyzed), 'skipped_rows': len(df) - len(analyzed),
            'needs_review': int((analyzed.analysis_status == 'Needs review').sum()),
            'mixed_reviews': int(analyzed.is_mixed.sum()), 'duplicate_reviews': int(pd.Series(texts).duplicated().sum()),
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': {label: round(100 * count / len(analyzed), 1) for label, count in sentiment_counts.items()},
            'aspect_counts': counts, 'aspect_positive': aspect_counts['Positive'], 'aspect_negative': aspect_counts['Negative'],
            'aspect_neutral': aspect_counts['Neutral'], 'aspect_mixed': aspect_counts['Mixed'],
            'reviews_with_aspects': sum(bool(aspect) for aspect in aspects),
            'reviews_without_aspects': sum(not aspect for aspect in aspects),
            'general_sentiment_counts': {label: sum(not aspect and prediction['sentiment'] == label
                for aspect, prediction in zip(aspects, predictions)) for label in CLASSES},
            'avg_confidence': round(float(analyzed.confidence.mean()), 3),
            'top_topics': analyzed.dominant_topic.value_counts().head(8).to_dict()
        }
        analytics['recommendations'] = generate_actionable_recommendations(analytics)
        return result_df, analytics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='The delivery was quick but the app crashes constantly.')
    print(json.dumps(SentimentInsightPipeline(ROOT / 'models').predict_single(parser.parse_args().text), indent=2))
