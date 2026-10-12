"""
Aspect-Based Sentiment Mining, Topic Modeling (LDA), and Actionable Insights Engine
"""

import re
import joblib
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.preprocess import clean_text

ASPECT_LEXICON = {
    "Product Quality": [
        "quality", "build", "material", "durability", "durable", "sturdy",
        "battery", "screen", "display", "hardware", "performance", "defect",
        "defective", "broke", "broken", "lag", "crash", "overheat", "heat",
        "finish", "casing", "hinge", "pixel", "microphone", "sound", "product", "device", "item",
        "damage", "faulty", "malfunction", "works perfectly", "stopped working", "does not work", "turn on"
    ],
    "Customer Support": [
        "support", "service", "agent", "representative", "rep", "call", "chat",
        "helpdesk", "warranty", "claim", "replacement", "refund", "exchange",
        "ticket", "courteous", "polite", "rude", "unresponsive", "hold", "email"
    ],
    "Delivery & Packaging": [
        "delivery", "shipping", "courier", "package", "packaging", "box",
        "wrap", "transit", "ship", "dispatch", "arrive", "arrival", "late",
        "delay", "delayed", "lost", "crushed", "doorstep", "tracking", "delivered", "parcel", "received"
    ],
    "Pricing & Value": [
        "price", "pricing", "cost", "expensive", "cheap", "value", "money",
        "budget", "affordable", "discount", "deal", "overpriced", "worth",
        "penny", "rip-off", "investment", "economic", "gouging"
    ],
    "Usability & UI": [
        "app", "software", "ui", "interface", "setup", "pair", "pairing",
        "bluetooth", "navigate", "navigation", "button", "control", "ergonomic",
        "ergonomics", "bug", "glitch", "freeze", "display", "setting", "manual", "instructions", "user guide", "easy to use", "hard to use"
    ]
}

class AspectSentimentAnalyzer:
    """Extracts aspect mentions and calculates clause-level sentiment."""
    
    def __init__(self, aspect_lexicon=None):
        self.aspect_lexicon = aspect_lexicon or ASPECT_LEXICON
        self.normalized_keywords = {
            aspect: [(kw, clean_text(kw, remove_stopwords=False)) for kw in keywords]
            for aspect, keywords in self.aspect_lexicon.items()
        }
        
    def detect_aspects(self, text: str):
        """Identifies which predefined business aspects are referenced in the review."""
        # Match complete normalized tokens/phrases: arrived -> arriv, not arriveed.
        # Boundaries still prevent app from matching apple or application fragments.
        normalized = ' ' + clean_text(text, remove_stopwords=False) + ' '
        found_aspects = {}
        
        for aspect, keywords in self.normalized_keywords.items():
            matches = [kw for kw, stem in keywords if ' ' + stem + ' ' in normalized]
            if matches:
                found_aspects[aspect] = list(set(matches))
                
        return found_aspects

    def contexts(self, review_text):
        # Contrast markers work with or without a preceding comma. Split separate
        # subjects joined by "and" so good delivery does not mask a broken app.
        clauses = re.split(
            r'[.!?;\n]+|\b(?:but|however|although|though|whereas|while|yet)\b|'
            r'\band\s+(?=(?:(?:the|my|their|this)\s+)?(?:app|product|device|battery|screen|delivery|shipping|packaging|support|service|price|quality|interface)\b)',
            review_text, flags=re.IGNORECASE)
        contexts = {}
        for clause in clauses:
            if not clause.strip():
                continue
            for aspect, keywords in self.detect_aspects(clause).items():
                entry = contexts.setdefault(aspect, {'keywords': set(), 'clauses': []})
                entry['keywords'].update(keywords)
                entry['clauses'].append(clause.strip(' ,'))
        return contexts

    def analyze_many(self, texts, classify):
        contexts = [self.contexts(text) for text in texts]
        snippets, owners = [], []
        for index, aspects in enumerate(contexts):
            for aspect, info in aspects.items():
                for clause in info['clauses']:
                    snippets.append(clause)
                    owners.append((index, aspect))
        predictions = classify(snippets) if snippets else []
        grouped = defaultdict(list)
        for owner, prediction in zip(owners, predictions):
            grouped[owner].append(prediction)
        output = []
        for index, aspects in enumerate(contexts):
            result = {}
            for aspect, info in aspects.items():
                predictions = grouped[index, aspect]
                labels = {p['sentiment'] for p in predictions}
                mixed = 'Positive' in labels and 'Negative' in labels
                mean = {label: sum(p['probabilities'].get(label, 0) for p in predictions) / len(predictions)
                        for label in ['Negative', 'Neutral', 'Positive']}
                label = 'Mixed' if mixed else max(mean, key=mean.get)
                result[aspect] = {'keywords': sorted(info['keywords']), 'snippet': '. '.join(info['clauses']),
                                  'sentiment': label, 'confidence': None if mixed else round(mean[label], 3)}
            output.append(result)
        return output

    def analyze_aspect_sentiments(self, review_text: str, sentiment_classifier=None, vectorizer=None):
        """Compatibility API, using the same clause extraction as batch analysis."""
        if sentiment_classifier is None or vectorizer is None:
            return {name: {'keywords': sorted(info['keywords']), 'snippet': '. '.join(info['clauses']),
                           'sentiment': 'Detected', 'confidence': None}
                    for name, info in self.contexts(review_text).items()}
        from src.preprocess import clean_text
        def classify(texts):
            keep_stopwords = hasattr(vectorizer, 'transformer_list')
            features = vectorizer.transform([clean_text(text, remove_stopwords=not keep_stopwords) for text in texts])
            probabilities = sentiment_classifier.predict_proba(features)
            classes = sentiment_classifier.classes_
            return [{'sentiment': str(classes[row.argmax()]),
                     'probabilities': {str(label): round(float(p), 4) for label, p in zip(classes, row)}}
                    for row in probabilities]
        return self.analyze_many([review_text], classify)[0]


class TopicModeler:
    """Unsupervised Latent Dirichlet Allocation (LDA) for topic discovery."""
    
    def __init__(self, n_topics=5, max_features=1000, random_state=42):
        self.n_topics = n_topics
        self.count_vectorizer = CountVectorizer(
            max_df=0.90, min_df=2, max_features=max_features, stop_words='english'
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics, random_state=random_state, learning_method='online', max_iter=25
        )
        self.is_fitted = False
        self.topic_names = []

    def fit(self, preprocessed_texts):
        """Fit count vectorizer and LDA on preprocessed texts."""
        dtm = self.count_vectorizer.fit_transform(preprocessed_texts)
        self.lda_model.fit(dtm)
        self.is_fitted = True
        self._generate_topic_labels()
        return self

    def _generate_topic_labels(self):
        """Generates interpretable topic descriptors from top weighted terms."""
        feature_names = self.count_vectorizer.get_feature_names_out()
        self.topic_names = []
        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            label = f"Topic {idx + 1}: {' / '.join(top_words[:3])}"
            self.topic_names.append({"id": idx, "label": label, "keywords": top_words})

    def get_topics(self):
        return self.topic_names

    def predict_topic(self, preprocessed_text):
        """Predicts dominant topic for a given preprocessed text."""
        if not self.is_fitted:
            return None
        dtm = self.count_vectorizer.transform([preprocessed_text])
        if dtm.nnz == 0:
            return None
        topic_probs = self.lda_model.transform(dtm)[0]
        dominant_idx = int(topic_probs.argmax())
        return {
            "topic_id": dominant_idx,
            "topic_label": self.topic_names[dominant_idx]["label"],
            "confidence": round(float(topic_probs[dominant_idx]), 3),
            "distribution": [round(float(p), 3) for p in topic_probs]
        }

    def save(self, filepath="models/lda_topic_model.joblib"):
        joblib.dump({
            "vectorizer": self.count_vectorizer,
            "lda_model": self.lda_model,
            "topic_names": self.topic_names,
            "n_topics": self.n_topics
        }, filepath)

    @classmethod
    def load(cls, filepath="models/lda_topic_model.joblib"):
        data = joblib.load(filepath)
        instance = cls(n_topics=data["n_topics"])
        instance.count_vectorizer = data["vectorizer"]
        instance.lda_model = data["lda_model"]
        instance.topic_names = data["topic_names"]
        instance.is_fitted = True
        return instance


def generate_actionable_recommendations(batch_analytics: dict):
    """
    Synthesizes aggregate statistics into prioritized executive business recommendations.
    """
    recommendations = []
    
    aspect_counts = batch_analytics.get("aspect_counts", {})
    aspect_negative = batch_analytics.get("aspect_negative", {})
    sentiment_counts = batch_analytics.get("sentiment_counts", {})
    total_reviews = sum(sentiment_counts.values()) or 1
    
    neg_pct = (sentiment_counts.get("Negative", 0) / total_reviews) * 100
    uncovered = batch_analytics.get('reviews_without_aspects', 0)
    if uncovered:
        recommendations.append({
            'severity': 'LOW', 'category': 'Aspect Coverage',
            'finding': f'{uncovered} reviews have no recognized business aspect.',
            'action': 'Check the selected feedback column and read general or unrecognized reviews. Ask for specific details before assigning an issue to a business area.',
            'impact': 'Missing aspect matches do not mean there are no business problems.'
        })
    
    # 1. Overall Health Alert
    if neg_pct > 30:
        recommendations.append({
            "severity": "CRITICAL",
            "category": "Customer Satisfaction Alert",
            "finding": f"High overall negative sentiment ({neg_pct:.1f}% of total reviews).",
            "action": "Read the negative reviews and confirm the most frequent issues before choosing a corrective action.",
            "impact": "May improve customer satisfaction; compare later feedback to assess the effect."
        })
    elif neg_pct < 15 and sentiment_counts.get('Positive', 0) / total_reviews >= .6:
        recommendations.append({
            "severity": "POSITIVE",
            "category": "Positive Feedback Trend",
            "finding": f"Strong positive reception with only {neg_pct:.1f}% negative feedback.",
            "action": "Leverage satisfied customer base for referral incentives, testimonials, and case studies.",
            "impact": "May support customer advocacy; positive sentiment alone does not establish loyalty."
        })

    # 2. Aspect-Specific Insights
    for aspect, total_mentions in aspect_counts.items():
        if total_mentions == 0:
            continue
        neg_mentions = aspect_negative.get(aspect, 0)
        neg_ratio = (neg_mentions / total_mentions) * 100
        
        if aspect == "Delivery & Packaging" and neg_ratio >= 25:
            recommendations.append({
                "severity": "HIGH",
                "category": "Logistics & Fulfillment",
                "finding": f"{neg_ratio:.1f}% negative sentiment in Delivery & Packaging ({neg_mentions} complaints).",
                "action": "Inspect the negative delivery and packaging reviews to distinguish delays, handling and packaging issues before changing fulfillment practices.",
                "impact": "May reduce transit-related returns; measure the effect after changes."
            })
        elif aspect == "Product Quality" and neg_ratio >= 25:
            recommendations.append({
                "severity": "CRITICAL",
                "category": "Quality Assurance",
                "finding": f"{neg_ratio:.1f}% negative sentiment in Product Quality ({neg_mentions} negative mentions).",
                "action": "Read the cited product-quality feedback and verify recurring problems with the quality team before prioritizing inspections or fixes.",
                "impact": "May help identify quality problems earlier; the batch does not establish a defect rate."
            })
        elif aspect == "Customer Support" and neg_ratio >= 20:
            recommendations.append({
                "severity": "HIGH",
                "category": "Support Operations",
                "finding": f"{neg_ratio:.1f}% negative sentiment regarding Customer Service.",
                "action": "Reduce wait times with smart triage routing and train front-line agents on empathetic de-escalation.",
                "impact": "May improve support satisfaction; measure outcomes after any process changes."
            })
        elif aspect == "Pricing & Value" and neg_ratio >= 25:
            recommendations.append({
                "severity": "MEDIUM",
                "category": "Pricing & Packaging",
                "finding": f"{neg_ratio:.1f}% of price-conscious reviews perceive poor value for money.",
                "action": "Introduce entry-tier bundles, seasonal discounts, or clearly articulate premium feature ROI in marketing.",
                "impact": "May improve perceived value; validate pricing changes separately."
            })
        elif aspect == "Usability & UI" and neg_ratio >= 25:
            recommendations.append({
                "severity": "MEDIUM",
                "category": "Software & UX",
                "finding": f"{neg_ratio:.1f}% negative sentiment in Usability & UI ({neg_mentions} negative mentions).",
                "action": "Inspect the cited software and usability reviews. Reproduce recurring problems and prioritize the most frequent verified issue.",
                "impact": "May improve usability; confirm the effect with subsequent feedback."
            })

    if not recommendations:
        recommendations.append({
            "severity": "LOW",
            "category": "Steady State Operations",
            "finding": "No monitored aspect crossed the configured negative-sentiment alert thresholds.",
            "action": "Continue baseline monitoring and maintain standard operating quality procedures.",
            "impact": "A larger representative sample may reveal issues this batch does not capture."
        })
        
    if total_reviews < 20:
        recommendations.insert(0, {
            'severity': 'LOW', 'category': 'Small Sample',
            'finding': f'This batch contains only {total_reviews} analyzed reviews.',
            'action': 'Use these results to inspect individual feedback. Collect more representative reviews before drawing broad business conclusions.',
            'impact': 'Small samples can give unstable percentages and should not establish business-wide trends.'
        })
    return recommendations
