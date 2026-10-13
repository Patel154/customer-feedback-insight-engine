"""Behavior tests for the errors that previously broke customer-file analysis."""
import io
import json
import unittest
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.app import app
from src.predictor import SentimentInsightPipeline
from src.train import load_and_clean_data


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.pipe = SentimentInsightPipeline(ROOT / 'models')

    def upload(self, text, filename='feedback.csv', column=None, endpoint='/api/batch_upload', **fields):
        content = text.encode('utf-8-sig') if isinstance(text, str) else text
        form = {'file': (io.BytesIO(content), filename), **fields}
        if column:
            form['text_column'] = column
        return self.client.post(endpoint, data=form, content_type='multipart/form-data')

    def test_single_batch_agree_and_skip_missing(self):
        texts = ['The product is not good.', 'The app is excellent.', 'The product is average.']
        frame, analytics = self.pipe.predict_batch(pd.DataFrame({'Feedback': texts + [None, 12, '   ']}), 'Feedback')
        self.assertEqual(analytics['total_samples'], 3)
        self.assertEqual(analytics['skipped_rows'], 3)
        self.assertEqual(sum(analytics['sentiment_counts'].values()), 3)
        for index, text in enumerate(texts):
            single = self.pipe.predict_single(text)
            self.assertEqual(frame.loc[index, 'predicted_sentiment'], single['sentiment'])
            self.assertAlmostEqual(frame.loc[index, 'confidence'], single['confidence'])

    def test_negation_and_curly_apostrophe(self):
        self.assertEqual(self.pipe.predict_single('The product is not good.')['sentiment'], 'Negative')
        self.assertEqual(self.pipe.predict_single('The product isn’t good.')['sentiment'], 'Negative')
        self.assertEqual(self.pipe.predict_single('The service is excellent.')['sentiment'], 'Positive')

    def test_single_endpoint_serializes_success(self):
        for text in ['The product is excellent.', 'The delivery was excellent but the app is terrible.', 'zxqv qqqxyz']:
            response = self.client.post('/api/predict', json={'text': text})
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertIsInstance(response.get_json()['requires_review'], bool)

    def test_mixed_clauses_have_separate_sentiments(self):
        for text in ['The delivery was excellent but the app is terrible.', 'The delivery was excellent and the app is terrible.']:
            result = self.pipe.predict_single(text)
            self.assertTrue(result['is_mixed'])
            self.assertEqual(result['aspects']['Delivery & Packaging']['sentiment'], 'Positive')
            self.assertEqual(result['aspects']['Usability & UI']['sentiment'], 'Negative')

    def test_aspect_word_boundaries(self):
        self.assertNotIn('Usability & UI', self.pipe.aspect_analyzer.detect_aspects('I bought an apple.'))

    def test_product_use_is_not_delivery(self):
        for text in ['It helps me find my lost keys.', 'Tracking updates are delayed.',
                     'My luggage was lost at the airport.', 'I was running late but setup was easy.']:
            self.assertNotIn('Delivery & Packaging', self.pipe.aspect_analyzer.detect_aspects(text))
        self.assertIn('Delivery & Packaging', self.pipe.aspect_analyzer.detect_aspects('Shipping was delayed.'))
        self.assertIn('Pricing & Value', self.pipe.aspect_analyzer.detect_aspects('Got it on sale, an absolute steal!'))
        self.assertIn('Customer Support', self.pipe.aspect_analyzer.detect_aspects('The helpful staff communicated clearly.'))

    def test_positive_context_and_idioms(self):
        for text in ['Super intuitive to use, set it up in minutes.',
                     'I love them. They tell me where my keys are.',
                     'I cannot recommend these more. They saved me hundreds of dollars.',
                     'We would not have found our keys without this tracker.']:
            with self.subTest(text=text):
                self.assertEqual(self.pipe.predict_single(text)['sentiment'], 'Positive')
        self.assertEqual(self.pipe.predict_single('I cannot recommend this defective product.')['sentiment'], 'Negative')

    def test_prediction_independent_of_other_rows(self):
        from src.context_sentiment import ContextSentimentModel
        model = ContextSentimentModel(ROOT / 'models/review_sentiment')
        text = 'The product is average.'
        alone = model.classify([text])[0]
        model.cache.clear()
        together = model.classify(['Super intuitive and easy to use. ' * 12, text])[1]
        self.assertEqual(alone, together)

    def test_long_feedback_is_not_silently_truncated(self):
        text = 'This device arrived yesterday. ' * 140 + 'Unfortunately the battery overheats and is dangerous.'
        result = self.pipe.predict_single(text)
        self.assertGreater(result['text_windows'], 1)
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1, places=3)

    def test_continued_aspect_opinion_is_in_evidence(self):
        result = self.pipe.predict_single('The interface is functional but looks outdated.')
        self.assertIn('looks outdated', result['aspects']['Usability & UI']['snippet'])
        self.assertNotEqual(result['aspects']['Usability & UI']['sentiment'], 'Positive')

    def test_aspect_inflections_and_implicit_functionality(self):
        for text in ['It arrived yesterday.', 'It arrives tomorrow.', 'It is arriving today.']:
            self.assertIn('Delivery & Packaging', self.pipe.aspect_analyzer.detect_aspects(text))
        self.assertIn('Product Quality', self.pipe.aspect_analyzer.detect_aspects('Works perfectly, exceeded my expectations.'))
        self.assertIn('Usability & UI', self.pipe.aspect_analyzer.detect_aspects('The user guide needs improvement.'))

    def test_damage_and_negation_regressions(self):
        for text, expected in [
            ('Product arrived damaged.', 'Negative'),
            ('The keyboard arrived broken.', 'Negative'),
            ('The package was not damaged.', 'Positive'),
            ('The device arrived yesterday.', 'Neutral'),
            ('Not bad at all.', 'Positive'),
            ('The instructions could not be better.', 'Positive'),
            ('The instructions could be better.', 'Negative'),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.pipe.predict_single(text)['sentiment'], expected)
        result = self.pipe.predict_single('As described, but manual could be better.')
        self.assertEqual(result['aspects']['Usability & UI']['sentiment'], 'Negative')

    def test_aspect_coverage_keeps_general_reviews_and_skips_separate(self):
        _, a = self.pipe.predict_batch(pd.DataFrame({'Feedback': ['Wonderful!', 'The app is terrible.', None]}), 'Feedback')
        self.assertEqual(a['reviews_with_aspects'], 1)
        self.assertEqual(a['reviews_without_aspects'], 1)
        self.assertEqual(sum(a['general_sentiment_counts'].values()), 1)
        self.assertEqual(a['skipped_rows'], 1)
        _, general = self.pipe.predict_batch(pd.DataFrame({'Feedback': ['Wonderful!', 'Terrible!']}), 'Feedback')
        self.assertEqual(sum(general['aspect_counts'].values()), 0)
        self.assertEqual(general['reviews_without_aspects'], 2)
        self.assertTrue(any(r['category'] == 'Aspect Coverage' for r in general['recommendations']))

    def test_aspect_totals_reconcile(self):
        frame, analytics = self.pipe.predict_batch(pd.DataFrame({'Feedback': [
            'The delivery was excellent but the app is terrible.', 'Product arrived damaged.', 'Wonderful!'
        ]}), 'Feedback')
        self.assertEqual(analytics['reviews_with_aspects'] + analytics['reviews_without_aspects'], analytics['total_samples'])
        for aspect, count in analytics['aspect_counts'].items():
            self.assertEqual(count, sum(analytics[f'aspect_{label}'][aspect] for label in ['positive', 'negative', 'neutral', 'mixed']))

    def test_standalone_aspect_api_matches_pipeline(self):
        text = 'The delivery was excellent but the app is terrible.'
        self.assertEqual(self.pipe.predict_single(text)['aspects'], self.pipe.aspect_analyzer.analyze_aspect_sentiments(text, classify=self.pipe._classify_many))

    def test_original_derived_column_names_preserved(self):
        response = self.upload('Feedback,confidence,original_confidence\nThe product is good.,old,older', column='Feedback')
        self.assertEqual(response.status_code, 200)
        row = response.get_json()['reviews'][0]
        self.assertEqual(row['original_confidence'], 'older')
        self.assertEqual(row['original_original_confidence'], 'old')

    def test_custom_column_and_full_export_payload(self):
        reviews = ['Customer,Comments'] + [f'{i},The product is excellent.' for i in range(310)]
        response = self.upload('\n'.join(reviews), column='Comments')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body['reviews']), 310)
        self.assertEqual(body['reviews'][-1]['source_row'], 311)
        self.assertEqual(body['reviews'][0]['analyzed_text'], 'The product is excellent.')
        self.assertIn('Customer', body['reviews'][0])

    def test_single_column_uppercase_csv(self):
        response = self.upload('Review Text\nThe service is excellent.\nThe service is terrible.', 'REVIEWS.CSV')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['analytics']['total_samples'], 2)

    def test_title_blank_rows_and_repeated_headers_preserve_source_rows(self):
        response = self.upload('Customer feedback report\n\nFeedback / Comments,Status\nExcellent service,Closed\n,\nFeedback / Comments,Status\nTerrible service,Open')
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body['text_column'], 'Feedback / Comments')
        self.assertEqual([r['source_row'] for r in body['reviews']], [4, 5, 6, 7])
        self.assertEqual(body['analytics']['skipped_rows'], 2)

    def test_duplicate_headers_rejected_before_pandas_mangles_them(self):
        response = self.upload('Feedback,Feedback\nGood,Bad', endpoint='/api/preview_upload')
        self.assertEqual(response.status_code, 400)
        self.assertIn('unique', response.get_json()['error'])

    def test_semicolon_and_utf16(self):
        response = self.upload('id;Feedback\n1;The product is great.\n2;The product is terrible.'.encode('utf-16'))
        self.assertEqual(response.status_code, 200)

    def test_excel_sheet_selection(self):
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            pd.DataFrame({'Notes':['Cover sheet']}).to_excel(writer, sheet_name='Cover', index=False)
            pd.DataFrame({'Customer':['A','B','C'], 'Feedback':['The product is great.',None,'The product is terrible.']}).to_excel(writer,sheet_name='Reviews',index=False)
        content = stream.getvalue()
        preview = self.upload(content, 'feedback.xlsx', endpoint='/api/preview_upload')
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.get_json()['sheets'], ['Cover','Reviews'])
        self.assertEqual(preview.get_json()['sheet'], 'Reviews')
        response = self.upload(content, 'feedback.xlsx', column='Feedback', sheet='Reviews')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['analytics']['skipped_rows'], 1)
        self.assertEqual(len(response.get_json()['reviews']), 3)

    def test_ambiguous_columns_require_choice(self):
        response = self.upload('Name,Review,Comments\nAlice,The product is great.,The app is terrible.')
        self.assertEqual(response.status_code, 400)
        self.assertIn('choose', response.get_json()['error'])

    def test_no_arbitrary_name_column_fallback(self):
        response = self.upload('Name,Rating\nAlice,5')
        self.assertEqual(response.status_code, 400)

    def test_metadata_column_is_not_analyzed_as_feedback(self):
        response = self.upload('Feedback,Status\nThe app is excellent.,Resolved', column='Status')
        self.assertEqual(response.status_code, 400)
        self.assertIn('metadata', response.get_json()['error'])

    def test_invalid_inputs_are_actionable(self):
        for body in [{'text':''},{'text':123},['not an object'],{'text':'12345'}, {'text':'बहुत अच्छा'}, {'text':'https://example.com'}, {'text':'<br>'}]:
            response = self.client.post('/api/predict',json=body)
            self.assertEqual(response.status_code, 400)
            self.assertIn('error',response.get_json())
        self.assertEqual(self.upload('Feedback\n123\n456').status_code,400)
        self.assertEqual(self.upload('Feedback\n',endpoint='/api/preview_upload').status_code,400)

    def test_unknown_words_flagged(self):
        result = self.pipe.predict_single('zxqv qqqxyz')
        self.assertTrue(result['requires_review'])
        self.assertIsNone(result['topic'])

    def test_training_excludes_external_holdout(self):
        training = load_and_clean_data()
        holdout = pd.read_csv(ROOT / 'data/processed/real_review_holdout.csv')
        self.assertFalse(set(training.cleaned_text) & {self.pipe.preprocess(t) for t in holdout.review_text})
        internal_train = pd.read_csv(ROOT / 'data/processed/train_data.csv')
        internal_test = pd.read_csv(ROOT / 'data/processed/test_data.csv')
        self.assertFalse(set(internal_train.cleaned_text) & set(internal_test.cleaned_text))

    def test_upload_limit_and_extension(self):
        self.assertEqual(self.upload('wrong',filename='file.pdf').status_code,400)
        oversized = 'Feedback\n' + 'Good\n' * 10001
        self.assertEqual(self.upload(oversized, endpoint='/api/preview_upload').status_code,400)

    def test_dashboard_has_no_remote_runtime_assets(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code,200)
        self.assertNotIn(b'cdn.jsdelivr',response.data)
        self.assertNotIn(b'fonts.googleapis',response.data)


if __name__ == '__main__':
    unittest.main()
