"""Local transformer inference with bounded batches and full-text windows."""
import json
import re
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class ContextSentimentModel:
    name = 'RoBERTa English sentiment (local ONNX)'

    def __init__(self, directory):
        directory = Path(directory)
        if not (directory / 'onnx/model_quantized.onnx').exists():
            raise RuntimeError('Context model is missing. Run python src/download_context_model.py once, then restart the app.')
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        original_labels = [config['id2label'][str(i)].capitalize() for i in range(len(config['id2label']))]
        self.star_model = len(original_labels) == 5 and original_labels[0].lower() == '1 star'
        self.labels = ['Negative', 'Neutral', 'Positive'] if self.star_model else original_labels
        self.name = 'BERT product-review sentiment (local ONNX)' if self.star_model else self.name
        if set(self.labels) != {'Negative', 'Neutral', 'Positive'}:
            raise ValueError('Unsupported model labels.')
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=512, stride=64)
        self.pad_id = config['pad_token_id']
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'onnx/model_quantized.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.cache = OrderedDict()
        self.lock = RLock()

    @staticmethod
    def prepare(text):
        # Preserve punctuation, negation and word order; stemming harms this model.
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
        text = re.sub(r'(?<!\w)@\w+', '@user', text)
        # Resolve common positive idioms whose literal negation fools classifiers.
        # A plain "cannot recommend" remains negative; the suffix is required.
        text = re.sub(r"\b(?:cannot|can not|can['’]t|could not|couldn['’]t) recommend (.*?)\b(?:enough|(?:any )?more)\b",
                      r'highly recommend \1', text, flags=re.I)
        text = re.sub(r"\b(?:cannot|can not|can['’]t) (?:have|get) enough\b", 'want more', text, flags=re.I)
        text = re.sub(r"\b(?:could not|couldn['’]t|cannot|can['’]t) be better\b", 'is excellent', text, flags=re.I)
        text = re.sub(r"\bnot bad(?: at all)?\b", 'quite good', text, flags=re.I)
        text = re.sub(r'\babsolute steal\b', 'excellent bargain', text, flags=re.I)
        text = re.sub(r"\b(?:would not|wouldn['’]t) have (found|located|recovered)([^.!?]*?) without\b",
                      r'successfully \1\2 thanks to', text, flags=re.I)
        return re.sub(r'\s+', ' ', text).strip()

    def classify(self, texts):
        with self.lock:
            return self._classify(texts)

    def _classify(self, texts):
        keys = [self.prepare(t) for t in texts]
        unique = list(dict.fromkeys(keys))
        resolved = {key: self.cache[key] for key in unique if key in self.cache}
        pending = [key for key in unique if key not in resolved]
        windows = []
        totals = {key: [] for key in pending}
        for key in pending:
            encoded = self.tokenizer.encode(key)
            for index, window in enumerate([encoded, *encoded.overflowing]):
                windows.append((key, window.ids, max(1, len(window.ids) - 2 - (64 if index else 0))))
        # Length bucketing avoids padding a short clause to a full-page review.
        windows.sort(key=lambda item: len(item[1]))
        # Dynamic int8 activation scales depend on other items in a batch.
        # One window per inference makes a review independent of neighboring rows.
        for start in range(0, len(windows)):
            batch = windows[start:start + 1]
            width = max(len(ids) for _, ids, _ in batch)
            ids = np.full((len(batch), width), self.pad_id, dtype=np.int64)
            mask = np.zeros_like(ids)
            for index, (_, tokens, _) in enumerate(batch):
                ids[index, :len(tokens)] = tokens
                mask[index, :len(tokens)] = 1
            feed = {'input_ids': ids, 'attention_mask': mask}
            if 'token_type_ids' in self.inputs:
                feed['token_type_ids'] = np.zeros_like(ids)
            logits = self.session.run(None, feed)[0]
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            if self.star_model:
                probabilities = np.column_stack((probabilities[:, :2].sum(axis=1), probabilities[:, 2], probabilities[:, 3:].sum(axis=1)))
            for (key, _, weight), row in zip(batch, probabilities):
                totals[key].append((row, weight))
        for key, chunks in totals.items():
            probability = np.average([p for p, _ in chunks], axis=0, weights=[w for _, w in chunks])
            labels = {self.labels[int(p.argmax())] for p, _ in chunks}
            resolved[key] = {'probabilities': {label: float(p) for label, p in zip(self.labels, probability)},
                             'window_count': len(chunks),
                             'window_disagreement': {'Positive', 'Negative'}.issubset(labels)}
        for key in unique:
            self.cache[key] = resolved[key]
            self.cache.move_to_end(key)
        while len(self.cache) > 2048:
            self.cache.popitem(last=False)
        return [resolved[key] for key in keys]
