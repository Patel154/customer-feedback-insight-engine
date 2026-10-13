"""One-time, pinned download. Prediction never sends reviews to a remote service."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'Xenova/bert-base-multilingual-uncased-sentiment'
REVISION = 'fad9b372f770c208d2efaf00d02f2cb3b61ec450'
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx']


def download():
    target = ROOT / 'models/review_sentiment'
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {name}...', flush=True)
        temporary = path.with_suffix(path.suffix + '.part')
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'repository': REPOSITORY, 'revision': REVISION, 'sha256': hashes,
                'original_model': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'license': 'MIT', 'labels': ['Negative', 'Neutral', 'Positive'],
                'mapping': 'Sum probabilities: stars 1-2 Negative, star 3 Neutral, stars 4-5 Positive'}
    (target / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Local context model is ready.', flush=True)


if __name__ == '__main__':
    download()
