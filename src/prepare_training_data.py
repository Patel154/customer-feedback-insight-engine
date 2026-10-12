"""Prepare attributed real reviews plus explicitly marked teaching examples."""
from pathlib import Path
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def read_archive():
    rows = []
    with zipfile.ZipFile(ROOT / 'data/raw/uci_sentiment.zip') as archive:
        for name in archive.namelist():
            if name.endswith('_labelled.txt') and '__MACOSX' not in name:
                source = Path(name).stem
                pending = ''
                for line in archive.read(name).decode('utf-8').splitlines():
                    if not line.strip():
                        continue
                    if '\t' not in line:
                        pending += line + ' '
                        continue
                    text, label = line.rsplit('\t', 1)
                    text, pending = pending + text, ''
                    rows.append({'review_text': text, 'sentiment': 'Positive' if label == '1' else 'Negative', 'source': source})
    return pd.DataFrame(rows)


def prepare():
    archive_path = ROOT / 'data/raw/uci_sentiment.zip'
    real = read_archive() if archive_path.exists() else pd.read_csv(ROOT / 'data/raw/uci_reviews.csv')
    real['key'] = real.review_text.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str.strip()
    # Conflicting duplicates cannot supply an unambiguous evaluation target.
    conflict = real.groupby('key').sentiment.nunique()
    real = real[~real.key.isin(conflict[conflict > 1].index)].drop_duplicates('key').drop(columns='key')
    train, holdout = train_test_split(real, test_size=.2, random_state=42, stratify=real.source + real.sentiment)
    real.to_csv(ROOT / 'data/raw/uci_reviews.csv', index=False)
    holdout.to_csv(ROOT / 'data/processed/real_review_holdout.csv', index=False)
    synthetic = pd.read_csv(ROOT / 'data/raw/customer_feedback.csv')[['review_text', 'sentiment']]
    synthetic['source'] = 'generated_customer_feedback'
    examples = []
    positive = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'perfect', 'awesome', 'reliable', 'helpful', 'pleasant', 'superb', 'outstanding', 'lovely']
    negative = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'useless', 'unreliable', 'broken', 'defective', 'unpleasant', 'frustrating']
    subjects = ['The product', 'The service', 'My purchase', 'This experience', 'The app', 'The delivery', 'The quality']
    for subject in subjects:
        for word in positive:
            examples.extend([(f'{subject} is {word}.', 'Positive'), (f'{subject} is not {word}.', 'Negative')])
        for word in negative:
            examples.extend([(f'{subject} is {word}.', 'Negative'), (f'{subject} is not {word}.', 'Positive')])
        for word in ['okay', 'average', 'ordinary', 'acceptable', 'mediocre', 'standard', 'so-so']:
            examples.append((f'{subject} is {word}.', 'Neutral'))
    examples += [(t, 'Neutral') for t in [
        'I purchased this item last week.', 'The package contains two cables.',
        'The device comes in black and white.', 'It weighs about one kilogram.',
        'The order number is on the invoice.', 'I have not tried it yet.',
        'The product arrived yesterday.', 'I received the parcel today.',
        'There are three buttons on the side.', 'The box includes a user manual.',
        'The battery capacity is 5000 mAh.', 'The warranty covers twelve months.',
        'I ordered two units for the office.', 'The screen measures six inches.',
        'The delivery date is shown in my account.', 'I have a question about the return policy.',
        'Can you send me the invoice?', 'What is the price of this item?',
        'When will my order arrive?', 'How do I update the app?',
        'There is nothing special about it.', 'Neither good nor bad.',
        'It is fine, nothing exceptional.', 'It does the basic job.',
        'Some things are good and some need improvement.'
    ]]
    additions = pd.DataFrame(examples, columns=['review_text', 'sentiment'])
    additions['source'] = 'curated_teaching_examples'
    # Contrastive teaching examples, not copied from customer uploads. Include
    # benign/negated alternatives so arrival alone is not a complaint signal.
    contextual = []
    for subject in ['The laptop', 'My headphones', 'The appliance', 'This camera', 'The charger', 'The furniture',
                    'The product', 'The device', 'The package']:
        for ending in ['arrived cracked.', 'was delivered damaged.', 'arrived with a defect.', 'is damaged.',
                       'arrived badly damaged.', 'was damaged.',
                       'was damaged on arrival.', 'stopped functioning after two days.',
                       'does not work anymore.', 'came broken.', 'failed after one use.']:
            contextual.append((f'{subject} {ending}', 'Negative'))
        for ending in ['arrived in perfect condition.', 'was delivered without damage.',
                       'has no defects.', 'works flawlessly.', 'is not damaged.', 'was not damaged.',
                       'arrived undamaged.', 'has never failed me.']:
            contextual.append((f'{subject} {ending}', 'Positive'))
        for ending in ['arrived on Tuesday.', 'arrived yesterday.', 'was delivered this morning.',
                       'comes in three colors.', 'has not been tested yet.']:
            contextual.append((f'{subject} {ending}', 'Neutral'))
    for subject in ['', 'It is ', 'This is ', 'The purchase was ', 'The overall experience was ']:
        for ending in ['not bad at all.', 'not terrible.', 'not disappointing.', 'better than I expected.']:
            contextual.append((f'{subject}{ending}', 'Positive'))
        for ending in ['not good at all.', 'not acceptable.', 'not satisfactory.', 'worse than I expected.']:
            contextual.append((f'{subject}{ending}', 'Negative'))
    for subject in ['The instructions', 'The interface', 'Customer support', 'The packaging',
                    'The documentation', 'The screen', 'The battery life', 'The delivery process']:
        for ending in ['could be better.', 'needs improvement.', 'should be improved.',
                       'is not as good as expected.', 'leaves a lot to be desired.', 'is inadequate.']:
            contextual.append((f'{subject} {ending}', 'Negative'))
        for ending in ['is much better now.', 'could not be better.', 'has improved considerably.',
                       'is better than expected.', 'does not need improvement.']:
            contextual.append((f'{subject} {ending}', 'Positive'))
        for ending in ['is adequate, nothing more.', 'is average overall.', 'is satisfactory but unremarkable.',
                       'meets the basic requirements.', 'is neither impressive nor disappointing.']:
            contextual.append((f'{subject} {ending}', 'Neutral'))
    contextual = pd.DataFrame(contextual, columns=['review_text', 'sentiment'])
    contextual['source'] = 'contrastive_teaching_examples'
    combined = pd.concat([train, synthetic, additions, contextual], ignore_index=True)
    # Keep the external holdout entirely outside training, including duplicate strings.
    holdout_keys = set(holdout.review_text.str.lower().str.strip())
    combined = combined[~combined.review_text.str.lower().str.strip().isin(holdout_keys)]
    combined.to_csv(ROOT / 'data/raw/training_feedback.csv', index=False)
    print(f'Training corpus: {len(combined)}; independent real-review holdout: {len(holdout)}')


if __name__ == '__main__':
    prepare()
