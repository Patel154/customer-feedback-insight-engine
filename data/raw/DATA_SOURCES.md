# Dataset provenance

The real English review corpus is **Sentiment Labelled Sentences**, donated by
Dimitrios Kotzias (2015), UCI Machine Learning Repository, DOI 10.24432/C57604.
Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
License: Creative Commons Attribution 4.0 (CC BY 4.0).
Associated paper: Kotzias, Denil, de Freitas and Smyth, *From Group to Individual
Labels Using Deep Features*, KDD 2015.

It contains Amazon, Yelp and IMDb positive/negative sentences. It has **no
neutral labels**. `src/prepare_training_data.py` removes duplicates and conflicting
duplicate labels, then reserves 20% by source and sentiment (seed 42) before
adding generated customer reviews and curated teaching examples to training.
`data/processed/real_review_holdout.csv` is never used in training or model selection.

Neutral training examples come from the original generated customer-feedback
dataset and explicitly marked curated examples. The internal benchmark therefore
includes synthetic text and is not a guarantee of accuracy on arbitrary files.
The separate real-review score is a binary-source evaluation with a three-class
classifier; real neutral feedback and domain shifts require further labeled data.

`contrastive_teaching_examples` adds authored paired examples of damage versus
undamaged receipt, negation, factual arrival, adequate performance and requests
for improvement. They are generated in `src/prepare_training_data.py`; customer
uploads, ratings and personal information are not incorporated into training.
Shared teaching templates can inflate internal validation, so report the external
score and its limited label/domain coverage alongside internal results.
