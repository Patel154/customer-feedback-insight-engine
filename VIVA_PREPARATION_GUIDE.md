# Machine Learning Viva Voce & Defense Guide
## Project: Automated Sentiment Analysis and Insight Extraction from Customer Feedback
**Subject:** Artificial Intelligence (5th Semester)

**Current implementation and evaluation:** The active pipeline uses pretrained
BERT product-review sentiment through local quantized ONNX inference, plus a
locally trained TF-IDF calibrated SVM specialist. BERT was not trained from
scratch in this project. It predicts five star probabilities mapped to three
sentiments. The active pipeline scored 538/594 (90.6%) on the fixed UCI binary
benchmark versus 503/594 (84.7%) for the SVM. These reviews were excluded from
local SVM training, but their errors informed hybrid refinement; upstream BERT
overlap is unknown. This is a development benchmark and does not measure
three-class accuracy on future uploads. Scores are uncalibrated confidence.

The classical algorithm comparison and its training-only CV remain an academic
baseline. The sections below on TF-IDF, stemming and classical training describe
that baseline; BERT instead receives contextual tokenization without stemming.

---

## 1. Project Overview & Business Motivation

### Q1: What is the primary objective of your project?
**Answer:** The primary objective is to build an automated, end-to-end Machine Learning and NLP platform that ingests raw, unstructured customer feedback, classifies its sentiment (Positive, Neutral, Negative) with high confidence, and performs **Aspect-Based Sentiment Analysis (ABSA)** and **Topic Modeling (LDA)** to extract actionable business insights and operational recommendations.

### Q2: Why is "Insight Extraction" necessary if we already have Sentiment Classification?
**Answer:** Standard sentiment classification only outputs a polarity label (e.g., *"This review is 95% Negative"*). However, an executive or product manager cannot act on just a label; they need to know **why** the customer is unhappy. Insight extraction pinpoints the exact operational department responsible (e.g., *Delivery delay*, *Hardware defect*, or *App Bluetooth crash*) and quantifies the complaint volume to prioritize managerial interventions.

---

## 2. Text Preprocessing & NLP Pipeline

### Q3: What preprocessing steps did you implement on raw text?
**Answer:** The BERT path removes HTML, normalizes URLs and common positive
idioms, and preserves punctuation, negation and word order. Its tokenizer uses
512-token windows with overlap; every window contributes to a long review.
The local SVM and LDA paths expand contractions and use deterministic offline
Snowball stemming. Single and batch analysis share the same prediction path.

### Q4: What is the difference between Stemming and Lemmatization? Why did you choose Stemming?
**Answer:**
- **Stemming** (e.g., Porter Stemmer) is a rule-based heuristic that chops off word affixes (prefixes/suffixes). It often produces non-real words (e.g., *"university"* $\to$ *"univers"*, *"trouble"* $\to$ *"troubl"*).
- **Lemmatization** uses a vocabulary and morphological analysis (WordNet lexical database) to return the actual valid base form (lemma) (e.g., *"better"* $\to$ *"good"*, *"crashes"* $\to$ *"crash"*).
- We chose offline Snowball stemming for reproducible execution without corpus downloads. Stems can be less readable than lemmas; topic labels should be interpreted as hints.

### Q5: Why is negation handling critical in Sentiment Analysis?
**Answer:** In sentiment analysis, negation words (*"not"*, *"never"*, *"hardly"*, *"neither"*) invert the polarity of the adjacent adjective or verb. If stopword filtering naively deletes *"not"*, the phrase *"not satisfied with battery"* reduces to *"satisfied battery"*, completely flipping the predicted class from Negative to Positive.

---

## 3. Feature Engineering & TF-IDF

### Q6: What is TF-IDF and how is it mathematically formulated?
**Answer:** TF-IDF stands for **Term Frequency - Inverse Document Frequency**. It reflects how important a word is to a document in a collection or corpus.
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Where:
- **$\text{TF}(t, d)$** is the frequency of term $t$ in document $d$ (often sublinearly scaled: $1 + \log(\text{TF})$).
- **$\text{IDF}(t, D)$** measures how common or rare a term is across all documents:
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
High TF-IDF occurs when a term appears frequently in a single document but rarely across the entire corpus, filtering out background noise.

### Q7: Why combine word and character n-grams?
**Answer:** Word unigrams, bigrams and trigrams (`ngram_range=(1, 3)`) retain context such as *"not working"* and *"would not recommend"*. Character 3–5 grams can add evidence when spelling varies. Both are TF-IDF weighted and joined with a FeatureUnion.

### Q8: What is Data Leakage in text classification and how did you prevent it?
**Answer:** Data leakage occurs when information from the test dataset leaks into the training pipeline before model training. If you fit `TfidfVectorizer.fit()` on the entire dataset before splitting, the vocabulary IDF weights include information from the test set. We strictly followed the **Featurization Ordering Rule**: split the raw data into $80\%$ Train and $20\%$ Test first; call `fit_transform()` exclusively on the Training set, and only call `transform()` on the Test set.

---

## 4. Machine Learning Algorithms & Mathematical Foundations

### Q9: Which algorithms did you benchmark, and which performed best?
**Answer:** We benchmarked 4 diverse algorithms:
1. **Multinomial Naive Bayes (MultinomialNB)**
2. **Logistic Regression (L2 regularized)**
3. **Linear Support Vector Machine (LinearSVC with Probability Calibration)**
4. **Random Forest Classifier (Ensemble of 80 Decision Trees)**

**Winning classical baseline:** Calibrated Linear SVM was selected by training-only 5-fold macro F1 (0.868). It measured 87.5% internal accuracy and 84.7% accuracy on the independent 594-review UCI positive/negative holdout. The internal test contains synthetic examples, and the external test does not cover real neutral feedback.

### Q10: Explain the mathematics of Multinomial Naive Bayes. What is the "naive" assumption?
**Answer:** By Bayes' Theorem:
$$P(y = c \mid \mathbf{x}) = \frac{P(y = c) \prod_{i=1}^n P(x_i \mid y = c)}{P(\mathbf{x})}$$
- **The Naive Assumption:** It assumes that all feature words $x_i$ are conditionally independent given the class label $c$, which is linguistically untrue (words strongly correlate with each other). Despite this violation, Naive Bayes performs surprisingly well for text classification because the decision boundary can be accurate even when probabilities are poorly calibrated.
- **Laplace Smoothing ($\alpha$):**
$$P(w_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$$
Where $\alpha = 0.5$ or $1.0$. Smoothing prevents zero-probability multiplication when an unseen word appears at test time.

### Q11: Why does Linear SVM perform exceptionally well on high-dimensional text data?
**Answer:**
1. Text representations using TF-IDF are very high-dimensional (e.g., 5,000 features) and sparse.
2. Cover's Theorem states that a complex classification problem cast into a high-dimensional space is more likely to be linearly separable than in a low-dimensional space.
3. Linear SVM finds the maximum margin hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ that maximizes $\frac{2}{\|\mathbf{w}\|}$ while penalizing margin violations using hinge loss with regularization parameter $C$:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
4. Since Linear SVM maximizes margins, it does not easily overfit even with thousands of features.

### Q12: Why did you wrap LinearSVC with `CalibratedClassifierCV`?
**Answer:** Standard `LinearSVC` optimizes the hinge loss and outputs signed geometric distances from the decision boundary, not normalized probabilities $[0, 1]$. To display confidence meters and class probability distributions on our web dashboard, we applied **Platt Scaling** (`CalibratedClassifierCV`), which fits a logistic sigmoid over the SVM decision values using cross-validation.

### Q13: Why did Random Forest have higher latency or lower efficiency than Linear SVM on text?
**Answer:** Random Forest builds multiple axis-aligned decision trees. High-dimensional sparse TF-IDF matrices (thousands of columns with mostly zeros) force decision trees to evaluate deep, fragmented splits across sparse features, consuming significant memory and training time without outperforming the smooth global linear boundary established by SVM or Logistic Regression.

---

## 5. Evaluation Metrics & Error Analysis

### Q14: Why did you rely on Macro-F1 rather than simple Accuracy?
**Answer:** Real-world customer reviews are often class-imbalanced (e.g., positive reviews often outnumber neutral reviews). A naive majority-class classifier could achieve 70% accuracy while failing completely on neutral or negative cases. **Macro-averaged F1-Score** calculates the harmonic mean of Precision and Recall independently for each class and averages them with equal weight:
$$\text{Macro-F1} = \frac{\text{F1}_{\text{Negative}} + \text{F1}_{\text{Neutral}} + \text{F1}_{\text{Positive}}}{3}$$
This penalizes models that neglect minority classes.

### Q15: Explain the Confusion Matrix. What are False Positives and False Negatives in our business context?
**Answer:**
- **True Positive (TP):** Correctly predicted negative review as Negative.
- **False Positive (FP):** Predicting a review as Negative when it was actually Positive (unnecessary escalation).
- **False Negative (FN):** Predicting a review as Positive when it was actually a severe complaint (missed critical issue; highest business risk).
In customer satisfaction analytics, minimizing False Negatives on severe complaints is paramount to prevent customer churn.

---

## 6. Topic Modeling & Aspect-Based Sentiment Mining

### Q16: How does Latent Dirichlet Allocation (LDA) work?
**Answer:** LDA is an unsupervised generative probabilistic model:
1. It assumes that every document is a mixture of a small number of latent topics: $\theta_d \sim \text{Dirichlet}(\boldsymbol{\alpha})$.
2. Each topic is a multinomial distribution over vocabulary words: $\phi_k \sim \text{Dirichlet}(\boldsymbol{\beta})$.
3. During training, Gibbs sampling or online variational inference iteratively updates topic assignments for each word to find cluster terms that co-occur across documents.

### Q17: How does your Aspect-Based Sentiment Analysis (ABSA) system work?
**Answer:**
1. **Aspect Detection:** Reviews are scanned for domain keywords spanning five core business facets: *Product Quality*, *Customer Support*, *Delivery & Packaging*, *Pricing & Value*, and *Usability & UI*.
2. **Clause Segmentation:** The feedback is split into clauses/sentences at punctuation boundaries.
3. **Clause-Level Sentiment Attribution:** The sentiment classifier evaluates the isolated clause containing the aspect keyword. This correctly separates mixed reviews such as *"The phone screen is great, but shipping was delayed by a week"* into *Product Quality: Positive* and *Delivery: Negative*.

---

## 7. System Architecture, Deployment & Production Readiness

### Q18: What is the inference time per review? Can this system scale to 1 million reviews?
**Answer:**
- Classical classifier timing is recorded in `models/model_metrics.json`; the active contextual benchmark timing is in `models/context_evaluation.json`. This benchmark measures classifier/feature prediction on the internal test batch; it does not include complete upload, aspect analysis, topic modeling or browser time. The demo supports 10,000 rows per upload, rather than claiming million-row production capacity.
- **To scale to 1M reviews/day:**
  1. Ingest reviews asynchronously via an Apache Kafka or RabbitMQ event queue.
  2. Deploy the Python inference pipeline on Celery workers or containerized microservices (Docker + Kubernetes).
  3. Cache frequent n-gram vectors using Redis.
  4. Store analytics in PostgreSQL or ClickHouse for real-time executive dashboarding.
