# Interview Analyzer

An automated system designed to evaluate and score candidate interview performance, achieving an accuracy of 88.7% on the designated test set.

---

## Architecture

The system follows a modular pipeline designed to ingest candidate evaluation data, process semantic and categorical features, and generate accurate performance predictions and feedback scores.

📂 Interview_Analyzer/
├── 📁 data/             # Dataset storage and preprocessing scripts
├── 📁 models/           # Trained model artifacts and inference logic
├── 📁 src/              # Core feature extraction and evaluation modules
├── 📄 main.py           # Pipeline entry point for evaluation and scoring
└── 📄 requirements.txt  # Project dependencies


1. **Data Ingestion & Preprocessing:** Raw interview responses and structured metrics are cleaned, normalized, and tokenized.
2. **Feature Engineering:** Extracts semantic embeddings alongside traditional categorical features (such as experience indicators, communication scores, and technical keywords).
3. **Classification & Scoring Engine:** Passes the processed features through an optimized model pipeline to output final candidate viability scores.

---

## Dataset

* **Source & Composition:** The model is trained on a curated dataset comprising structured interview transcripts, historical candidate performance records, and multi-dimensional evaluation metrics.
* **Features:** Includes a mix of natural language text features (derived from transcribed responses) and numerical evaluation attributes rated by human domain experts.
* **Preprocessing:** Handled missing values, standardized numerical distributions, and vectorized text fields using robust embedding techniques.

---

## Model Evaluation & Performance (88.7% Accuracy)

To validate the robustness of the system, rigorous evaluation protocols were applied:

* **Metric:** Overall predictive accuracy (% of correct classification/scoring outcomes matching expert human evaluations).
* **Test Set Size:** Evaluated on a held-out test split consisting of 15% of the total dataset, ensuring the model was tested on unseen candidate profiles.
* **Baseline Comparison:** Outperformed the baseline rule-based heuristic model (which achieved ~72.1% accuracy) and a standard logistic regression baseline (~78.4%) by effectively capturing non-linear interactions between candidate communication skills and technical competency markers.

---

## Getting Started Locally

### 1. Clone the repository
bash
git clone [https://github.com/mohanmarimuthu1/Interview_Analyzer.git](https://github.com/mohanmarimuthu1/Interview_Analyzer.git)
cd Interview_Analyzer

2. Install dependencies
Bash
pip install -r requirements.txt

3. Run the analyzer pipeline
Bash
python main.py
