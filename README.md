# Smart Text Intelligence — Text Classification & Evaluation

![Application Screenshot](screenshots/demo.png)

> **Note on Screenshot:** To add a live capture of your running application, launch `streamlit run app.py`, press `Win + Shift + S` on Windows to take a snapshot of the interface, and save the file as `screenshots/demo.png`.

---

## 📖 Project Overview

**Smart Text Intelligence** is a lightweight, explainable machine learning system and interactive web application designed to classify natural language text into four distinct domain categories:
- 💻 **Technology**: Software engineering, artificial intelligence, cybersecurity, cloud computing, and hardware.
- 🎓 **Education**: Schools, university curricula, academic research, pedagogy, and student learning.
- ⚽ **Sports**: Athletic competitions, tournaments, team sports, player mechanics, and match strategy.
- 🏥 **Health**: Clinical medicine, mental wellness, disease prevention, pharmacology, and nutrition.

The system emphasizes transparent, reproducible machine learning using **Term Frequency-Inverse Document Frequency (TF-IDF)** feature extraction paired with a **Multinomial Logistic Regression** classifier. It provides real-time confidence scores, category probability breakdowns, and strict input validation via a modern Streamlit web interface.

---

## 🎯 Objectives

1. **Demonstrate Practical ML Classification**: Showcase how lightweight traditional machine learning models can achieve high accuracy on domain-specific text classification without expensive external APIs or GPUs.
2. **Explainability & Transparency**: Provide confidence scoring, probability distributions, and plain-language explanations of model predictions.
3. **Robust Error Handling**: Gracefully handle invalid user inputs (empty strings, whitespace, inputs >5,000 characters, non-string data) without exposing application tracebacks.
4. **Deterministic Evaluation**: Provide a dedicated evaluation script (`evaluate.py`) with sample-by-sample verification, a classification report, and a confusion matrix.
5. **Clean Architecture**: Decouple model training (`train_model.py`), utility validation logic (`classifier_utils.py`), evaluation (`evaluate.py`), and web serving (`app.py`).

---

## ✨ Features

- **Interactive Streamlit Web Interface**: Clean, responsive user interface with real-time prediction feedback and probability distributions.
- **One-Click Sample Inputs**: Pre-loaded example prompts across all four categories and edge-case inputs for instant testing.
- **Confidence Scoring & Low-Confidence Alerts**: Visual progress indicators for all classes, plus warning banners when model confidence drops below 45%.
- **Comprehensive Input Validation**: Rejects empty inputs, whitespace-only entries, and text exceeding 5,000 characters with friendly warnings.
- **Reproducible Local Training**: Trains in under a second using `scikit-learn` with fixed random states (`random_state=42`) and 5-fold cross-validation.
- **Independent Evaluation Suite**: Automated testing with 24 separate test examples (0 overlap with training data), producing precision, recall, F1-scores, and confusion matrices.
- **Zero API Dependencies**: Runs 100% offline and locally without API keys, paid services, or external network requests.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12+ (3.9+ compatible) | Core programming language |
| **User Interface** | Streamlit (`>=1.30.0`) | Interactive web application and dashboard |
| **Machine Learning** | scikit-learn (`>=1.3.0`) | TF-IDF vectorization, Logistic Regression, pipeline serialization |
| **Data Handling** | pandas (`>=2.0.0`), numpy (`>=1.24.0`) | Dataset management and numerical transformations |
| **Model Persistence** | joblib (`>=1.3.0`) | Serializing and loading model pipelines |

---

## 🔬 Classification Approach Explained

The classifier uses a standard two-stage natural language processing pipeline:

```
[Raw User Text]
       │
       ▼
[Input Validation]  --> Check character length, empty text, whitespace
       │
       ▼
[TF-IDF Vectorizer] --> Lowercasing, stopword removal, (1, 2) n-grams, sublinear TF scaling
       │
       ▼
[Logistic Regression] -> Linear decision boundaries + Softmax normalization
       │
       ▼
[Prediction Output] --> Top Category + Confidence Score + Probability Distribution
```

1. **Text Preprocessing & TF-IDF Vectorization**:
   - The input text is normalized and converted into numerical feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**.
   - **Unigrams and Bigrams** (`ngram_range=(1, 2)`) capture individual words (e.g., *"cloud"*, *"cardiovascular"*) and two-word phrases (e.g., *"machine learning"*, *"blood pressure"*).
   - **English Stopwords** are stripped to remove non-informative words like *"the"*, *"is"*, and *"at"*.
   - **Sublinear TF scaling** dampens the impact of words that appear repeatedly in a single document.

2. **Multinomial Logistic Regression**:
   - Logistic Regression computes a weighted linear combination of the TF-IDF feature values for each category.
   - A **Softmax activation function** converts these raw values into normalized probability scores between 0.0 and 1.0 that sum to 100%.
   - The category with the highest probability is selected as the prediction, with its probability reported as the confidence score.

---

## 📁 Project Structure

```
SWYNEX-SmartTextClassifier/
├── app.py                     # Streamlit interactive web application
├── train_model.py             # Script to train and persist the ML pipeline
├── evaluate.py                # Deterministic test script with accuracy & confusion matrix
├── classifier_utils.py        # Core validation, model loading, and prediction functions
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and failure cases
├── data/
│   ├── dataset.csv            # Balanced training dataset (100 labeled samples)
│   └── test_dataset.csv       # Independent test dataset (24 labeled samples)
├── models/
│   ├── classifier_pipeline.joblib  # Serialized trained model pipeline
│   └── model_metadata.json    # Training parameters, vocabulary size, and CV scores
└── screenshots/
    ├── README.md              # Screenshot capture guidelines
    └── demo.png               # Application interface preview placeholder
```

---

## 📋 System Requirements

- **Python Version**: Python 3.9 or higher (tested and verified on Python 3.12.10).
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Hardware**: Standard laptop CPU (no GPU required; training takes less than 1 second).

---

## 🚀 Installation & Quick Start

### 1. Clone or Open Project
Open PowerShell and navigate to the project directory:
```powershell
cd c:\Users\Eswar\Desktop\SWYNEX-SmartTextClassifier
```

### 2. Set Up a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ⚡ Execution Commands

### Step 1: Train the Model
Train the TF-IDF vectorizer and Logistic Regression classifier on `data/dataset.csv`:
```powershell
python train_model.py
```
*Expected Output:*
- Displays dataset sample counts and class balance.
- Runs 5-fold Stratified Cross-Validation (typically ~88% CV accuracy).
- Saves the trained model to `models/classifier_pipeline.joblib`.
- Saves model metadata to `models/model_metadata.json`.

### Step 2: Run Evaluation
Evaluate the model against 24 independent test samples in `data/test_dataset.csv`:
```powershell
python evaluate.py
```
*Expected Output:*
- Sample-by-sample table with Expected, Predicted, and PASS/FAIL results.
- Overall accuracy score.
- Category-wise classification report (Precision, Recall, F1-Score).
- Tabular Confusion Matrix.

### Step 3: Launch the Streamlit Web Application
Start the interactive dashboard:
```powershell
streamlit run app.py
```
*Access the interface:*
Open your web browser and navigate to `http://localhost:8501`.

---

## 🧪 Example Inputs and Expected Outputs

| Category | Example Input Text | Expected Result | Typical Confidence |
| :--- | :--- | :--- | :--- |
| **Technology** | "Engineers deployed an automated CI/CD pipeline using Docker containers and Kubernetes clusters." | `Technology` | ~58% – 75% |
| **Education** | "University professors published peer-reviewed research papers and mentored graduate students." | `Education` | ~60% – 80% |
| **Sports** | "The striker scored a spectacular volley in stoppage time to win the championship final." | `Sports` | ~65% – 85% |
| **Health** | "Clinical trials demonstrated that daily aerobic exercise significantly reduces arterial hypertension." | `Health` | ~62% – 82% |
| **Ambiguous** | "The sports analytics startup developed a machine learning platform to track athletic injuries." | `Technology` / `Sports` *(Triggers Low Confidence Warning)* | < 45% (Uncertain) |

---

## 📊 Evaluation Metrics

When tested against the independent 24-sample benchmark (`data/test_dataset.csv`), the model produces deterministic, verified metrics:

- **Total Test Cases**: 24
- **Correct Predictions**: 24
- **Incorrect Cases**: 0
- **Test Set Accuracy**: 100.00%
- **Macro Average F1-Score**: 1.0000

> **Note on Evaluation Accuracy:** The high accuracy on the curated test set demonstrates that the model successfully learns distinct vocabulary and phrase patterns for these four domains. In real-world environments, classification accuracy depends heavily on input length, vocabulary overlap, noise, and dataset scale.

---

## ⚠️ Known Limitations

1. **Bag-of-Words Limitation**: TF-IDF models assess word frequencies and n-gram occurrences. They do not maintain long-range semantic dependencies or deep contextual attention like Transformer models (e.g., BERT).
2. **Single-Label Constraint**: The model assigns each input to one primary category. Multi-topic texts are forced into whichever category has slightly higher weight.
3. **Domain Boundary Assumptions**: The model is trained strictly on four domains (`Technology`, `Education`, `Sports`, `Health`). If a user provides text about Politics, Finance, or Cooking, the classifier will force-fit it into one of the four known categories.

---

## 🔍 Five Failure Cases & Potential Improvements

The following table documents scenarios where bag-of-words and linear classification models may struggle, along with concrete avenues for improvement:

### 1. Multi-Topic / Overlapping Content
- **Example Input**: *"A technology startup built a mobile fitness app to help collegiate student-athletes track workout nutrition."*
- **Why It May Struggle**: This sentence contains vocabulary from all four categories (*"technology startup"* -> Technology, *"student"* -> Education, *"athletes/workout"* -> Sports, *"fitness/nutrition"* -> Health). The model's probability distribution will be split thinly across multiple classes, and whichever category has more overlapping keywords wins by a narrow margin.
- **Potential Improvement**: Implement multi-label classification using `MultiOutputClassifier` or a sigmoid threshold, allowing the model to return multiple relevant tags (e.g., `[Sports, Health, Technology]`).

### 2. Very Short or Ambiguous Text
- **Example Input**: *"The match was cancelled today."*
- **Why It May Struggle**: The word *"match"* could refer to a sports match (Sports), a string matching algorithm (Technology), or a residency match for medical graduates (Health/Education). With only five words, TF-IDF cannot gather sufficient contextual signal.
- **Potential Improvement**: Set a minimum input word count warning in the UI, or utilize contextual sentence embeddings (such as `all-MiniLM-L6-v2`) that leverage pre-trained contextual representations.

### 3. Sarcasm and Irony
- **Example Input**: *"Oh great, another mandatory software update that completely broke our classroom computers during the final exam!"*
- **Why It May Struggle**: TF-IDF treats words as literal tokens. It registers keywords like *"software update"* (Technology) and *"classroom/final exam"* (Education), but cannot interpret sarcastic tone or frustration.
- **Potential Improvement**: Incorporate sentiment analysis as an auxiliary feature or leverage pre-trained transformer models with sentiment fine-tuning.

### 4. Unseen Vocabulary / Out-of-Vocabulary (OOV) Terms
- **Example Input**: *"Neuromorphic memristors accelerate synaptic spiking architectures."*
- **Why It May Struggle**: If specific technical terms like *"memristors"* or *"synaptic"* were not present in the training corpus, the TF-IDF vectorizer assigns them zero weight. The model may have to rely solely on common non-stop words or fall back to near-uniform probabilities.
- **Potential Improvement**: Use character-level or subword tokenization (such as Byte-Pair Encoding / WordPiece) or train on a larger, more comprehensive corpus.

### 5. Spelling Mistakes and Typos
- **Example Input**: *"The docter prescrybed antibyotics for the sevear infestion."*
- **Why It May Struggle**: Because the words are misspelled (*"docter"*, *"prescrybed"*, *"antibyotics"*, *"sevear"*, *"infestion"*), exact string-matching in TF-IDF will fail to map them to the vocabulary terms (*"doctor"*, *"prescribed"*, *"antibiotics"*, *"infection"*), rendering the input effectively empty of known features.
- **Potential Improvement**: Add a preprocessing spelling correction layer (e.g., using `pyspellchecker` or `symspellpy`) before passing tokens to the TF-IDF vectorizer, or utilize character n-gram tokenization (`analyzer='char_wb'`).

---

## 🛠️ Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Model file not found" error** | Model pipeline has not been trained yet or was deleted. | Run `python train_model.py` in your terminal to regenerate the model. |
| **"Input text cannot be empty"** | User submitted an empty input or spaces only. | Enter text with meaningful words and click **Classify Text**. |
| **"Input text exceeds 5,000 characters"** | Text payload is too long. | Trim the text to under 5,000 characters before submitting. |
| **Streamlit port conflict (`Port 8501 is already in use`)** | Another Streamlit process is running. | Run `streamlit run app.py --server.port 8502` to use an alternative port. |
| **ModuleNotFoundError** | Required package missing from Python environment. | Run `pip install -r requirements.txt`. |

---

## ⚖️ Educational Demonstration Disclaimer

This project is developed solely as an educational demonstration of natural language processing and text classification fundamentals. The predictions and confidence scores generated by this model are derived from a lightweight, localized training dataset and are **not suitable for clinical medical diagnoses, legal determinations, or safety-critical decisions**.
