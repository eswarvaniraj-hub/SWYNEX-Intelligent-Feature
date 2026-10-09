# Smart Text Intelligence: Text Classification and Evaluation

A lightweight, explainable text classification app built with Python, scikit-learn and Streamlit. It classifies text into four categories: **Technology**, **Education**, **Sports** and **Health**. It shows a confidence score for every prediction and warns the user when the model is unsure.

> This project is an educational demonstration. It is not suitable for high-stakes decisions.

<img width="1815" height="941" alt="Screenshot 2026-10-09 195928" src="https://github.com/user-attachments/assets/d0b20628-7835-4ba1-a738-820723da830b" />

---

## Objectives

- Add an intelligent feature (text classification) to a simple prototype.
- Handle invalid input and failures gracefully.
- Evaluate the model on separate test data and document where it fails.
- Provide a simple interactive demo.

## Features

- Text classification into 4 categories using TF-IDF + Logistic Regression
- Confidence score and probability distribution for each prediction
- Low-confidence warning (below 45%)
- Input validation with friendly error messages
- Sample inputs to try instantly
- Evaluation script with PASS/FAIL results, classification report and confusion matrix
- Automated test suite for edge cases
- Runs fully locally: no API keys, no paid services

## Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.12 |
| Machine learning | scikit-learn (TF-IDF, Logistic Regression) |
| Web interface | Streamlit |
| Data handling | pandas, numpy |
| Model saving | joblib |

## How It Works

1. **TF-IDF** converts text into numbers. Words that are distinctive for a topic get higher weights than very common words.
2. **Logistic Regression** takes those numbers and calculates a probability for each of the four categories.
3. The category with the highest probability is the prediction, and that probability is shown as the **confidence score**.
4. If the top probability is below **45%**, the app warns the user that the prediction may be unreliable.

The model is trained locally on a small labeled dataset (`data/dataset.csv`). Predictions are never hardcoded or random.

## Folder Structure

```
SWYNEX-SmartTextClassifier/
├── app.py                 # Streamlit web interface
├── classifier_utils.py    # Input validation, model loading, prediction
├── train_model.py         # Trains and saves the model
├── evaluate.py            # Evaluates the model on the test dataset
├── test_suite.py          # Automated edge-case tests
├── requirements.txt
├── README.md
├── data/
│   ├── dataset.csv        # Training data (100 samples, 25 per category)
│   └── test_dataset.csv   # Test data (24 samples, no overlap with training)
├── models/
│   ├── classifier_pipeline.joblib
│   └── model_metadata.json
└── screenshots/
    └── demo.png
```

## Installation

Requires **Python 3.9 or newer** (developed and tested on Python 3.12).

```powershell
pip install -r requirements.txt
```

## Usage

**1. Train the model** (also regenerates the model files):
```powershell
python train_model.py
```

**2. Run the evaluation:**
```powershell
python evaluate.py
```

**3. Run the automated tests:**
```powershell
python test_suite.py
```

**4. Launch the web app:**
```powershell
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## Example Inputs

| Input | Expected Category |
|---|---|
| The cybersecurity team patched a critical vulnerability in the server. | Technology |
| University students submitted their term papers before the deadline. | Education |
| The tennis player served an ace on match point. | Sports |
| Physicians recommend regular exercise for heart health. | Health |

## Error Handling

The app handles these cases with friendly messages instead of crashing:

- Empty input
- Input containing only spaces or tabs
- Input longer than 5,000 characters
- Non-string input (numbers, `None`) passed to the prediction function
- Missing model file (the message tells the user to run `python train_model.py`)
- Model loading or prediction errors

These cases are verified by `test_suite.py`.

## Evaluation Results

| Metric | Result |
|---|---|
| Training samples | 100 (25 per category) |
| Test samples | 24 (6 per category, separate from training data) |
| Test accuracy | 100% (24 / 24 correct) |
| 5-fold cross-validation accuracy (training data) | 88% (±4%) |
| Automated tests | 9 / 9 passed |

**Note:** Accuracy depends on the quality and size of the dataset. The test set is small and its sentences are clear examples of each topic, so 100% on it does not mean the model will be perfect on real-world text. The cross-validation score (88%) is a more cautious estimate of general performance.

## Known Limitations

- The training dataset is small (100 samples), so vocabulary coverage is limited.
- The model only knows four categories. Any other topic will still be forced into one of them.
- It uses word frequency, not true understanding of meaning or context.
- English text only.

## Failure Cases and Possible Improvements

These are situations where the model may struggle. They do not always produce wrong predictions.

**1. Text containing multiple topics**
- Example: *"The sports analytics startup developed a machine learning platform to track athletic injuries."*
- Why it may fail: It mixes Sports, Technology and Health vocabulary, so the probabilities split and confidence drops.
- Improvement: Use multi-label classification so a text can have more than one category.

**2. Very short or ambiguous text**
- Example: *"Training session"*
- Why it may fail: Too few words give the model little evidence, and "training" could fit Sports, Education or Technology.
- Improvement: Require a minimum length, or ask the user for more context.

**3. Sarcasm**
- Example: *"Oh great, another brilliant software update that deleted all my files."*
- Why it may fail: The model counts words and cannot detect tone, so it relies on the technology words alone.
- Improvement: Use a model that understands context, such as a transformer-based model.

**4. Unseen vocabulary**
- Example: *"The pickleball tournament drew record crowds."*
- Why it may fail: Words not in the training data are ignored, so the model has little to go on.
- Improvement: Add more training data covering more vocabulary, or use pretrained word embeddings.

**5. Spelling mistakes**
- Example: *"The studants recieved thier examenation results."*
- Why it may fail: Misspelled words do not match the vocabulary learned during training.
- Improvement: Add spell-correction preprocessing or use character-level n-grams.

## Troubleshooting

| Problem | Solution |
|---|---|
| "Model file not found" message | Run `python train_model.py` to regenerate the model |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `streamlit` is not recognized | Run `python -m streamlit run app.py` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Screenshot not showing | Make sure the file is saved as `screenshots/demo.png` |

## Capturing the Screenshot

1. Run `streamlit run app.py` and open `http://localhost:8501`.
2. Select a sample and click **Classify Text** so the result and confidence score are visible.
3. Press `Win + Shift + S`, snip the window, and save it as `screenshots/demo.png`.

## Disclaimer

This project is an educational demonstration of text classification. It is trained on a small dataset and is not intended for high-stakes or production use.
 
 
 
 
