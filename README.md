# Customer-Support-Ticket-Analyzer
A machine learning-powered tool for analyzing customer support tickets. It performs:

- **Issue Type Classification**
- **Urgency Level Prediction**
- **Entity Extraction** (Products, Dates, Complaint Keywords)
- Optional **Gradio Web Interface** for interactive use

---

## 📁 Dataset

The project uses a dataset with the following columns:

- `ticket_id`: Unique ID for each ticket
- `ticket_text`: The text of the support ticket
- `issue_type`: Type of issue (label)
- `urgency_level`: Level of urgency (`Low`, `Medium`, `High`) – label
- `product`: Ground truth product name

---

## 🧹 Data Preprocessing

- Lowercasing and removing special characters
- Tokenization
- Stopword removal
- Lemmatization
- Null value handling (if any)

---

## ⚙️ Feature Engineering

- TF-IDF Vectorization
- Additional features:
  - Ticket length
  - Sentiment score (can be extended)

---

## 🧠 Model Training (Multi-task Learning)

Two separate classifiers are trained:

1. **Issue Type Classifier** – Best: `Random Forest` / `Logistic Regression`
2. **Urgency Level Classifier** – Best: `SVM`

Classical ML models used:
- Logistic Regression
- Random Forest
- SVM

Evaluation metrics:
- Accuracy
- Macro Precision
- Classification Report

---

## 🔍 Entity Extraction

Extracted using rule-based methods:
- **Products**: From predefined list
- **Dates**: Regex-based (e.g., “13 April”, “04 March” → `13-April-2025`)
- **Complaint Keywords**: From issue_type labels

Returns extracted entities as a dictionary.

---

## 🛠 Main Function

```python
analyze_ticket(ticket_text)
```

Returns:
- Predicted issue type
- Predicted urgency level
- Extracted entities in JSON format

---

## 🌐 Gradio Interface (Optional)

Interactive interface using Gradio. Users can:

- Input raw support ticket text
- Get model predictions and extracted entities

### To Run:

```bash
pip install -r requirements.txt
python app.py
```

---

## 📦 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- nltk, re
- gradio

Install with:

```bash
pip install pandas numpy scikit-learn nltk gradio
```

---

## 📄 Example Input

```
"Order #53356 for RoboChef Blender is 18 days late. Ordered on 13 April."
```

### Output:
```json
{
  "issue_type": "late delivery",
  "urgency_level": "high",
  "entities": {
    "products": ["robochef blender"],
    "dates": ["13-april-2025"],
    "complaints": ["late delivery"]
  }
}
```

---

## 📌 Notes

- Extendable to include Named Entity Recognition with spaCy or transformers.
- Can be deployed as a web app or API.
