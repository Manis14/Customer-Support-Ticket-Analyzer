import pandas as pd
import numpy as np
import re
import string
import json
import gradio as gr
import joblib


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dateutil.parser import parse
import warnings 




# === Entity Extraction ===
products = ['smartwatch v2', 'soundwave 300', 'photosnap cam', 'ecobreeze ac',
                'robochef blender', 'powermax battery', 'vision led tv',
                'protab x1', 'fitrun treadmill', 'ultraclean vacuum']

complaints= ['broken', 'not working', 'late', 'error', 'issue', 'defective', 'cracked', 'missing', 'damaged', 'faulty']
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
    
def extract_entities(text):
    entities = {'products': [], 'dates': [], 'complaints': []}
    text_lower = text.lower()

    for product in products:
        if product in text_lower:
            entities['products'].append(product)

    for complaint in complaints:
        if complaint in text_lower:
            entities['complaints'].append(complaint)

    date_matches = re.findall(r'\b(?:\d{1,2}\s)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', text_lower)
    for date_str in date_matches:
        try:
            parsed_date = parse(date_str, fuzzy=True, default=pd.Timestamp('2025-01-01'))
            entities['dates'].append(parsed_date.strftime('%d-%B-%Y'))
        except Exception:
            pass

    return entities


# === Analysis Function ===
def analyze_ticket(ticket_text):
    vectorizer = joblib.load('vectorizer.pkl')
    issue_model = joblib.load('issue_model.pkl')
    urgency_model = joblib.load('urgency_model.pkl')
    le_issue = joblib.load('le_issue.pkl')
    le_urgency = joblib.load('le_urgency.pkl')
    
    text_cleaned = preprocess(ticket_text)
    X_input = vectorizer.transform([text_cleaned])

   # Predict
    issue_pred = issue_model.predict(X_input)
    urgency_pred = urgency_model.predict(X_input)

    # Decode labels
    issue = le_issue.inverse_transform(issue_pred)[0]
    urgency = le_urgency.inverse_transform(urgency_pred)[0]

    # Extract entities
    entities = extract_entities(ticket_text)
    
    return {
        "Predicted Issue Type": issue,
        "Predicted Urgency Level": urgency,
        "Extracted Entities": entities
    }

# === Gradio Interface ===
def gradio_interface(ticket_text):
    result = analyze_ticket(ticket_text)
    print(result)
    return json.dumps(result, indent=2)
    
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=5, placeholder="Enter customer ticket text here..."),
    outputs="text",
    title="Customer Support Ticket Analyzer",
    description="Predict issue type, urgency level, and extract entities from support tickets."
)

iface.launch()
