# app.py
import torch

if hasattr(torch, 'classes'):
    try:
        _ = dir(torch.classes)
    except Exception:
        pass

import streamlit as st
import pandas as pd
from models.inference import get_models, run_inference
from evaluation.metrics import evaluate_task, compare_models

st.set_page_config(page_title="NLP Task Evaluator", layout="wide")
st.title("\U0001F9E0 NLP Model Evaluation Dashboard")

# Load dataset
df = pd.read_csv("data/sample_nlp_dataset.csv")

# Task selector
task = st.sidebar.selectbox("Select Task", ["Sentiment", "Summarization", "Text Generation"])

# Model selector
available_models = get_models(task)
selected_models = st.sidebar.multiselect("Choose Models", available_models, default=available_models[:1])

all_results = {}

if st.button("Run Evaluation"):
    for model_name in selected_models:
        st.subheader(f"Model: {model_name}")

        # Inference
        df_pred = run_inference(df.copy(), task, model_name)

        # Evaluation
        metric_result, plots = evaluate_task(df_pred, task)

        all_results[model_name] = metric_result

        # Show scores & plots
        st.write(metric_result)
        for plot in plots:
            st.pyplot(plot)

    # Comparison Plot
    if len(selected_models) > 1:
        st.subheader("\U0001F4CA Model Comparison")
        comp_fig = compare_models(all_results, task)
        st.pyplot(comp_fig)
