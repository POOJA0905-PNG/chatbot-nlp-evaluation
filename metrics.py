from utils.plot_helpers import (
    plot_confusion,
    plot_rouge,
    plot_bertscore,
    plot_model_comparison
)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from evaluate import load as load_metric
import matplotlib.pyplot as plt
import seaborn as sns
import os
from shutil import rmtree

# Load external evaluation metrics
rouge = load_metric("rouge")
bertscore = load_metric("bertscore")

def evaluate_task(df, task):
    plots = []

    if task == "Sentiment":
        y_true = df['sentiment']
        y_pred = df['pred']

        # Confusion Matrix
        plots.append(plot_confusion(y_true, y_pred, labels=["POSITIVE", "NEGATIVE"]))

        report = classification_report(y_true, y_pred, output_dict=True)
        return report, plots

    elif task == "Summarization":
        scores = rouge.compute(predictions=df['pred'], references=df['summary'])

        # ROUGE Plot
        plots.append(plot_rouge(scores))

        return scores, plots

    elif task == "Text Generation":
        try:
            scores = bertscore.compute(
                predictions=df['pred'],
                references=df['expected_generation'],
                lang="en",
                use_fast_tokenizer=False
            )
        except FileNotFoundError:
            print("⚠️ Cache issue detected. Clearing BERTScore cache and retrying...")
            rmtree(os.path.expanduser("~/.cache/huggingface/metrics/bert_score"), ignore_errors=True)
            scores = bertscore.compute(
                predictions=df['pred'],
                references=df['expected_generation'],
                lang="en",
                use_fast_tokenizer=False
            )

        # BERTScore Plot
        plots.append(plot_bertscore(scores))

        return scores, plots

def compare_models(model_scores):
    """
    Compare multiple models based on their evaluation scores.
    Input:
        model_scores: dict
            Format - {
                "Model A": {"rouge1": 0.45, "rougeL": 0.42},
                "Model B": {"rouge1": 0.48, "rougeL": 0.44}
            }
    Output:
        A comparison plot (e.g., bar plot)
    """
    return plot_model_comparison(model_scores)
