import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_rouge(rouge_scores):
    keys = list(rouge_scores.keys())
    values = [rouge_scores[k] for k in keys]  # directly use float scores

    fig, ax = plt.subplots()
    ax.bar(keys, values, color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_title('ROUGE Score Comparison')
    ax.set_ylabel('Score')
    return fig


def plot_bertscore(score_dict):
    precision = sum(score_dict["precision"]) / len(score_dict["precision"])
    recall = sum(score_dict["recall"]) / len(score_dict["recall"])
    f1 = sum(score_dict["f1"]) / len(score_dict["f1"])
    
    fig, ax = plt.subplots()
    sns.barplot(x=["Precision", "Recall", "F1"], y=[precision, recall, f1], ax=ax)
    ax.set_title("BERTScore")
    ax.set_ylim(0, 1)
    return fig

def plot_model_comparison(results_dict, task):
    fig, ax = plt.subplots()
    model_names = list(results_dict.keys())

    if task == "Sentiment":
        values = [results_dict[m]["weighted avg"]["f1-score"] for m in model_names]
        ylabel = "F1 Score"
    elif task == "Summarization":
        values = [results_dict[m]["rougeL"] for m in model_names]
        ylabel = "ROUGE-L F1"
    elif task == "Text Generation":
        values = [sum(results_dict[m]["f1"]) / len(results_dict[m]["f1"]) for m in model_names]
        ylabel = "BERTScore F1"
    else:
        values, ylabel = [], ""

    sns.barplot(x=model_names, y=values, ax=ax)
    ax.set_title(f"{task} - Model Comparison")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    return fig
