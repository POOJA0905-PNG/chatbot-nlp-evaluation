# models/inference.py
from transformers import pipeline

# Dictionary to store model IDs for each task
MODEL_ZOO = {
    "Sentiment": {
        "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
        "RoBERTa": "siebert/sentiment-roberta-large-english"
    },
    "Summarization": {
        "T5-small": "t5-small",
        "BART": "facebook/bart-large-cnn"
    },
    "Text Generation": {
        "GPT2": "gpt2",
        "GPT-Neo": "EleutherAI/gpt-neo-125M"
    }
}

# Function to retrieve model names for a selected task
def get_models(task):
    return list(MODEL_ZOO[task].keys())

# Function to run inference based on task and model
def run_inference(df, task, model_name):
    # Mapping display task names to actual Hugging Face task names
    task_map = {
        "Sentiment": "sentiment-analysis",
        "Summarization": "summarization",
        "Text Generation": "text-generation"
    }

    if task not in task_map:
        raise ValueError(f"Unsupported task: {task}")

    # Load the pipeline with the correct task name and model ID
    model_id = MODEL_ZOO[task][model_name]
    pipe = pipeline(task_map[task], model=model_id)

    # Apply the pipeline to each row in the dataframe
    if task == "Sentiment":
        df['pred'] = df['text'].apply(lambda x: pipe(x)[0]['label'])
    elif task == "Summarization":
        df['pred'] = df['text'].apply(lambda x: pipe(x[:1024], max_length=50, min_length=10, do_sample=False)[0]['summary_text'])
    elif task == "Text Generation":
        df['pred'] = df['text'].apply(lambda x: pipe(x, max_new_tokens=50)[0]['generated_text'])

    return df
