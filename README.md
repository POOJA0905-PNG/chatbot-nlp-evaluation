# chatbot-nlp-evaluation
Chatbot NLP Evaluation Dashboard is a comprehensive tool designed to evaluate chatbot-generated text across three major NLP tasks:
Summarization – Measures how well the chatbot can summarize conversations or content.
Question-Answering – Evaluates the chatbot’s ability to respond to questions accurately.
Sentiment Analysis – Determines the emotional tone of chatbot responses (positive, negative, or neutral).
This project integrates fine-tuned transformer models including RoBERTa-large, LLaMA-2 7B, and GPT-2, offering real-time inference, comparison, and visualization through an interactive Streamlit dashboard.
It aims to simplify the selection and evaluation of large language models (LLMs) for developers, researchers, and non-technical users alike, using intuitive visual feedback and metric-based analysis.

Key Features:
Dynamic Model Selection based on input type
Multi-model Evaluation for NLP tasks
Real-time Results with graphical visualizations
Fine-tuned Models for improved accuracy
Scalable Architecture for future NLP task integration
User-Friendly Interface built with Streamlit

Structure of the Project:
nlp_evaluator/
│
├── app.py                         # 🚀 Main Streamlit app file – connects UI to backend logic
│
├── data/                          # 📁 Contains datasets for evaluation
│   └── sample_nlp_dataset.csv     # A sample CSV used for testing and evaluation
│
├── evaluation/                    # 📁 Module for evaluation logic
│   ├── metrics.py                 # 📊 Calculates metrics (accuracy, F1, etc.)
│
├── models/                        # 📁 Handles inference logic
│   └── inference.py              # 🧠 Loads models and runs predictions on input text
│
├── utils/                         # 📁 Utility scripts (if any added later)
│    └── plot_helpers.py           # 📈 Functions to create visual plots like confusion matrix
│
├── venv/                          # 📁 Virtual environment (auto-created folder for dependencies)
│
└── README.md                      # 📄 Project description, setup guide, usage, etc.
