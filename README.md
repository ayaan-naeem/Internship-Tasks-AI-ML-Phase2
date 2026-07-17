# AI/ML Engineering – Advanced Internship Tasks

**Name:** Muhammad Ayaan Naeem
**DHC-ID:** 5016
**Organization:** DevelopersHub Corporation
**Internship Track:** AI/ML Engineering – Advanced Internship Tasks
**Due Date:** 21st July, 2026

This repository contains my submission for **3 of the 5 Advanced Tasks**:
- Task 1: News Topic Classifier Using BERT
- Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
- Task 5: Auto Tagging Support Tickets Using LLM

---

## Repository Structure

```
├── task1_news_topic_classifier_bert/
│   ├── news_topic_classifier_bert.ipynb
├── task2_ml_pipeline_churn_prediction/
│   ├── ml_pipeline_churn_prediction.ipynb
├── task5_auto_tagging_support_tickets/
│   └── auto_tagging_support_tickets.ipynb
```

---

## Task 1: News Topic Classifier Using BERT

### Objective
Fine-tune a transformer model (BERT) to classify news headlines into topic categories.

### Methodology / Approach
- Loaded and explored the **AG News Dataset** (Hugging Face Datasets)
- Tokenized and preprocessed headlines using the `bert-base-uncased` tokenizer
- Fine-tuned `bert-base-uncased` using Hugging Face `Transformers` (`Trainer` API)
- Evaluated performance using **accuracy** and **F1-score**
- Deployed the trained model as an interactive demo using **Streamlit**

### Key Results / Observations
- Accuracy: _[fill in after training]_
- F1-score: _[fill in after training]_
- Observations: _[e.g., which topic categories were most/least confused, training time, dataset size used]_

---

## Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

### Objective
Build a reusable, production-ready ML pipeline to predict customer churn.

### Methodology / Approach
- Loaded and cleaned the **Telco Customer Churn Dataset**
- Built preprocessing steps (scaling numerical features, encoding categorical features) using `sklearn.pipeline.Pipeline` and `ColumnTransformer`
- Trained and compared **Logistic Regression** and **Random Forest** classifiers
- Tuned hyperparameters using `GridSearchCV`
- Exported the final trained pipeline using `joblib` for reuse/deployment

### Key Results / Observations
- Best model: _[e.g., Random Forest / Logistic Regression]_
- Best hyperparameters: _[fill in]_
- Accuracy / F1 / ROC-AUC: _[fill in]_
- Observations: _[e.g., most predictive features, class imbalance handling]_

---

## Task 5: Auto Tagging Support Tickets Using LLM

### Objective
Automatically tag free-text support tickets into categories using an LLM.

### Methodology / Approach
- Loaded a free-text support ticket dataset
- Designed prompts for **zero-shot** classification using an LLM
- Applied **few-shot prompting** with labeled examples to improve accuracy
- Compared zero-shot vs. few-shot (and optionally fine-tuned) performance
- Generated the **top 3 most probable tags** per ticket, ranked by confidence

### Key Results / Observations
- Zero-shot accuracy: _[fill in]_
- Few-shot accuracy: _[fill in]_
- Observations: _[e.g., which categories were hardest to classify, effect of example count on accuracy]_

---

## Tools & Technologies Used
- Hugging Face Transformers
- scikit-learn (Pipeline, GridSearchCV, joblib)
- Streamlit
- LLM prompting (zero-shot / few-shot)
- Python, Jupyter Notebook

## Submission Notes
- Each task notebook includes: problem statement, dataset loading & preprocessing, model development & training, evaluation metrics, visualizations, and a final summary of insights.
- Code is organized with clear structure and comments explaining major steps.
- GitHub repo link submitted via Google Classroom for each completed task.
