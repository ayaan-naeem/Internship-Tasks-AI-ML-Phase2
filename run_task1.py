import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('TASK 1: News Topic Classifier Using BERT')
print('='*60)

# ── 1. Load Dataset ────────────────────────────────────────────
train_df = pd.read_csv('task1_bert_news_classifier/ag_news_train.csv')
test_df  = pd.read_csv('task1_bert_news_classifier/ag_news_test.csv')

label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
print(f'\nTrain shape: {train_df.shape}')
print(f'Test shape : {test_df.shape}')
print(f'\nLabel distribution (train):\n{train_df["label_name"].value_counts().to_string()}')

# ── 2. EDA Plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('AG News Dataset - EDA', fontsize=15, fontweight='bold')

colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
vc = train_df['label_name'].value_counts()
axes[0].bar(vc.index, vc.values, color=colors)
axes[0].set_title('Class Distribution (Train)')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Category')

train_df['text_len'] = train_df['text'].str.split().str.len()
sns.boxplot(data=train_df, x='label_name', y='text_len', ax=axes[1], palette=colors)
axes[1].set_title('Text Length by Category (words)')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Word Count')

plt.tight_layout()
plt.savefig('task1_bert_news_classifier/eda_plots.png', dpi=150, bbox_inches='tight')
print('\nEDA plots saved.')

# ── 3. Load Pre-Fine-Tuned BERT Model ─────────────────────────
print('\nLoading pre-fine-tuned BERT model (fabriceyhc/bert-base-uncased-ag_news)...')
print('(Downloads ~440MB on first run)')

MODEL_NAME = 'fabriceyhc/bert-base-uncased-ag_news'
id2label = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

classifier = pipeline(
    'text-classification',
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1,       # CPU
    truncation=True,
    max_length=128
)
print('Model loaded successfully!')

# ── 4. Evaluate on Test Set ────────────────────────────────────
# Use 500 samples for speed on CPU
test_sample = test_df.sample(n=500, random_state=42).reset_index(drop=True)
print(f'\nEvaluating on {len(test_sample)} test samples...')

preds_raw = classifier(test_sample['text'].tolist(), batch_size=16)
# Map model output labels to our label names
label_remap = {'LABEL_0': 'World', 'LABEL_1': 'Sports',
               'LABEL_2': 'Business', 'LABEL_3': 'Sci/Tech'}
preds = [label_remap.get(p['label'], p['label']) for p in preds_raw]

y_true = test_sample['label_name'].tolist()
acc = accuracy_score(y_true, preds)
f1  = f1_score(y_true, preds, average='macro')

print(f'\n--- BERT Evaluation Results ---')
print(f'Accuracy  : {acc:.4f}')
print(f'Macro F1  : {f1:.4f}')
print(f'\nClassification Report:')
print(classification_report(y_true, preds,
      target_names=['World','Sports','Business','Sci/Tech']))

# ── 5. Evaluation Plots ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('BERT News Classifier - Evaluation Results', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_true, preds,
                      labels=['World','Sports','Business','Sci/Tech'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['World','Sports','Business','Sci/Tech'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')
axes[0].tick_params(axis='x', rotation=15)

# Per-class F1
from sklearn.metrics import f1_score
per_class_f1 = f1_score(y_true, preds, average=None,
                         labels=['World','Sports','Business','Sci/Tech'])
axes[1].bar(['World','Sports','Business','Sci/Tech'], per_class_f1, color=colors)
axes[1].set_title('Per-Class F1 Score')
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0, 1.0)
for i, v in enumerate(per_class_f1):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('task1_bert_news_classifier/evaluation_plots.png', dpi=150, bbox_inches='tight')
print('\nEvaluation plots saved.')

# ── 6. Sample Predictions ──────────────────────────────────────
print('\n--- Sample Predictions ---')
for i in range(5):
    row = test_sample.iloc[i]
    print(f'Text    : {row["text"][:80]}...')
    print(f'Actual  : {row["label_name"]}')
    print(f'BERT    : {preds[i]}')
    print()

print('='*60)
print('TASK 1 COMPLETE')
print(f'Model     : fabriceyhc/bert-base-uncased-ag_news')
print(f'Accuracy  : {acc:.4f}')
print(f'Macro F1  : {f1:.4f}')
print('='*60)
