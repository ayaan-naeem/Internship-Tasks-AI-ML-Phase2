import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('TASK 5: Auto Tagging Support Tickets Using LLM')
print('='*60)

# ── 1. Load Dataset ────────────────────────────────────────────
df = pd.read_csv('task5_auto_tag_tickets/support_tickets.csv')
print(f'\nDataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'\nCategory distribution:')
print(df['category'].value_counts().to_string())

candidate_labels = df['category'].unique().tolist()
print(f'\nCandidate tags: {candidate_labels}')

# Use a sample for speed (zero-shot on CPU is slow)
df_sample = df.sample(n=120, random_state=42).reset_index(drop=True)
print(f'\nUsing {len(df_sample)} samples for evaluation')

# ── 2. EDA Plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Support Ticket Dataset Overview', fontsize=14, fontweight='bold')

vc = df['category'].value_counts()
colors_eda = sns.color_palette('Set2', len(vc))
axes[0].barh(vc.index, vc.values, color=colors_eda)
axes[0].set_title('Tickets per Category')
axes[0].set_xlabel('Count')

df['text_length'] = df['text'].str.len()
sns.boxplot(data=df, x='category', y='text_length', ax=axes[1], palette='Set2')
axes[1].set_title('Ticket Text Length by Category')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('task5_auto_tag_tickets/eda_plots.png', dpi=150, bbox_inches='tight')
print('EDA plots saved.')

# ── 3. Zero-Shot Classification (BART-MNLI) ────────────────────
from transformers import pipeline as hf_pipeline

print('\nLoading zero-shot classifier (facebook/bart-large-mnli)...')
print('(First run downloads ~1.5GB - please wait)')
zero_shot = hf_pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=-1  # CPU
)
print('Zero-shot model loaded!')

print('Running zero-shot classification on 120 samples...')
zs_preds = []
zs_top3  = []
for i, row in df_sample.iterrows():
    result = zero_shot(row['text'], candidate_labels=candidate_labels, multi_label=False)
    top1   = result['labels'][0]
    top3   = list(zip(result['labels'][:3], [round(s, 3) for s in result['scores'][:3]]))
    zs_preds.append(top1)
    zs_top3.append(top3)
    if (i + 1) % 20 == 0:
        print(f'  Processed {i+1}/{len(df_sample)}')

df_sample['zs_pred'] = zs_preds
df_sample['zs_top3'] = zs_top3

zs_acc = accuracy_score(df_sample['category'], df_sample['zs_pred'])
zs_f1  = f1_score(df_sample['category'], df_sample['zs_pred'], average='macro')
print(f'\n--- Zero-Shot Results (BART-large-MNLI) ---')
print(f'Accuracy : {zs_acc:.4f}')
print(f'Macro F1 : {zs_f1:.4f}')
print(classification_report(df_sample['category'], df_sample['zs_pred']))

# ── 4. Few-Shot with Flan-T5 ───────────────────────────────────
from transformers import T5ForConditionalGeneration, T5Tokenizer

print('\nLoading few-shot model (google/flan-t5-base)...')
print('(First run downloads ~900MB - please wait)')
tokenizer_t5 = T5Tokenizer.from_pretrained('google/flan-t5-base')
model_t5     = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
print('Flan-T5 loaded!')

few_shot_prompt = """Classify the support ticket into exactly one of these categories:
Billing Issue, Technical Support, Account Management, Shipping & Delivery, Product Inquiry, Cancellation & Refund

Examples:
Ticket: "I was charged twice for my subscription this month." -> Billing Issue
Ticket: "The app keeps crashing whenever I open it." -> Technical Support
Ticket: "I forgot my password and cannot reset it." -> Account Management
Ticket: "My order has not arrived after 2 weeks." -> Shipping & Delivery
Ticket: "Does your product support integration with Slack?" -> Product Inquiry
Ticket: "I want to cancel my subscription immediately." -> Cancellation & Refund

Ticket: "{text}" -> """

def few_shot_predict(text):
    prompt = few_shot_prompt.format(text=text[:200])
    inputs  = tokenizer_t5(prompt, return_tensors='pt', max_length=512, truncation=True)
    outputs = model_t5.generate(**inputs, max_new_tokens=15)
    pred    = tokenizer_t5.decode(outputs[0], skip_special_tokens=True).strip()
    # Match to closest valid label
    pred_lower = pred.lower()
    for lbl in candidate_labels:
        if lbl.lower() in pred_lower or pred_lower in lbl.lower():
            return lbl
    # Keyword fallback
    kw = {
        'Billing Issue':          ['billing','charged','invoice','charge','payment','billed'],
        'Technical Support':      ['technical','crash','error','bug','slow','not working','broken'],
        'Account Management':     ['account','password','login','email','username','locked'],
        'Shipping & Delivery':    ['ship','deliver','order','package','tracking','arrived'],
        'Product Inquiry':        ['product','plan','trial','feature','support','integration','api'],
        'Cancellation & Refund':  ['cancel','refund','money back','unsubscribe'],
    }
    for lbl, words in kw.items():
        if any(w in pred_lower for w in words):
            return lbl
    return candidate_labels[0]

print('Running few-shot classification on 120 samples...')
fs_preds = []
for i, row in df_sample.iterrows():
    pred = few_shot_predict(row['text'])
    fs_preds.append(pred)
    if (i + 1) % 20 == 0:
        print(f'  Processed {i+1}/{len(df_sample)}')

df_sample['fs_pred'] = fs_preds
fs_acc = accuracy_score(df_sample['category'], df_sample['fs_pred'])
fs_f1  = f1_score(df_sample['category'], df_sample['fs_pred'], average='macro')
print(f'\n--- Few-Shot Results (Flan-T5-base) ---')
print(f'Accuracy : {fs_acc:.4f}')
print(f'Macro F1 : {fs_f1:.4f}')
print(classification_report(df_sample['category'], df_sample['fs_pred']))

# ── 5. Comparison Plots ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Zero-Shot vs Few-Shot Performance', fontsize=14, fontweight='bold')

metrics  = ['Accuracy', 'Macro F1']
zs_vals  = [zs_acc, zs_f1]
fs_vals  = [fs_acc, fs_f1]
x        = np.arange(len(metrics))
width    = 0.35

axes[0].bar(x - width/2, zs_vals, width, label='Zero-Shot (BART)', color='#2196F3')
axes[0].bar(x + width/2, fs_vals, width, label='Few-Shot (Flan-T5)', color='#4CAF50')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Overall Performance Comparison')
axes[0].legend()
axes[0].set_ylabel('Score')
for i, (v1, v2) in enumerate(zip(zs_vals, fs_vals)):
    axes[0].text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', fontsize=11, fontweight='bold')
    axes[0].text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', fontsize=11, fontweight='bold')

# Per-category accuracy
cat_zs = df_sample.groupby('category').apply(
    lambda g: (g['category'] == g['zs_pred']).mean(), include_groups=False)
cat_fs = df_sample.groupby('category').apply(
    lambda g: (g['category'] == g['fs_pred']).mean(), include_groups=False)
cats   = cat_zs.index.tolist()
x2     = np.arange(len(cats))
axes[1].bar(x2 - width/2, cat_zs.values, width, label='Zero-Shot', color='#2196F3')
axes[1].bar(x2 + width/2, cat_fs.values, width, label='Few-Shot',  color='#4CAF50')
axes[1].set_xticks(x2)
axes[1].set_xticklabels([c.replace(' & ', '\n& ') for c in cats], rotation=15, fontsize=8)
axes[1].set_ylim(0, 1.15)
axes[1].set_title('Per-Category Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('task5_auto_tag_tickets/comparison_plot.png', dpi=150, bbox_inches='tight')
print('\nComparison plot saved.')

# ── 6. Top-3 Tags Output ───────────────────────────────────────
print('\n--- Sample: Top-3 Tags per Ticket (Zero-Shot) ---')
for _, row in df_sample.head(8).iterrows():
    print(f'Ticket : {row["text"][:70]}...')
    print(f'Actual : {row["category"]}')
    print(f'Top-3  : {row["zs_top3"]}')
    print()

# Save predictions
df_sample[['ticket_id', 'text', 'category', 'zs_pred', 'fs_pred', 'zs_top3']].to_csv(
    'task5_auto_tag_tickets/predictions.csv', index=False)
print('Predictions saved to predictions.csv')

print('\n' + '='*60)
print('TASK 5 COMPLETE')
print(f'Zero-Shot (BART-MNLI) : Accuracy={zs_acc:.4f}, F1={zs_f1:.4f}')
print(f'Few-Shot  (Flan-T5)   : Accuracy={fs_acc:.4f}, F1={fs_f1:.4f}')
print('='*60)
