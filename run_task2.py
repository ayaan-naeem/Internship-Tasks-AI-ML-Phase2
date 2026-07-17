import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
import joblib
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('TASK 2: End-to-End ML Pipeline - Telco Churn Prediction')
print('='*60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('task2_ml_pipeline_churn/telco_churn.csv')
print(f'\nDataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'\nChurn distribution:\n{df["Churn"].value_counts()}')

# ── 2. Preprocessing ──────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df.drop('customerID', axis=1, inplace=True)
print(f'\nAfter cleaning: {df.shape}, Churn rate: {df["Churn"].mean():.2%}')

# ── 3. EDA Plots ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Telco Churn - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Churn distribution
churn_counts = df['Churn'].value_counts()
axes[0,0].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
              colors=['#2196F3','#F44336'], startangle=90)
axes[0,0].set_title('Churn Distribution')

# Monthly Charges by Churn
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[0,1],
            palette=['#2196F3','#F44336'])
axes[0,1].set_title('Monthly Charges by Churn')
axes[0,1].set_xticklabels(['No Churn', 'Churn'])

# Tenure by Churn
sns.histplot(data=df, x='tenure', hue='Churn', ax=axes[1,0],
             palette=['#2196F3','#F44336'], bins=30)
axes[1,0].set_title('Tenure Distribution by Churn')

# Contract type
contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
axes[1,1].bar(contract_churn.index, contract_churn.values,
              color=['#F44336','#FF9800','#4CAF50'])
axes[1,1].set_title('Churn Rate by Contract Type')
axes[1,1].set_ylabel('Churn Rate')
axes[1,1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('task2_ml_pipeline_churn/eda_plots.png', dpi=150, bbox_inches='tight')
print('\nEDA plots saved.')

# ── 4. Feature/Target Split ───────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f'Numeric cols: {numeric_cols}')
print(f'Categorical cols: {categorical_cols}')

# ── 5. Build Pipelines ────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Logistic Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:,1]

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_prob_lr)
print(f'\n--- Logistic Regression ---')
print(f'Accuracy : {lr_acc:.4f}')
print(f'F1 Score : {lr_f1:.4f}')
print(f'ROC-AUC  : {lr_auc:.4f}')

# Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:,1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)
print(f'\n--- Random Forest ---')
print(f'Accuracy : {rf_acc:.4f}')
print(f'F1 Score : {rf_f1:.4f}')
print(f'ROC-AUC  : {rf_auc:.4f}')

# ── 6. GridSearchCV ───────────────────────────────────────────
print('\nRunning GridSearchCV (this may take a minute)...')
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, scoring='f1',
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
print(f'Best params : {grid_search.best_params_}')
print(f'Best CV F1  : {grid_search.best_score_:.4f}')

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:,1]
best_acc = accuracy_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)
best_auc = roc_auc_score(y_test, y_prob_best)
print(f'\n--- Best Tuned Model (Test Set) ---')
print(f'Accuracy : {best_acc:.4f}')
print(f'F1 Score : {best_f1:.4f}')
print(f'ROC-AUC  : {best_auc:.4f}')
print(f'\nClassification Report:\n{classification_report(y_test, y_pred_best, target_names=["No Churn","Churn"])}')

# ── 7. Evaluation Plots ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn','Churn'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix (Best RF)')

# ROC Curves
RocCurveDisplay.from_predictions(y_test, y_prob_lr, ax=axes[1],
    name=f'Logistic Regression (AUC={lr_auc:.3f})', color='#2196F3')
RocCurveDisplay.from_predictions(y_test, y_prob_rf, ax=axes[1],
    name=f'Random Forest (AUC={rf_auc:.3f})', color='#FF9800')
RocCurveDisplay.from_predictions(y_test, y_prob_best, ax=axes[1],
    name=f'Tuned RF (AUC={best_auc:.3f})', color='#4CAF50')
axes[1].set_title('ROC Curves')
axes[1].plot([0,1],[0,1],'k--')

# Model Comparison Bar Chart
models = ['Logistic\nRegression', 'Random\nForest', 'Tuned\nRandom Forest']
accuracies = [lr_acc, rf_acc, best_acc]
f1_scores = [lr_f1, rf_f1, best_f1]
aucs = [lr_auc, rf_auc, best_auc]
x = np.arange(len(models))
width = 0.25
axes[2].bar(x - width, accuracies, width, label='Accuracy', color='#2196F3')
axes[2].bar(x, f1_scores, width, label='F1 Score', color='#4CAF50')
axes[2].bar(x + width, aucs, width, label='ROC-AUC', color='#FF9800')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models)
axes[2].set_ylim(0.6, 1.0)
axes[2].set_title('Model Comparison')
axes[2].legend()
axes[2].set_ylabel('Score')

plt.tight_layout()
plt.savefig('task2_ml_pipeline_churn/evaluation_plots.png', dpi=150, bbox_inches='tight')
print('Evaluation plots saved.')

# Feature Importance
rf_clf = best_model.named_steps['classifier']
feature_names_num = numeric_cols
feature_names_cat = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
all_features = feature_names_num + feature_names_cat
importances = rf_clf.feature_importances_
feat_df = pd.DataFrame({'feature': all_features, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances (Tuned Random Forest)', fontsize=14)
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('task2_ml_pipeline_churn/feature_importance.png', dpi=150, bbox_inches='tight')
print('Feature importance plot saved.')

# ── 8. Export Pipeline ────────────────────────────────────────
joblib.dump(best_model, 'task2_ml_pipeline_churn/churn_pipeline.pkl')
print('\nPipeline exported to churn_pipeline.pkl')

# Verify it loads and predicts
loaded = joblib.load('task2_ml_pipeline_churn/churn_pipeline.pkl')
sample = X_test.iloc[:3]
pred = loaded.predict(sample)
prob = loaded.predict_proba(sample)[:,1]
print(f'Pipeline verify - predictions: {pred}, proba: {prob.round(3)}')

print('\n' + '='*60)
print('TASK 2 COMPLETE')
print(f'Final Model  : Tuned Random Forest')
print(f'Accuracy     : {best_acc:.4f}')
print(f'F1 Score     : {best_f1:.4f}')
print(f'ROC-AUC      : {best_auc:.4f}')
print('='*60)
