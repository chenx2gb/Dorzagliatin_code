#!/usr/bin/env python3
"""
Multi-Model Machine Learning Analysis for Type 2 Diabetes Remission Prediction

This script implements five machine learning algorithms for predicting
type 2 diabetes remission: Random Forest, XGBoost, L1-regularized Logistic
Regression, Elastic Net, and Ensemble Voting classifier.

Key methodologies:
- Feature extraction: Baseline (BL) and change (Δ) values
- Feature importance ensemble: Simple averaging (normalized to [0,1] interval)
- K-means clustering: Elbow method for optimal cluster number determination
- SHAP analysis: TreeExplainer for model interpretability
- Cross-validation: 5-fold stratified cross-validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set paths relative to script location
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures" / "publication_figures"
ANALYSIS_DIR = BASE_DIR / "analysis"
OUTPUT_DIR = SCRIPT_DIR.parent / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("Machine Learning Analysis for Type 2 Diabetes Remission Prediction")
print("="*80)

# Load and preprocess data
dream_seed = pd.read_csv(DATA_DIR / "dream_seed.csv")

baseline = dream_seed[dream_seed['AVISIT'] == '基线'].copy()
week52 = dream_seed[dream_seed['AVISIT'] == '访视15（第52周）'].copy()
baseline = baseline.rename(columns=lambda x: f"{x}_baseline" if x != 'USUBJID' else x)
week52 = week52.rename(columns=lambda x: f"{x}_week52" if x != 'USUBJID' else x)
merged = baseline.merge(week52, on='USUBJID', how='inner')

# Define features to exclude (data leakage prevention)
EXCLUDE_FEATURES = ['GLUC_FAS', 'GLUC_30M', 'GLUC_2H', 'HBA1C', 'HBA1Cm',
                    'USUBJID', 'seed_num', 'study', 'AVISIT', 'AVISITN',
                    'remission_52w', 'hba1c_52w', 'source_file', 'DELC120', 'DELC0']

# Extract available metabolic indicators
available_cols = [col for col in dream_seed.columns if col not in EXCLUDE_FEATURES]
print(f"\n[Data Info] Available metabolic indicators: {len(available_cols)}")

# Feature categorization based on biological significance
DYNAMIC_BETA = ['CPEP_30M', 'INSU_30M', 'DELTA30', 'DI']
STATIC_BETA = ['HOMA_B', 'HOMA2_B', 'CPEP_FAS', 'INSU_FAS', 'I0_G0']
OTHERS = ['HOMA_IR', 'HOMA2_IR']

all_features = [f for f in available_cols if f in DYNAMIC_BETA + STATIC_BETA + OTHERS]
beta_features = [f for f in all_features if f in DYNAMIC_BETA + STATIC_BETA]

print(f"[Data Info] Total metabolic indicators: {len(all_features)}")
print(f"  - Beta-cell function: {len(beta_features)}")
print(f"  - Insulin sensitivity: {len([f for f in all_features if f in OTHERS])}")

# Calculate change values
for ind in all_features:
    if f'{ind}_baseline' in merged.columns and f'{ind}_week52' in merged.columns:
        merged[f'{ind}_change'] = merged[f'{ind}_week52'] - merged[f'{ind}_baseline']

# Define outcome variable
if 'remission_52w_week52' in merged.columns:
    merged['remission'] = (merged['remission_52w_week52'] == 'Y').astype(int)
else:
    merged['remission'] = (merged['HBA1C_week52'] < 6.5).astype(int)

# Construct feature matrix
feature_cols = []
feature_labels = []
feature_categories = []

for ind in all_features:
    category = "beta_cell" if ind in beta_features else "others"

    if f'{ind}_baseline' in merged.columns:
        feature_cols.append(f'{ind}_baseline')
        feature_labels.append(f'{ind} (BL)')
        feature_categories.append(category)

    if f'{ind}_change' in merged.columns:
        feature_cols.append(f'{ind}_change')
        feature_labels.append(f'{ind} (Δ)')
        feature_categories.append(category)

X = merged[feature_cols].copy()
y = merged['remission'].copy()

# Handle missing values
mask = X.notna().all(axis=1)
X_clean = X[mask]
y_clean = y[mask]

print(f"\n[Data Summary]")
print(f"  Total features: {len(feature_cols)}")
print(f"  Complete samples: {len(X_clean)}/{len(X)}")
print(f"  Remission rate: {y_clean.mean()*100:.1f}%")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Train machine learning models
print("\n[1/5] Training Random Forest...")
rf_param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [6, 8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}
rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_grid = GridSearchCV(rf_base, rf_param_grid,
                       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                       scoring='roc_auc', n_jobs=-1, verbose=0)
rf_grid.fit(X_clean, y_clean)
rf_best = rf_grid.best_estimator_
rf_importances = rf_best.feature_importances_
print(f"  Best CV AUC: {rf_grid.best_score_:.4f}")

print("\n[2/5] Training XGBoost...")
xgb_param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}
xgb_base = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
xgb_grid = GridSearchCV(xgb_base, xgb_param_grid,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                        scoring='roc_auc', n_jobs=-1, verbose=0)
xgb_grid.fit(X_scaled, y_clean)
xgb_best = xgb_grid.best_estimator_
xgb_importances = xgb_best.feature_importances_
print(f"  Best CV AUC: {xgb_grid.best_score_:.4f}")

print("\n[3/5] Training Logistic Regression (L1)...")
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1'],
    'solver': ['liblinear'],
    'class_weight': ['balanced', None]
}
lr_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr_grid = GridSearchCV(lr_base, lr_param_grid,
                       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                       scoring='roc_auc', n_jobs=-1, verbose=0)
lr_grid.fit(X_scaled, y_clean)
lr_best = lr_grid.best_estimator_
lr_coefs = np.abs(lr_best.coef_[0])
print(f"  Best CV AUC: {lr_grid.best_score_:.4f}")

print("\n[4/5] Training Elastic Net...")
en_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'penalty': ['elasticnet'],
    'solver': ['saga'],
    'class_weight': ['balanced', None]
}
en_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
en_grid = GridSearchCV(en_base, en_param_grid,
                       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                       scoring='roc_auc', n_jobs=-1, verbose=0)
en_grid.fit(X_scaled, y_clean)
en_best = en_grid.best_estimator_
en_coefs = np.abs(en_best.coef_[0])
print(f"  Best CV AUC: {en_grid.best_score_:.4f}")

print("\n[5/5] Training Ensemble Voting...")
voting_clf = VotingClassifier(
    estimators=[('rf', rf_best), ('xgb', xgb_best), ('lr', lr_best), ('en', en_best)],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_scaled, y_clean)
voting_pred_proba = voting_clf.predict_proba(X_scaled)[:, 1]
voting_auc = roc_auc_score(y_clean, voting_pred_proba)
print(f"  Ensemble AUC: {voting_auc:.4f}")

# Compute ensemble feature importance
print("\n[Feature Importance] Computing ensemble importance (simple averaging)...")

def normalize_importance(imp):
    return (imp - imp.min()) / (imp.max() - imp.min())

rf_norm = normalize_importance(rf_importances)
xgb_norm = normalize_importance(xgb_importances)
lr_norm = normalize_importance(lr_coefs)
en_norm = normalize_importance(en_coefs)

ensemble_df = pd.DataFrame({
    'feature': feature_cols,
    'label': feature_labels,
    'category': feature_categories,
    'RF_importance': rf_norm,
    'XGB_importance': xgb_norm,
    'LR_importance': lr_norm,
    'EN_importance': en_norm
})

# Simple averaging (as per manuscript methods)
ensemble_df['ensemble_score'] = ensemble_df[[
    'RF_importance',
    'XGB_importance',
    'LR_importance',
    'EN_importance'
]].mean(axis=1)

ensemble_df = ensemble_df.sort_values('ensemble_score', ascending=False)

print(f"  Top 10 features:")
for idx, row in ensemble_df.head(10).iterrows():
    print(f"    {row['label']:25s} Score={row['ensemble_score']:.4f}")

# K-means clustering analysis
print("\n[K-means Clustering] Performing unsupervised patient phenotyping...")

change_features = [f'{ind}_change' for ind in all_features if f'{ind}_change' in merged.columns]
print(f"  Change features for clustering: {len(change_features)}")

X_cluster = merged.loc[X_clean.index, change_features].dropna()
y_cluster = y_clean.loc[X_cluster.index]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Elbow method for optimal cluster number
print("  Applying Elbow method...")
inertias = []
k_range = range(2, min(11, len(X_cluster)//3))
for n_clusters in k_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)

if len(inertias) >= 3:
    second_derivatives = []
    for i in range(1, len(inertias)-1):
        second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
        second_derivatives.append(second_deriv)
    best_n_clusters = list(k_range)[second_derivatives.index(max(second_derivatives)) + 1]
    print(f"  Optimal clusters: {best_n_clusters}")
else:
    best_n_clusters = 3
    print(f"  Using default: {best_n_clusters} clusters")

kmeans_final = KMeans(n_clusters=best_n_clusters, random_state=RANDOM_STATE, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)

cluster_stats = []
for cluster_id in range(best_n_clusters):
    cluster_mask = cluster_labels == cluster_id
    cluster_size = cluster_mask.sum()
    cluster_remission = y_cluster[cluster_mask].sum()
    cluster_remission_rate = cluster_remission / cluster_size if cluster_size > 0 else 0
    cluster_stats.append({
        'Cluster': cluster_id,
        'Size': cluster_size,
        'Remission_Rate': cluster_remission_rate
    })
    print(f"  Cluster {cluster_id}: n={cluster_size}, Remission={cluster_remission_rate*100:.1f}%")

cluster_stats_df = pd.DataFrame(cluster_stats)

# SHAP analysis
print("\n[SHAP Analysis] Computing SHAP values...")
explainer = shap.TreeExplainer(rf_best)
shap_values = explainer.shap_values(X_clean)

if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

print(f"  SHAP values computed for {len(X_clean)} samples")

# Save results
print("\n[Saving Results]")
ensemble_df.to_csv(OUTPUT_DIR / "Ensemble_Feature_Importance.csv", index=False)
cluster_stats_df.to_csv(OUTPUT_DIR / "kmeans_cluster_stats.csv", index=False)

performance_summary = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Logistic Regression (L1)', 'Elastic Net', 'Ensemble Voting'],
    'CV_AUC': [rf_grid.best_score_, xgb_grid.best_score_, lr_grid.best_score_, en_grid.best_score_, voting_auc]
})
performance_summary.to_csv(OUTPUT_DIR / "Model_Performance.csv", index=False)

print(f"  Saved: Ensemble_Feature_Importance.csv")
print(f"  Saved: kmeans_cluster_stats.csv")
print(f"  Saved: Model_Performance.csv")

# Generate publication figures
print("\n[Visualization] Generating figures...")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

top_n_heatmap = min(15, len(ensemble_df))
top_n_barplot = min(15, len(ensemble_df))

# Figure 1: Feature importance heatmap
fig, ax = plt.subplots(figsize=(10, 8))
top_features = ensemble_df.head(top_n_heatmap)
heatmap_data = top_features[['RF_importance', 'XGB_importance', 'LR_importance', 'EN_importance']].T
heatmap_data.columns = top_features['label']
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False,
            cbar_kws={'label': 'Normalized Importance'},
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_xlabel('Metabolic Parameters', fontsize=12, fontweight='bold')
ax.set_ylabel('Machine Learning Methods', fontsize=12, fontweight='bold')
ax.set_title(f'Feature Importance Heatmap (Top {top_n_heatmap})', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ML_5Methods_Importance_Heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "ML_5Methods_Importance_Heatmap.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: ML_5Methods_Importance_Heatmap.png/pdf")

# Figure 2: Ensemble importance bar plot
fig, ax = plt.subplots(figsize=(10, 8))
top_features = ensemble_df.head(top_n_barplot)
colors = ['#FF8C00' if cat == 'beta_cell' else '#808080' for cat in top_features['category']]
ax.barh(range(len(top_features)), top_features['ensemble_score'],
        color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['label'])
ax.invert_yaxis()
ax.set_xlabel('Ensemble Importance (Simple Average)', fontsize=12, fontweight='bold')
ax.set_ylabel('Metabolic Parameters', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_n_barplot} Features', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF8C00', edgecolor='black', label='Beta-cell function'),
    Patch(facecolor='#808080', edgecolor='black', label='Others')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"ML_Ensemble_Importance_Top{top_n_barplot}.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / f"ML_Ensemble_Importance_Top{top_n_barplot}.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: ML_Ensemble_Importance_Top{top_n_barplot}.png/pdf")

# Figure 3: Random Forest ROC curve
fig, ax = plt.subplots(figsize=(8, 8))
rf_pred_proba = rf_best.predict_proba(X_clean)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_clean, rf_pred_proba)
rf_auc = roc_auc_score(y_clean, rf_pred_proba)
ax.plot(rf_fpr, rf_tpr, color='#1f77b4', lw=3, label=f'Random Forest (AUC = {rf_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.500)')
ax.fill_between(rf_fpr, rf_tpr, alpha=0.2, color='#1f77b4')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Random Forest ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(FIGURES_DIR / "RF_ROC_Curve.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "RF_ROC_Curve.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: RF_ROC_Curve.png/pdf")

# Figure 4: Multi-model ROC comparison
fig, ax = plt.subplots(figsize=(8, 8))
models = {
    'Random Forest': rf_best,
    'XGBoost': xgb_best,
    'Logistic Regression': lr_best,
    'Elastic Net': en_best,
    'Ensemble Voting': voting_clf
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for idx, (name, model) in enumerate(models.items()):
    if name == 'Ensemble Voting':
        y_pred_proba = voting_pred_proba
    else:
        if name == 'Random Forest':
            y_pred_proba = model.predict_proba(X_clean)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_clean, y_pred_proba)
    auc = roc_auc_score(y_clean, y_pred_proba)
    ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ML_5Methods_ROC_Comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "ML_5Methods_ROC_Comparison.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: ML_5Methods_ROC_Comparison.png/pdf")

print("\n" + "="*80)
print("Analysis completed successfully!")
print("="*80)
print(f"\nSummary:")
print(f"  - Metabolic indicators: {len(all_features)}")
print(f"  - Total features: {len(feature_cols)}")
print(f"  - Clusters: {best_n_clusters}")
print(f"  - Complete samples: {len(X_clean)}")
print(f"  - Remission rate: {y_clean.mean()*100:.1f}%")
print("="*80)
