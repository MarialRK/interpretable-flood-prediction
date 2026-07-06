"""
train_randomforest.py - Train Random Forest for flood prediction
Model 2 of 6 - Detailed training and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🟢 MODEL 2 OF 6: RANDOM FOREST")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n📂 Loading dataset...")
try:
    dataset = pd.read_csv('processed_data/flood_dataset_large.csv')
    print(f"   ✅ Loaded {len(dataset)} samples")
except:
    print("   ❌ Dataset not found! Running generate_data.py first...")
    print("   Please run: python generate_data.py")
    exit()

X = dataset[['month', 'rainfall_mm', 'rainfall_anomaly', 'ndvi', 'elevation_m', 'slope_pct', 'wet_season']]
y = dataset['flood']

print(f"\n📊 Features: {list(X.columns)}")
print(f"🎯 Target: flood (1={y.sum()}, 0={len(y)-y.sum()})")

# ============================================================
# SPLIT DATA
# ============================================================

print("\n📊 Splitting data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples ({y_train.sum()} floods)")
print(f"   Testing: {len(X_test)} samples ({y_test.sum()} floods)")

# ============================================================
# SCALE FEATURES
# ============================================================

print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaling complete")

# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n" + "-" * 50)
print("🤖 Training Random Forest...")
print("-" * 50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("   🔄 Training with 100 trees...")
rf_model.fit(X_train_scaled, y_train)
print("   ✅ Training complete")

# ============================================================
# EVALUATE
# ============================================================

print("\n📊 Evaluating on test set...")
y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)

print(f"\n📈 RESULTS:")
print(f"   ✅ Accuracy:  {accuracy:.4f}")
print(f"   ✅ Precision: {precision:.4f}")
print(f"   ✅ Recall:    {recall:.4f}")
print(f"   ✅ F1-Score:  {f1:.4f}")
print(f"   ✅ AUC-ROC:   {auc_roc:.4f}")

# ============================================================
# CROSS-VALIDATION
# ============================================================

print("\n📊 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"   CV F1-Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/randomforest_confusion.png', dpi=300)
print("   ✅ Saved confusion matrix: figures/randomforest_confusion.png")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n📊 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='#2CA02C')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/randomforest_feature_importance.png', dpi=300)
print("   ✅ Saved feature importance: figures/randomforest_feature_importance.png")

# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_roc:.3f})', color='#2CA02C')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig('figures/randomforest_roc.png', dpi=300)
print("   ✅ Saved ROC curve: figures/randomforest_roc.png")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)

with open('models/randomforest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   ✅ Saved: models/randomforest_model.pkl")

with open('models/randomforest_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved: models/randomforest_scaler.pkl")

with open('models/randomforest_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✅ Saved: models/randomforest_features.pkl")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(f"✅ RANDOM FOREST TRAINING COMPLETE!")
print("=" * 60)
print(f"   F1-Score: {f1:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   AUC-ROC: {auc_roc:.4f}")
print("\n📁 Figures saved:")
print("   - figures/randomforest_confusion.png")
print("   - figures/randomforest_feature_importance.png")
print("   - figures/randomforest_roc.png")
print("\n🟢 Model 2 of 6 COMPLETE!")
print("   Run train_svm.py next!")