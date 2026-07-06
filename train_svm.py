"""
train_svm.py - Train SVM for flood prediction
Model 3 of 6 - Full training visibility with kernel comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🟡 MODEL 3 OF 6: SUPPORT VECTOR MACHINE (SVM)")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n📂 Loading dataset...")
dataset = pd.read_csv('processed_data/flood_dataset_large.csv')
print(f"   ✅ Loaded {len(dataset)} samples")

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
# SCALE FEATURES (CRITICAL FOR SVM)
# ============================================================

print("\n📊 Scaling features (SVM is sensitive to scale)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaling complete")

# ============================================================
# TRAIN SVM WITH DIFFERENT KERNELS
# ============================================================

print("\n" + "-" * 50)
print("🤖 Training SVM with different kernels...")
print("-" * 50)

kernels = ['rbf', 'linear', 'poly']
svm_results = []

for kernel in kernels:
    print(f"\n   🔄 Testing {kernel} kernel...")
    
    svm_model = SVC(
        kernel=kernel,
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    
    svm_results.append({
        'kernel': kernel,
        'model': svm_model,
        'accuracy': accuracy,
        'f1': f1,
        'y_pred': y_pred,
        'y_proba': y_proba
    })

# Select best kernel
best_result = max(svm_results, key=lambda x: x['f1'])
print(f"\n   🏆 Best kernel: {best_result['kernel']} (F1={best_result['f1']:.4f})")

# Use best model
svm_model = best_result['model']
y_pred = best_result['y_pred']
y_proba = best_result['y_proba']

# ============================================================
# EVALUATE
# ============================================================

print("\n📊 Evaluating on test set...")

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
print(f"   ✅ Best Kernel: {best_result['kernel']}")

# ============================================================
# CROSS-VALIDATION
# ============================================================

print("\n📊 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='f1')
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/svm_confusion.png', dpi=300)
print("   ✅ Saved confusion matrix: figures/svm_confusion.png")

# ============================================================
# FEATURE IMPORTANCE (using coefficient for linear kernel)
# ============================================================

print("\n📊 Feature Analysis:")

if best_result['kernel'] == 'linear':
    print("   (Using linear kernel coefficients for importance)")
    importance = np.abs(svm_model.coef_[0])
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='#F18F01')
    plt.title('SVM Feature Importance (Linear Kernel)')
    plt.xlabel('|Coefficient|')
    plt.tight_layout()
    plt.savefig('figures/svm_feature_importance.png', dpi=300)
    print("   ✅ Saved feature importance: figures/svm_feature_importance.png")
else:
    print("   ⚠️ Feature importance not available for non-linear kernel")

# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'SVM ({best_result["kernel"]}) (AUC = {auc_roc:.3f})', color='#F18F01')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend()
plt.tight_layout()
plt.savefig('figures/svm_roc.png', dpi=300)
print("   ✅ Saved ROC curve: figures/svm_roc.png")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)

with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("   ✅ Saved: models/svm_model.pkl")

with open('models/svm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved: models/svm_scaler.pkl")

with open('models/svm_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✅ Saved: models/svm_features.pkl")

# Save kernel info
with open('models/svm_kernel.txt', 'w') as f:
    f.write(f"Best Kernel: {best_result['kernel']}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(f"✅ SVM TRAINING COMPLETE!")
print("=" * 60)
print(f"   Best Kernel: {best_result['kernel']}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   AUC-ROC: {auc_roc:.4f}")
print("\n📁 Figures saved:")
print("   - figures/svm_confusion.png")
print("   - figures/svm_roc.png")
if best_result['kernel'] == 'linear':
    print("   - figures/svm_feature_importance.png")
print("\n🟡 Model 3 of 6 COMPLETE!")
print("   Run train_lstm.py next!")