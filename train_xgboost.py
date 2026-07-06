"""
train_xgboost.py - Train XGBoost with FULL training visibility
Model 1 of 6 - Shows every iteration, loss, and validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔵 MODEL 1 OF 6: XGBOOST - FULL TRAINING VISIBILITY")
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

print("\n📊 Splitting data (70% train, 15% validation, 15% test)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"   Training: {len(X_train)} samples ({y_train.sum()} floods)")
print(f"   Validation: {len(X_val)} samples ({y_val.sum()} floods)")
print(f"   Testing: {len(X_test)} samples ({y_test.sum()} floods)")

# ============================================================
# SCALE FEATURES
# ============================================================

print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaling complete")

# ============================================================
# TRAIN XGBOOST WITH EARLY STOPPING
# ============================================================

print("\n" + "-" * 50)
print("🤖 Training XGBoost...")
print("-" * 50)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled, label=y_val)
dtest = xgb.DMatrix(X_test_scaled)

# Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'error'],
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

evals = [(dtrain, 'train'), (dval, 'val')]

# Store training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'iterations': []
}

print("\n🔴 TRAINING PROGRESS:")
print("=" * 70)
print(" Iter | Train Loss | Val Loss | Train Acc | Val Acc")
print("-" * 70)

# Custom callback class
class CustomCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        step = epoch
        if step % 5 == 0:
            train_loss = evals_log['train']['logloss'][-1]
            val_loss = evals_log['val']['logloss'][-1]
            train_err = evals_log['train']['error'][-1]
            val_err = evals_log['val']['error'][-1]
            train_acc = 1 - train_err
            val_acc = 1 - val_err
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['iterations'].append(step)
            
            print(f" {step:4d} | {train_loss:.4f}   | {val_loss:.4f}  | {train_acc:.4f}   | {val_acc:.4f}")
        
        return False

# Train with early stopping
print(" Starting training... (early stopping with patience=10)")
print()

model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=10,
    callbacks=[CustomCallback()],
    verbose_eval=False
)

print("-" * 70)
print(f"✅ Training completed at {model.best_iteration} iterations")
print(f"   Best Validation Loss: {model.best_score:.4f}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================

print("\n📊 Evaluating on test set...")
y_pred_proba = model.predict(dtest)
y_pred_binary = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 FINAL RESULTS:")
print(f"   ✅ Accuracy:  {accuracy:.4f}")
print(f"   ✅ Precision: {precision:.4f}")
print(f"   ✅ Recall:    {recall:.4f}")
print(f"   ✅ F1-Score:  {f1:.4f}")
print(f"   ✅ AUC-ROC:   {auc_roc:.4f}")

# ============================================================
# CROSS-VALIDATION
# ============================================================

print("\n📊 Cross-Validation (5-fold)...")
try:
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=100,
        nfold=5,
        early_stopping_rounds=10,
        metrics=['logloss', 'error'],
        seed=42
    )
    print(f"   CV LogLoss: {cv_results['test-logloss-mean'].iloc[-1]:.4f}")
    print(f"   CV Error: {cv_results['test-error-mean'].iloc[-1]:.4f}")
    print(f"   CV Accuracy: {1 - cv_results['test-error-mean'].iloc[-1]:.4f}")
except Exception as e:
    print(f"   ⚠️ CV skipped: {e}")

# ============================================================
# PLOT LEARNING CURVES
# ============================================================

print("\n📊 Creating learning curve plots...")
os.makedirs('figures', exist_ok=True)

if len(history['train_loss']) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history['iterations'], history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
    ax1.plot(history['iterations'], history['val_loss'], label='Validation Loss', color='red', marker='s', markersize=4)
    ax1.axvline(x=model.best_iteration, color='green', linestyle='--', label=f'Best Model (iter={model.best_iteration})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('XGBoost - Loss vs Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['iterations'], history['train_acc'], label='Train Accuracy', color='blue', marker='o', markersize=4)
    ax2.plot(history['iterations'], history['val_acc'], label='Validation Accuracy', color='red', marker='s', markersize=4)
    ax2.axvline(x=model.best_iteration, color='green', linestyle='--', label=f'Best Model (iter={model.best_iteration})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('XGBoost - Accuracy vs Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/xgboost_learning_curves.png', dpi=300)
    print("   ✅ Saved learning curves: figures/xgboost_learning_curves.png")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_binary)
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'])
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/xgboost_confusion.png', dpi=300)
print("   ✅ Saved confusion matrix: figures/xgboost_confusion.png")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n📊 Feature Importance:")
importance = model.get_score(importance_type='weight')
if importance:
    feature_importance = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='#2E86AB')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance (Weight)')
    plt.tight_layout()
    plt.savefig('figures/xgboost_feature_importance.png', dpi=300)
    print("   ✅ Saved feature importance: figures/xgboost_feature_importance.png")
else:
    print("   ⚠️ No feature importance available")

# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_roc:.3f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.tight_layout()
plt.savefig('figures/xgboost_roc.png', dpi=300)
print("   ✅ Saved ROC curve: figures/xgboost_roc.png")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)

model.save_model('models/xgboost_model.json')
print("   ✅ Saved: models/xgboost_model.json")

with open('models/xgboost_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved: models/xgboost_scaler.pkl")

with open('models/xgboost_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✅ Saved: models/xgboost_features.pkl")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(f"✅ XGBoost TRAINING COMPLETE!")
print("=" * 60)
print(f"   Best Iteration: {model.best_iteration}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   AUC-ROC: {auc_roc:.4f}")
print("\n📁 Figures saved:")
print("   - figures/xgboost_learning_curves.png")
print("   - figures/xgboost_confusion.png")
print("   - figures/xgboost_feature_importance.png")
print("   - figures/xgboost_roc.png")
print("\n🔴 Model 1 of 6 COMPLETE!")