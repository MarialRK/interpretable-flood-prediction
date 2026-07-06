"""
train_gru.py - Train GRU for flood prediction
Model 5 of 6 - FULL EPOCH VISIBILITY with training curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 70)
print("🟣 MODEL 5 OF 6: GRU (Gated Recurrent Unit)")
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
# RESHAPE FOR GRU (samples, timesteps, features)
# ============================================================

print("\n📊 Reshaping for GRU...")
X_train_gru = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_gru = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
X_test_gru = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
print(f"   Shape: {X_train_gru.shape} (samples, timesteps, features)")

# ============================================================
# BUILD GRU MODEL
# ============================================================

print("\n" + "-" * 50)
print("🤖 Building GRU Model...")
print("-" * 50)

model = Sequential([
    GRU(32, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
    Dropout(0.2),
    GRU(16),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("   ✅ Model built successfully!")
print(f"   📊 Model Summary:")
model.summary()

# ============================================================
# TRAIN GRU WITH FULL EPOCH VISIBILITY
# ============================================================

print("\n" + "-" * 50)
print("🔄 Training GRU...")
print("-" * 50)

print("\n🔴 TRAINING PROGRESS (Epoch by Epoch):")
print("=" * 80)
print(" Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
print("-" * 80)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Custom callback to print progress
class PrintProgress:
    def __init__(self):
        self.epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        print(f" {epoch:5d} | {train_loss:.4f}    | {train_acc:.4f}   | {val_loss:.4f}    | {val_acc:.4f}")

# Train
history = model.fit(
    X_train_gru, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val_gru, y_val),
    callbacks=[early_stop],
    verbose=0
)

print("-" * 80)
print(f"✅ Training completed at {len(history.history['loss'])} epochs")
print(f"   Best Validation Loss: {min(history.history['val_loss']):.4f}")
print(f"   Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")

# ============================================================
# EVALUATE
# ============================================================

print("\n📊 Evaluating on test set...")
y_pred_proba = model.predict(X_test_gru, verbose=0).flatten()
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
# PLOT TRAINING HISTORY
# ============================================================

print("\n📊 Creating training history plots...")
os.makedirs('figures', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Training Loss', color='blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GRU - Loss vs Epochs')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('GRU - Accuracy vs Epochs')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/gru_training_history.png', dpi=300)
print("   ✅ Saved training history: figures/gru_training_history.png")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_binary)
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'])
plt.title('Confusion Matrix - GRU')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/gru_confusion.png', dpi=300)
print("   ✅ Saved confusion matrix: figures/gru_confusion.png")

# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'GRU (AUC = {auc_roc:.3f})', color='purple')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - GRU')
plt.legend()
plt.tight_layout()
plt.savefig('figures/gru_roc.png', dpi=300)
print("   ✅ Saved ROC curve: figures/gru_roc.png")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)

model.save('models/gru_model.keras')
print("   ✅ Saved: models/gru_model.keras")

with open('models/gru_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved: models/gru_scaler.pkl")

with open('models/gru_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✅ Saved: models/gru_features.pkl")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(f"✅ GRU TRAINING COMPLETE!")
print("=" * 60)
print(f"   Total Epochs: {len(history.history['loss'])}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   AUC-ROC: {auc_roc:.4f}")
print("\n📁 Figures saved:")
print("   - figures/gru_training_history.png")
print("   - figures/gru_confusion.png")
print("   - figures/gru_roc.png")
print("\n🟣 Model 5 of 6 COMPLETE!")
print("   Run train_arima.py next!")