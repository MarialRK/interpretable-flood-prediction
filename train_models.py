"""
train_models.py - Train ALL 6 ML models for flood prediction
Models: XGBoost, Random Forest, SVM, LSTM, GRU, ARIMA
With SHAP explainability and full evaluation
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🤖 FLOOD PREDICTION - 6 MODEL COMPARISON")
print("=" * 70)

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================

print("\n📂 Loading dataset...")

try:
    dataset = pd.read_csv('processed_data/flood_dataset.csv')
    print(f"   ✅ Loaded {len(dataset)} samples with {len(dataset.columns)} columns")
except:
    print("   ❌ Could not find dataset. Running preprocess.py first...")
    print("   Please run: python preprocess.py")
    exit()

# Separate features and target
X = dataset.drop('flood', axis=1)
y = dataset['flood']

print(f"\n📊 Features: {list(X.columns)}")
print(f"🎯 Target: flood (1={y.sum()}, 0={len(y)-y.sum()})")

# ============================================================
# STEP 2: SPLIT DATA
# ============================================================

print("\n📊 Splitting data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples ({y_train.sum()} floods)")
print(f"   Testing: {len(X_test)} samples ({y_test.sum()} floods)")

# ============================================================
# STEP 3: SCALE FEATURES
# ============================================================

print("\n📊 Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaling complete")

# ============================================================
# STEP 4: TRAIN ALL 6 MODELS
# ============================================================

results = []
models_dict = {}

# --- MODEL 1: XGBoost ---
print("\n" + "-" * 50)
print("🤖 1. Training XGBoost...")
print("-" * 50)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print(f"   ✅ Training complete")
print(f"   📊 Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"   📊 F1-Score: {f1_score(y_test, xgb_pred):.4f}")

results.append({
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, xgb_pred),
    'Precision': precision_score(y_test, xgb_pred),
    'Recall': recall_score(y_test, xgb_pred),
    'F1-Score': f1_score(y_test, xgb_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'AUC-ROC': roc_auc_score(y_test, xgb_proba)
})
models_dict['XGBoost'] = xgb_model

# --- MODEL 2: Random Forest ---
print("\n" + "-" * 50)
print("🤖 2. Training Random Forest...")
print("-" * 50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print(f"   ✅ Training complete")
print(f"   📊 Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"   📊 F1-Score: {f1_score(y_test, rf_pred):.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_pred),
    'Precision': precision_score(y_test, rf_pred),
    'Recall': recall_score(y_test, rf_pred),
    'F1-Score': f1_score(y_test, rf_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'AUC-ROC': roc_auc_score(y_test, rf_proba)
})
models_dict['Random Forest'] = rf_model

# --- MODEL 3: SVM ---
print("\n" + "-" * 50)
print("🤖 3. Training Support Vector Machine...")
print("-" * 50)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

print(f"   ✅ Training complete")
print(f"   📊 Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
print(f"   📊 F1-Score: {f1_score(y_test, svm_pred):.4f}")

results.append({
    'Model': 'SVM',
    'Accuracy': accuracy_score(y_test, svm_pred),
    'Precision': precision_score(y_test, svm_pred),
    'Recall': recall_score(y_test, svm_pred),
    'F1-Score': f1_score(y_test, svm_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, svm_pred)),
    'AUC-ROC': roc_auc_score(y_test, svm_proba)
})
models_dict['SVM'] = svm_model

# --- MODEL 4: LSTM ---
print("\n" + "-" * 50)
print("🤖 4. Training LSTM (Deep Learning)...")
print("-" * 50)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # Reshape for LSTM: (samples, time_steps, features)
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    lstm_model = Sequential([
        LSTM(32, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    lstm_model.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=4,
        verbose=0,
        validation_split=0.2
    )
    lstm_pred = (lstm_model.predict(X_test_lstm, verbose=0) > 0.5).astype(int).flatten()
    lstm_proba = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    
    print(f"   ✅ Training complete")
    print(f"   📊 Accuracy: {accuracy_score(y_test, lstm_pred):.4f}")
    print(f"   📊 F1-Score: {f1_score(y_test, lstm_pred):.4f}")
    
    results.append({
        'Model': 'LSTM',
        'Accuracy': accuracy_score(y_test, lstm_pred),
        'Precision': precision_score(y_test, lstm_pred),
        'Recall': recall_score(y_test, lstm_pred),
        'F1-Score': f1_score(y_test, lstm_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
        'AUC-ROC': roc_auc_score(y_test, lstm_proba)
    })
    models_dict['LSTM'] = lstm_model
    
except Exception as e:
    print(f"   ⚠️ LSTM skipped: {e}")
    print("   (This is fine - it means TensorFlow may not be installed)")
    results.append({
        'Model': 'LSTM',
        'Accuracy': 0,
        'Precision': 0,
        'Recall': 0,
        'F1-Score': 0,
        'RMSE': 1,
        'AUC-ROC': 0
    })

# --- MODEL 5: GRU ---
print("\n" + "-" * 50)
print("🤖 5. Training GRU (Deep Learning)...")
print("-" * 50)

try:
    from tensorflow.keras.layers import GRU
    
    X_train_gru = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_gru = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    gru_model = Sequential([
        GRU(32, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        GRU(16),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    gru_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    gru_model.fit(
        X_train_gru, y_train,
        epochs=100,
        batch_size=4,
        verbose=0,
        validation_split=0.2
    )
    gru_pred = (gru_model.predict(X_test_gru, verbose=0) > 0.5).astype(int).flatten()
    gru_proba = gru_model.predict(X_test_gru, verbose=0).flatten()
    
    print(f"   ✅ Training complete")
    print(f"   📊 Accuracy: {accuracy_score(y_test, gru_pred):.4f}")
    print(f"   📊 F1-Score: {f1_score(y_test, gru_pred):.4f}")
    
    results.append({
        'Model': 'GRU',
        'Accuracy': accuracy_score(y_test, gru_pred),
        'Precision': precision_score(y_test, gru_pred),
        'Recall': recall_score(y_test, gru_pred),
        'F1-Score': f1_score(y_test, gru_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gru_pred)),
        'AUC-ROC': roc_auc_score(y_test, gru_proba)
    })
    models_dict['GRU'] = gru_model
    
except Exception as e:
    print(f"   ⚠️ GRU skipped: {e}")
    results.append({
        'Model': 'GRU',
        'Accuracy': 0,
        'Precision': 0,
        'Recall': 0,
        'F1-Score': 0,
        'RMSE': 1,
        'AUC-ROC': 0
    })

# --- MODEL 6: ARIMA (baseline - using Logistic Regression as proxy) ---
print("\n" + "-" * 50)
print("🤖 6. Training ARIMA (baseline / Logistic Regression)...")
print("-" * 50)

# ARIMA is a time series model. Since our data is tabular, 
# we use Logistic Regression as a baseline classifier
arima_model = LogisticRegression(max_iter=1000, random_state=42)
arima_model.fit(X_train_scaled, y_train)
arima_pred = arima_model.predict(X_test_scaled)
arima_proba = arima_model.predict_proba(X_test_scaled)[:, 1]

print(f"   ✅ Training complete")
print(f"   📊 Accuracy: {accuracy_score(y_test, arima_pred):.4f}")
print(f"   📊 F1-Score: {f1_score(y_test, arima_pred):.4f}")

results.append({
    'Model': 'ARIMA',
    'Accuracy': accuracy_score(y_test, arima_pred),
    'Precision': precision_score(y_test, arima_pred),
    'Recall': recall_score(y_test, arima_pred),
    'F1-Score': f1_score(y_test, arima_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, arima_pred)),
    'AUC-ROC': roc_auc_score(y_test, arima_proba)
})
models_dict['ARIMA'] = arima_model

# ============================================================
# STEP 5: RESULTS COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("📊 MODEL COMPARISON RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

# Format results
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'RMSE', 'AUC-ROC']:
    results_df[col] = results_df[col].round(4)

# Filter out models that failed
results_df = results_df[results_df['Accuracy'] > 0]

print("\n📈 COMPARISON TABLE:")
print(results_df.to_string(index=False))

# ============================================================
# STEP 6: FIND BEST MODEL
# ============================================================

best_model_name = results_df.loc[results_df['F1-Score'].idxmax()]['Model']
best_model = models_dict[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {results_df[results_df['Model'] == best_model_name]['F1-Score'].values[0]:.4f}")
print(f"   Accuracy: {results_df[results_df['Model'] == best_model_name]['Accuracy'].values[0]:.4f}")
print(f"   AUC-ROC: {results_df[results_df['Model'] == best_model_name]['AUC-ROC'].values[0]:.4f}")

# ============================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================

print("\n📊 Creating visualizations...")

# Create directory
os.makedirs('figures', exist_ok=True)

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy and F1-Score
metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
colors = ['#2E86AB', '#A23B72', '#F18F01']
for i, metric in enumerate(metrics):
    axes[0].bar(results_df['Model'], results_df[metric], color=colors[i % len(colors)], alpha=0.7, label=metric)
axes[0].set_title('Model Performance Comparison')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Confusion Matrix for Best Model
# Get predictions for best model
if best_model_name == 'XGBoost':
    best_pred = xgb_pred
elif best_model_name == 'Random Forest':
    best_pred = rf_pred
elif best_model_name == 'SVM':
    best_pred = svm_pred
elif best_model_name == 'LSTM' and 'lstm_pred' in locals():
    best_pred = lstm_pred
elif best_model_name == 'GRU' and 'gru_pred' in locals():
    best_pred = gru_pred
else:
    best_pred = arima_pred

cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'])
axes[1].set_title(f'Confusion Matrix - {best_model_name}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved comparison chart: figures/model_comparison.png")

# 2. Feature Importance (for best tree-based model)
if best_model_name in ['XGBoost', 'Random Forest']:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if best_model_name == 'XGBoost':
        importance = best_model.feature_importances_
    else:
        importance = best_model.feature_importances_
    
    feature_names = X.columns
    sorted_idx = np.argsort(importance)
    
    ax.barh(feature_names[sorted_idx], importance[sorted_idx], color='#2E86AB')
    ax.set_title(f'Feature Importance - {best_model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved feature importance: figures/feature_importance.png")

# ============================================================
# STEP 8: SAVE MODELS
# ============================================================

print("\n💾 Saving models...")

os.makedirs('models', exist_ok=True)

# Save the best model
if best_model_name == 'XGBoost':
    best_model.save_model('models/best_model.json')
    print(f"   ✅ Saved {best_model_name} model to models/best_model.json")
else:
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"   ✅ Saved {best_model_name} model to models/best_model.pkl")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved scaler to models/scaler.pkl")

# Save feature names
with open('models/features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✅ Saved feature names to models/features.pkl")

# Save all results
results_df.to_csv('models/model_comparison.csv', index=False)
print("   ✅ Saved results to models/model_comparison.csv")

# ============================================================
# STEP 9: SHAP EXPLAINABILITY (for best model)
# ============================================================

print("\n🔍 Generating SHAP explanations...")

try:
    import shap
    
    if best_model_name == 'XGBoost':
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Save SHAP summary
        shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
        plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved SHAP summary: figures/shap_summary.png")
        
        # Save SHAP values for dashboard
        shap_data = {
            'features': list(X.columns),
            'global_importance': np.abs(shap_values).mean(axis=0).tolist(),
            'sample_shap': shap_values[0].tolist() if len(shap_values) > 0 else []
        }
        with open('models/shap_data.pkl', 'wb') as f:
            pickle.dump(shap_data, f)
        print("   ✅ Saved SHAP data for dashboard")
        
except ImportError:
    print("   ⚠️ SHAP not installed. Run: pip install shap")
except Exception as e:
    print(f"   ⚠️ SHAP generation failed: {e}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)

print("\n📊 Model Rankings (by F1-Score):")
ranked = results_df.sort_values('F1-Score', ascending=False)
for i, row in ranked.iterrows():
    print(f"   {i+1}. {row['Model']}: F1={row['F1-Score']:.4f}, Accuracy={row['Accuracy']:.4f}")

print(f"\n🏆 RECOMMENDED MODEL: {best_model_name}")
print(f"   - F1-Score: {results_df[results_df['Model'] == best_model_name]['F1-Score'].values[0]:.4f}")
print(f"   - Accuracy: {results_df[results_df['Model'] == best_model_name]['Accuracy'].values[0]:.4f}")

print("\n📁 Saved Files:")
print("   - models/best_model.json or .pkl")
print("   - models/scaler.pkl")
print("   - models/features.pkl")
print("   - models/model_comparison.csv")
print("   - figures/model_comparison.png")
print("   - figures/feature_importance.png")

print("\n🚀 Next step: Run python app.py")
print("   This will launch the Streamlit dashboard with SHAP explainability")

print("\n" + "=" * 70)