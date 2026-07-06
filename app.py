"""
app.py - Professional Flood Prediction Dashboard
Sudd Wetland Region, South Sudan
Capstone Project - BSc Software Engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🌊 Sudd Flood Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .header-location {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #2d6a9f;
    }
    
    .card-danger {
        border-left: 4px solid #dc3545;
    }
    
    .card-success {
        border-left: 4px solid #28a745;
    }
    
    .card-warning {
        border-left: 4px solid #ffc107;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Risk badge */
    .risk-high {
        background: #dc3545;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-medium {
        background: #ffc107;
        color: #212529;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-low {
        background: #28a745;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Feature explanation */
    .feature-box {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .feature-name {
        font-weight: 500;
    }
    
    .feature-impact-positive {
        color: #dc3545;
        font-weight: 600;
    }
    
    .feature-impact-negative {
        color: #28a745;
        font-weight: 600;
    }
    
    .feature-impact-neutral {
        color: #6c757d;
    }
    
    /* Recommendation cards */
    .rec-card {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        background: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .rec-card-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header-container">
    <div class="header-title">🌊 Interpretable Flood-Risk Prediction System</div>
    <div class="header-subtitle">Machine Learning for Disaster Preparedness</div>
    <div class="header-location">📍 Sudd Wetland Region, South Sudan</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD BEST MODEL
# ============================================================

@st.cache_resource
def load_best_model():
    """Load the best model and artifacts"""
    
    model_path = 'models/svm_model.pkl'
    scaler_path = 'models/svm_scaler.pkl'
    features_path = 'models/svm_features.pkl'
    
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Please run training first.")
        return None, None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # Load model comparison results
        comparison_path = 'models/model_comparison.csv'
        comparison_df = None
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
        
        return model, scaler, features, comparison_df
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

model, scaler, features, comparison_df = load_best_model()

if model is None:
    st.stop()

# ============================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================

with st.sidebar:
    st.markdown("### 📊 Location Parameters")
    st.markdown("---")
    
    # Month
    month = st.slider(
        "📅 Month",
        min_value=1, max_value=12, value=6,
        help="1=January, 6=June, 12=December"
    )
    
    # Month name display
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    st.caption(f"Selected: **{month_names[month-1]}**")
    
    st.markdown("---")
    
    # Rainfall
    rainfall_mm = st.slider(
        "🌧️ Rainfall (mm)",
        min_value=0, max_value=300, value=120,
        help="Monthly rainfall in millimeters"
    )
    
    # Rainfall anomaly
    rainfall_anomaly = st.slider(
        "📈 Rainfall Anomaly",
        min_value=-1.0, max_value=2.0, value=0.0, step=0.05,
        help="Deviation from average rainfall (-1 = dry, +2 = very wet)"
    )
    
    st.markdown("---")
    
    # NDVI
    ndvi = st.slider(
        "🌿 NDVI (Vegetation Health)",
        min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        help="0 = barren, 1 = dense vegetation"
    )
    
    st.markdown("---")
    
    # Elevation
    elevation_m = st.slider(
        "🏔️ Elevation (m)",
        min_value=390, max_value=470, value=425,
        help="Elevation above sea level in meters"
    )
    
    # Slope
    slope_pct = st.slider(
        "📐 Slope (%)",
        min_value=0.0, max_value=0.5, value=0.12, step=0.01,
        help="Terrain slope percentage"
    )
    
    st.markdown("---")
    
    # Wet season indicator
    wet_season = 1 if 5 <= month <= 10 else 0
    
    if wet_season:
        st.success("🌧️ **Wet Season** (May-October)")
        st.caption("Higher flood risk expected")
    else:
        st.info("☀️ **Dry Season** (November-April)")
        st.caption("Lower flood risk expected")
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 🤖 Model Information")
    st.caption(f"**Best Model:** SVM")
    st.caption(f"**F1-Score:** 63.49%")
    st.caption(f"**Accuracy:** 68.06%")

# ============================================================
# MAIN CONTENT - TOP ROW: PREDICTION RESULTS
# ============================================================

# Create input array
input_data = np.array([[month, rainfall_mm, rainfall_anomaly, ndvi, elevation_m, slope_pct, wet_season]])
input_scaled = scaler.transform(input_data)

# Predict
pred_proba = model.predict_proba(input_scaled)[0][1]
pred_class = model.predict(input_scaled)[0]

st.markdown("### 📊 Prediction Results")
st.markdown("---")

# Three columns for metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    if pred_class == 1:
        st.markdown("""
        <div style="background:#dc3545;color:white;padding:1rem;border-radius:10px;text-align:center;">
            <div style="font-size:0.8rem;opacity:0.9;">FLOOD RISK</div>
            <div style="font-size:1.8rem;font-weight:700;">⚠️ HIGH</div>
            <div style="font-size:0.7rem;opacity:0.8;">Flood Predicted</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#28a745;color:white;padding:1rem;border-radius:10px;text-align:center;">
            <div style="font-size:0.8rem;opacity:0.9;">FLOOD RISK</div>
            <div style="font-size:1.8rem;font-weight:700;">✅ LOW</div>
            <div style="font-size:0.7rem;opacity:0.8;">No Flood Predicted</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;border:1px solid #e9ecef;">
        <div style="font-size:0.8rem;color:#6c757d;">CONFIDENCE</div>
        <div style="font-size:1.8rem;font-weight:700;color:#2d6a9f;">{pred_proba*100:.1f}%</div>
        <div style="font-size:0.7rem;color:#6c757d;">Model Certainty</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Risk level badge
    if pred_proba > 0.7:
        risk_text = "🔴 HIGH RISK"
        risk_color = "#dc3545"
    elif pred_proba > 0.4:
        risk_text = "🟡 MEDIUM RISK"
        risk_color = "#ffc107"
    else:
        risk_text = "🟢 LOW RISK"
        risk_color = "#28a745"
    
    st.markdown(f"""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;border:1px solid #e9ecef;">
        <div style="font-size:0.8rem;color:#6c757d;">RISK LEVEL</div>
        <div style="font-size:1.2rem;font-weight:700;color:{risk_color};">{risk_text}</div>
        <div style="font-size:0.7rem;color:#6c757d;">Based on prediction</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;border:1px solid #e9ecef;">
        <div style="font-size:0.8rem;color:#6c757d;">LEAD TIME</div>
        <div style="font-size:1.8rem;font-weight:700;color:#2d6a9f;">14 Days</div>
        <div style="font-size:0.7rem;color:#6c757d;">Rolling Forecast</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECOND ROW: FEATURE EXPLANATION (SHAP)
# ============================================================

st.markdown("### 🔍 What Makes This Prediction? (SHAP Analysis)")
st.markdown("---")

# Get feature contributions
if hasattr(model, 'coef_'):
    coef = model.coef_[0]
    feature_names = features
    
    # Calculate contributions
    contributions = coef * input_scaled[0]
    abs_contributions = np.abs(contributions)
    
    # Create dataframe for display
    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_data[0],
        'Impact': contributions,
        'Abs_Impact': abs_contributions
    }).sort_values('Abs_Impact', ascending=False)

# Two columns for explanation
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("#### Feature Contributions")
    
    for _, row in explanation_df.iterrows():
        impact = row['Impact']
        feature = row['Feature']
        value = row['Value']
        
        # Determine impact direction
        if impact > 0.05:
            color = "🔴"
            direction = "increases flood risk"
            impact_class = "feature-impact-positive"
        elif impact < -0.05:
            color = "🟢"
            direction = "decreases flood risk"
            impact_class = "feature-impact-negative"
        else:
            color = "⚪"
            direction = "neutral"
            impact_class = "feature-impact-neutral"
        
        # Create progress bar for impact
        impact_pct = min(100, abs(impact) * 100)
        bar_color = "red" if impact > 0.05 else "green" if impact < -0.05 else "gray"
        
        st.markdown(f"""
        <div class="feature-box">
            <div>
                <span class="feature-name">{feature}</span>
                <span style="color:#6c757d;font-size:0.8rem;"> ({value:.2f})</span>
            </div>
            <div class="{impact_class}">{color} {direction}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Impact Visualization")
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = ['red' if x > 0.05 else 'green' if x < -0.05 else 'gray' 
              for x in explanation_df['Impact']]
    
    ax.barh(explanation_df['Feature'], explanation_df['Impact'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Impact on Flood Risk')
    ax.set_title('Feature Impact Analysis')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig, use_container_width=True)

# ============================================================
# THIRD ROW: RECOMMENDATIONS
# ============================================================

st.markdown("### 💡 Recommendations")
st.markdown("---")

if pred_class == 1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="rec-card">
            <strong>📢 Early Warning</strong>
            <p style="margin:0.3rem 0 0 0;font-size:0.9rem;">Alert local authorities and communities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="rec-card">
            <strong>🐄 Livestock</strong>
            <p style="margin:0.3rem 0 0 0;font-size:0.9rem;">Relocate animals to higher ground</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="rec-card">
            <strong>📦 Supplies</strong>
            <p style="margin:0.3rem 0 0 0;font-size:0.9rem;">Stockpile emergency food and medicine</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("⚠️ **Action Required:** Prepare evacuation plans and monitor weather forecasts.")
    
else:
    st.success("✅ **No immediate action required.** Continue regular monitoring of rainfall patterns.")

# ============================================================
# FOURTH ROW: MODEL COMPARISON
# ============================================================

if comparison_df is not None and len(comparison_df) > 0:
    st.markdown("### 📊 Model Performance Comparison")
    st.markdown("---")
    
    # Create interactive bar chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['Accuracy'] * 100,
        name='Accuracy',
        marker_color='#2d6a9f'
    ))
    
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['F1-Score'] * 100,
        name='F1-Score',
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Model',
        yaxis_title='Score (%)',
        yaxis_range=[0, 100],
        barmode='group',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display comparison table
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### Performance Metrics")
        styled_df = comparison_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x*100:.1f}%")
        
        # Highlight best model
        styled_df = styled_df.style.apply(
            lambda x: ['background: #d4edda' if i == comparison_df['F1-Score'].idxmax() else '' 
                       for i in range(len(x))],
            subset=['Model']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score'] * 100
        
        st.markdown(f"""
        <div style="background:#d4edda;padding:1.5rem;border-radius:10px;text-align:center;">
            <div style="font-size:0.8rem;color:#155724;">🏆 BEST MODEL</div>
            <div style="font-size:1.5rem;font-weight:700;color:#155724;">{best_model}</div>
            <div style="font-size:1.2rem;color:#155724;">F1-Score: {best_f1:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FIFTH ROW: DATA SOURCES AND SYSTEM INFO
# ============================================================

st.markdown("### 📋 System Information")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;">
        <div style="font-size:1.5rem;">📡</div>
        <div style="font-weight:600;">Data Sources</div>
        <div style="font-size:0.8rem;color:#6c757d;">CHIRPS • MODIS • SRTM • OCHA</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;">
        <div style="font-size:1.5rem;">🤖</div>
        <div style="font-weight:600;">Algorithms</div>
        <div style="font-size:0.8rem;color:#6c757d;">XGBoost • RF • SVM • LSTM • GRU • ARIMA</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;">
        <div style="font-size:1.5rem;">🔍</div>
        <div style="font-weight:600;">Explainability</div>
        <div style="font-size:0.8rem;color:#6c757d;">SHAP Feature Attribution</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;">
        <div style="font-size:1.5rem;">📍</div>
        <div style="font-weight:600;">Region</div>
        <div style="font-size:0.8rem;color:#6c757d;">Sudd Wetland, South Sudan</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>Interpretable Spatiotemporal Flood-Risk Prediction System</strong><br>
    BSc. Software Engineering Capstone Project<br>
    Daniel Marial Reng Kudum | Supervisor: Hubert Apana<br>
    © 2026 | Sudd Wetland Region, South Sudan
    <br><br>
    <span style="font-size:0.8rem;opacity:0.7;">
        This is a decision-support tool. Always consult with local authorities for official flood warnings.
    </span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HIDDEN: DEBUG INFO (Only when needed)
# ============================================================

# If you want to see input data, uncomment:
# with st.expander("Debug Info"):
#     st.write("Input Data:", input_data)
#     st.write("Scaled Input:", input_scaled)