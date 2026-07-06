# 🌊 Interpretable Flood-Risk Prediction System

### Sudd Wetland Region, South Sudan

An interpretable machine learning system for predicting flood risk in the Sudd Wetland Region of South Sudan using satellite-derived environmental data and historical flood information. The project combines multiple machine learning models with explainable AI (SHAP) to support disaster preparedness and informed decision-making.

---

## 📖 Overview

Flooding is one of the most destructive natural hazards affecting communities across South Sudan. This project develops an interpretable flood prediction system capable of providing **14-day early warning forecasts** by integrating multiple environmental datasets and comparing six machine learning models.

The application provides:

- 🌧️ Flood risk prediction
- 📊 Interactive visualizations
- 🧠 Explainable AI using SHAP
- 📈 Model comparison dashboard
- 🚨 Actionable recommendations for disaster preparedness

---

## ✨ Key Features

- Predicts flood risk using six machine learning models
- Compares classical ML and deep learning approaches
- Uses SHAP explainability for transparent predictions
- Interactive Streamlit dashboard
- 14-day flood forecasting capability
- Built specifically for the Sudd Wetland Region of South Sudan

---

## 🌐 Live Application

🔗 **Streamlit App**

https://sudd-flood-predict.streamlit.app

---

## 📹 Project Demonstration

🎥 **Video Walkthrough**

https://screenrec.com/share/gHVDLylB1n

---

# 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------:|----------:|--------:|----------:|---------:|
| **Support Vector Machine (Best)** | **68.06%** | **64.29%** | **62.70%** | **63.49%** | **70.81%** |
| ARIMA | 66.67% | 61.54% | 58.82% | 60.00% | 64.83% |
| XGBoost | 61.11% | 57.14% | 50.00% | 53.33% | 59.69% |
| LSTM | 61.11% | 57.14% | 50.00% | 53.33% | 59.06% |
| GRU | 61.11% | 57.14% | 50.00% | 53.33% | 56.87% |
| Random Forest | 61.11% | 60.00% | 45.45% | 51.72% | 64.75% |

---

# 📂 Project Structure

```
interpretable-flood-prediction/
│
├── app.py
├── generate_data.py
├── train_svm.py
├── train_rf.py
├── train_xgboost.py
├── train_lstm.py
├── train_gru.py
├── train_arima.py
│
├── models/
├── processed_data/
├── figures/
├── shap_outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/MarialRK/interpretable-flood-prediction.git

cd interpretable-flood-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Generate the Dataset

```bash
python generate_data.py
```

---

## 5. Train the Models

```bash
python train_svm.py

python train_rf.py

python train_xgboost.py

python train_lstm.py

python train_gru.py

python train_arima.py
```

---

## 6. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at

```
http://localhost:8501
```

---

# 🛰️ Data Sources

| Dataset | Source | Purpose |
|----------|--------|---------|
| CHIRPS | UCSB | Rainfall estimation |
| MODIS NDVI | NASA | Vegetation monitoring |
| SRTM DEM | NASA | Elevation |
| OCHA Flood Records | United Nations | Historical flood events |

---

# 🔬 Methodology

The workflow consists of:

1. Data collection
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. Explainability using SHAP
7. Deployment with Streamlit

---

# 📊 Results

Among all evaluated models, **Support Vector Machine (SVM)** achieved the best overall performance.

### Best Model

- Accuracy: **68.06%**
- Precision: **64.29%**
- Recall: **62.70%**
- F1 Score: **63.49%**
- ROC-AUC: **70.81%**

The results demonstrate that traditional machine learning models can outperform deep learning models when working with relatively small environmental datasets.

---

# 🧠 Explainable AI

This project integrates **SHAP (SHapley Additive Explanations)** to improve model transparency.

SHAP enables users to:

- Understand why a prediction was made
- Identify the most influential variables
- Improve trust in machine learning predictions
- Support decision-making for disaster management

---

# 🎯 Project Objectives

## Objective 1

Collect and integrate environmental datasets.

✅ Achieved

---

## Objective 2

Compare multiple machine learning approaches.

✅ Achieved

Six models were successfully implemented and evaluated.

---

## Objective 3

Develop an interpretable prediction platform.

✅ Achieved

Interactive Streamlit dashboard with SHAP explanations.

---

# 📈 Discussion

## Contributions

- Demonstrates flood prediction using satellite-derived datasets.
- Provides interpretable machine learning for humanitarian applications.
- Supports disaster preparedness in data-scarce environments.
- Establishes a baseline framework for future flood prediction research in South Sudan.

---

# ⚠️ Limitations

- Dataset covers only 2019–2020.
- Limited historical flood observations.
- Additional environmental variables could further improve prediction accuracy.

---

# 🚀 Future Improvements

- Extend to 15+ years of historical data.
- Connect to live satellite APIs.
- Deploy automated daily predictions.
- Develop Android and iOS mobile applications.
- Integrate SMS-based early warning notifications.
- Improve prediction accuracy using larger datasets.

---

# 📸 Screenshots

Add screenshots here after deployment.

Example:

```
screenshots/

dashboard.png

prediction.png

shap_summary.png

model_comparison.png
```

---

# 📚 Technologies Used

- Python
- Streamlit
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- SHAP
- Pandas
- NumPy
- Matplotlib

---

# 👨‍💻 Author

**Daniel Marial Reng Kudum**

BSc Software Engineering

African Leadership University

Supervisor: **Hubert Apana**

---

# 📄 Citation

If you use this project, please cite:

```
Daniel Marial Reng Kudum.

Interpretable Flood-Risk Prediction System for the Sudd Wetland Region of South Sudan.

African Leadership University.

2026.
```

---

# 📜 License

This project was developed for academic and research purposes.

© 2026 Daniel Marial Reng Kudum. All rights reserved.
