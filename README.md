# Delhi House Price Prediction System

A production-ready, modular, and interactive Machine Learning application built using Python to estimate house prices in Delhi. The system applies regression modeling on real-world property listings scraped from MagicBricks, performing robust cleaning, data enrichment, and model optimization.

---

## 🚀 Project Overview

Predicting property valuations in dense metropolitan areas like Delhi is challenging due to the complex relationship between physical parameters (Area, BHK) and geographical location (Locality). This system solves this by:
1. **Fetching and Cleaning** a real-world listing dataset.
2. **Enriching Features** with statistically correlated parameters (`House Age`, `Floors`, `Private Garage`, `Lot Size`, `Condition`).
3. **Comparing Multiple Regressors** (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest).
4. **Optimizing Hyperparameters** using 5-Fold Cross-Validation and Grid Search.
5. **Providing an Interactive UI** via Streamlit for single property valuations and bulk predictions.

---

## 🗂️ Folder Structure

```
c:\Users\Dell\OneDrive\Desktop\python_internship_project3/
│── app.py                   # Streamlit interactive application
│── train_model.py           # Orchestration script to run ML pipeline
│── predict.py               # Inference class for single/batch predictions
│── preprocessing.py         # Data cleaning, outlier removal, & pipeline fitting
│── feature_engineering.py   # Creation of engineered metrics (Age, Luxury Score)
│── model.py                 # Regressor instantiation, evaluation, & GridSearchCV
│── utils.py                 # Currency formatting, dataset downloading, & logging
│── requirements.txt         # Project package dependencies
│── README.md                # System documentation
│── house_price_model.pkl    # Serialized Random Forest model (joblib)
│── scaler.pkl               # Serialized numerical scaler (joblib)
│── encoder.pkl              # Serialized categorical encoder (joblib)
├── dataset/
│   ├── house_prices.csv     # Local cache of downloaded MagicBricks dataset
│   ├── house_prices_processed.csv # Preprocessed dataset including engineered features
│   └── model_metrics.json   # Model evaluation scores in JSON format
├── notebooks/
│   └── EDA.ipynb            # Jupyter Notebook conducting step-by-step EDA
├── assets/                  # CSS styling or project assets
└── charts/                  # Evaluation plots generated during training
    ├── actual_vs_predicted.png
    ├── residuals_distribution.png
    └── feature_importance.png
```

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.11+
- **Machine Learning & Preprocessing:** Scikit-Learn, Pandas, NumPy
- **Model Persistence:** Joblib
- **Interactive Visualizations:** Plotly Express
- **Static Visualizations:** Seaborn, Matplotlib
- **Web App Interface:** Streamlit

---

## 📊 Dataset Description

The system utilizes the **Delhi House Price Dataset (MagicBricks)**, consisting of listing configurations:
- **Price (Target):** Valuation in Indian Rupees (INR).
- **Area:** Size of the listing in square feet.
- **BHK:** Number of bedrooms (BHK).
- **Bathroom:** Number of bathroom facilities.
- **Locality:** Specific neighborhood zone in Delhi (e.g. Saket, Vasant Kunj).
- **Parking:** Allocated parking spaces.
- **Furnishing:** Property furnishing state (`Furnished`, `Semi-Furnished`, `Unfurnished`).
- **Status:** Construction state (`Ready to Move`, `Under Construction`).
- **Transaction:** Listing category (`Resale`, `New Property`).
- **Type:** Listing type (`Apartment`, `Builder Floor`).

### Engineered / Enriched Features:
- **Year_Built:** Determined based on construction status (deterministic pseudo-random distribution).
- **House_Age:** Computed as `2026 - Year_Built`.
- **Floors:** Total building levels, statistically correlated with listing type.
- **Garage:** Private garage capacity derived from parking allotment.
- **Lot_Size:** Outer land footprint scaled based on building area and type.
- **Condition:** State assessment (`New`, `Excellent`, `Good`, `Average`, `Fair`) calculated from age and status.
- **Total_Rooms:** Combined sum of Bedrooms (BHK) and Bathrooms.
- **Luxury_Score:** An index from 1.0 to 5.0 scoring property quality based on furnishing, parking counts, and condition.

---

## 🧹 Data Preprocessing & Feature Engineering Steps

1. **Deduplication:** Removal of duplicate listings.
2. **Imputation:** Numeric values imputed with column medians; categorical fields filled with modes.
3. **Deterministic Enrichment:** Statistical imputation of missing parameters (e.g. Age, Lot Size, Floors) using a fixed seed (`random_state=42`) to guarantee identical dataset builds.
4. **Outlier Removal:** Uses Interquartile Range (IQR) with a `2.5` multiplier on `Price` and `Area` to filter extreme data anomalies while retaining premium-tier luxury properties.
5. **Feature Scaling:** Standardizes numerical columns using `StandardScaler`.
6. **Categorical Encoding:** Maps categories into binary vectors using `OneHotEncoder` configured with `handle_unknown='ignore'` to handle new locality entries dynamically.
7. **Leakage Prevention:** Transformers are fitted **only** on the training split (80%), then applied independently to transform test and runtime inference splits.

---

## 📈 Model Comparison & Evaluation

Multiple regression models were trained, cross-validated (5-Fold), and evaluated on the 20% test split. Standard metrics calculated:
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

The **Tuned Random Forest Regressor** (tuned via `GridSearchCV` on estimators, depths, and split criteria) serves as the primary production model, outperforming baseline Linear Regression by capturing non-linear pricing effects across localities.

---

## 💻 Installation & Usage Guide

### Prerequisites
- Python 3.11+ installed.

### 1. Setup Environment
Clone or navigate to the project directory:
```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
```

### 2. Install Dependencies
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

### 3. Run Model Training Pipeline
Executes preprocessing, model selection, hyperparameter grid search, and serializes model files:
```powershell
.\.venv\Scripts\python train_model.py
```
This command generates:
- `house_price_model.pkl`, `scaler.pkl`, `encoder.pkl` (Model pickles)
- `dataset/house_prices_processed.csv` and `dataset/model_metrics.json`
- `charts/` (Visualizations)

### 4. Generate the Jupyter Notebook
Run the helper notebook compiler:
```powershell
.\.venv\Scripts\python -c "import sys; sys.path.append(r'C:\Users\Dell\.gemini\antigravity-ide\brain\93dd62f9-14af-42f9-ae0c-107a13c21d08\scratch'); import generate_notebook"
```

### 5. Launch the Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit run app.py
```

---

## 🌐 Deployment Instructions

- **Streamlit Cloud:** Connect your GitHub repository containing this layout. Configure the run command to `streamlit run app.py`.
- **Render / Docker:**
  A basic `Dockerfile` can expose port `8501` to serve the Streamlit app.
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install -r requirements.txt
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

---

## 🔮 Future Improvements

- **Interactive Geospatial Map:** Integrating Geopy to map coordinates of listings and plot them on an interactive map.
- **SHAP Integration:** Integrating the SHAP package to display local feature attribution bars directly on the Streamlit prediction box.
- **API Endpoint:** Expose a FastAPI microservice alongside the Streamlit client.
