import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

from utils import format_indian_currency, format_indian_currency_short, logger
from predict import HousePricePredictor

# Set page configuration
st.set_page_config(
    page_title="Delhi House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Card design */
    .metric-box {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    /* Gradient prediction card */
    .prediction-box {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
        color: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
    }
    
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-val {
        font-size: 3rem;
        font-weight: 700;
        margin: 15px 0;
        letter-spacing: -1px;
    }
    
    .prediction-desc {
        font-size: 0.95rem;
        opacity: 0.8;
    }
    
    /* Title Stylings */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 20px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 30px;
    }
    
    /* Custom buttons */
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to check if models are trained
def is_model_trained():
    return os.path.exists("house_price_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("encoder.pkl")

# Load unique localities and types for dropdowns
@st.cache_data
def get_dataset_dropdown_options():
    default_localities = [
        'Lajpat Nagar', 'Vasant Kunj', 'Punjabi Bagh', 'Dwarka Sector 13', 
        'Saket', 'Kalkaji', 'Okhla Phase 1', 'Patparganj', 'Chhattarpur', 
        'Uttam Nagar', 'Rohini Sector 24', 'Greater Kailash'
    ]
    default_furnishings = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    default_status = ['Ready to Move', 'Under Construction']
    default_transactions = ['Resale', 'New Property']
    default_types = ['Apartment', 'Builder Floor']
    default_conditions = ['Excellent', 'Good', 'New', 'Average', 'Fair']
    
    processed_path = "dataset/house_prices_processed.csv"
    if os.path.exists(processed_path):
        try:
            df = pd.read_csv(processed_path)
            localities = sorted(df['Locality'].unique().tolist())
            furnishings = sorted(df['Furnishing'].unique().tolist())
            status = sorted(df['Status'].unique().tolist())
            transactions = sorted(df['Transaction'].unique().tolist())
            types = sorted(df['Type'].unique().tolist())
            conditions = sorted(df['Condition'].unique().tolist())
            return localities, furnishings, status, transactions, types, conditions
        except Exception as e:
            logger.warning(f"Error loading dropdown options from CSV: {e}")
            
    return default_localities, default_furnishings, default_status, default_transactions, default_types, default_conditions

# Main app entry
def main():
    # Sidebar navigation
    st.sidebar.markdown(
        "<div style='text-align: center; padding-bottom: 20px;'>"
        "<h1 style='color: #4F46E5; font-size: 1.8rem; font-weight: 800; margin: 0;'>Delhi Real Estate</h1>"
        "<p style='color: #64748B; font-size: 0.9rem; margin-top: 5px;'>Predictive Analytics System</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home & Overview", "Dataset Preview", "Data Visualizations", "Predict Price", "Batch Prediction", "Model Performance"]
    )
    
    # Global model check warning
    if not is_model_trained() and page in ["Predict Price", "Batch Prediction", "Model Performance"]:
        st.warning("⚠️ Warning: The machine learning models have not been trained yet. Please run `train_model.py` first to generate the models and charts.")
        st.info("💡 You can navigate to 'Home & Overview' or 'Dataset Preview' in the meantime.")
    
    # ------------------
    # Page: Home
    # ------------------
    if page == "Home & Overview":
        st.markdown("<h1 class='section-title'>🏠 House Price Prediction System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>A production-ready machine learning solution to estimate property valuations in Delhi using Linear Regression and tree ensemble models.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class='metric-box'>
                <h3>🎯 Project Objective</h3>
                <p>Estimating real estate prices in high-density urban areas like Delhi is highly complex due to varying local factors. This system provides real-estate brokers, buyers, and sellers with a data-driven model to predict house valuations. It processes real-world MagicBricks listings, performs outlier filtration, engineers key features, and deploys tuned regressors.</p>
                
                <h4>🏗️ Pipeline Flow</h4>
                <ol>
                    <li><b>Data Ingestion:</b> Downloads raw MagicBricks CSV dataset from GitHub.</li>
                    <li><b>Cleaning & Imputation:</b> Handles missing values, removes duplicates, and filters extreme outliers using IQR.</li>
                    <li><b>Feature Engineering:</b> Computes <code>House Age</code>, <code>Total Rooms</code>, and custom <code>Luxury Score</code> indices.</li>
                    <li><b>Model Training:</b> Fits multiple regressors (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest) using 5-Fold Cross-Validation.</li>
                    <li><b>Hyperparameter Tuning:</b> Runs <code>GridSearchCV</code> to optimize Random Forest Regressor.</li>
                    <li><b>Inference:</b> Loads saved <code>scaler.pkl</code>, <code>encoder.pkl</code>, and <code>house_price_model.pkl</code> for fast pricing predictions.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-box'>
                <h3>🛠️ Technology Stack</h3>
                <ul>
                    <li><b>Language:</b> Python 3.11+</li>
                    <li><b>Libraries:</b></li>
                    <ul>
                        <li>Scikit-Learn</li>
                        <li>Pandas & NumPy</li>
                        <li>Joblib (Serialization)</li>
                    </ul>
                    <li><b>Visualizations:</b></li>
                    <ul>
                        <li>Plotly Express</li>
                        <li>Seaborn / Matplotlib</li>
                    </ul>
                    <li><b>Frontend:</b> Streamlit Dashboard</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        # Display system status
        status_color = "green" if is_model_trained() else "orange"
        status_text = "Trained & Active" if is_model_trained() else "Untrained (Run train_model.py)"
        
        st.markdown(f"""
        <div style='background-color: white; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0; display: flex; align-items: center; justify-content: space-between;'>
            <div>
                <span style='font-weight: 600; color: #1E293B;'>ML Engine Status:</span>
                <span style='color: {status_color}; font-weight: 700; margin-left: 8px;'>● {status_text}</span>
            </div>
            <div style='font-size: 0.9rem; color: #64748B;'>
                Current Directory: <code>c:\\Users\\Dell\\OneDrive\\Desktop\\python_internship_project3</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------
    # Page: Dataset Preview
    # ------------------
    elif page == "Dataset Preview":
        st.markdown("<h1 class='section-title'>📊 Dataset Preview & Statistics</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Explore the raw dataset shape, schemas, and metrics before training.</p>", unsafe_allow_html=True)
        
        raw_path = "dataset/house_prices.csv"
        processed_path = "dataset/house_prices_processed.csv"
        
        if not os.path.exists(raw_path):
            st.info("📥 Dataset not found locally. Click below to download it.")
            if st.button("Download Dataset"):
                with st.spinner("Downloading dataset..."):
                    from utils import download_dataset
                    download_dataset()
                    st.success("Dataset downloaded!")
                    st.rerun()
            return
            
        df = pd.read_csv(raw_path)
        
        st.markdown("### 🔍 Dataset Summary")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Records", f"{df.shape[0]:,}")
        m_col2.metric("Features Count", f"{df.shape[1]}")
        m_col3.metric("Target Variable", "Price (INR)")
        
        st.markdown("### 📋 First & Last Records Preview")
        st.write("First 5 rows:")
        st.dataframe(df.head())
        st.write("Last 5 rows:")
        st.dataframe(df.tail())
        
        tab1, tab2 = st.columns(2)
        with tab1:
            st.markdown("### 🗂️ Data Schema & Missing Values")
            dtypes_df = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Null Values": df.isnull().sum(),
                "Non-Null Count": df.notnull().sum()
            })
            st.dataframe(dtypes_df)
            
        with tab2:
            st.markdown("### 🔢 Numerical Metrics Summary")
            st.dataframe(df.describe())

    # ------------------
    # Page: Data Visualizations
    # ------------------
    elif page == "Data Visualizations":
        st.markdown("<h1 class='section-title'>📈 Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Interactive visualizations to understand features, price distributions, and correlation patterns.</p>", unsafe_allow_html=True)
        
        processed_path = "dataset/house_prices_processed.csv"
        if not os.path.exists(processed_path):
            st.warning("⚠️ Processed dataset not found. Please execute the training pipeline first to build feature distributions.")
            return
            
        df = pd.read_csv(processed_path)
        
        viz_option = st.selectbox(
            "Select Chart to Display",
            [
                "Correlation Heatmap", 
                "Price Distribution by Room Counts (BHK & Bathrooms)", 
                "Top 10 Locality Price Trends",
                "Price Distribution by Furnishing Status",
                "Average Price by Property Age & Type"
            ]
        )
        
        if viz_option == "Correlation Heatmap":
            st.markdown("#### Correlation Heatmap")
            st.write("Heatmap of Pearson correlation coefficients between numerical features and the target variable.")
            num_cols = df.select_dtypes(include=[np.number]).columns
            corr_cols = [c for c in num_cols if c not in ['Year_Built', 'Price_per_SqFt']]
            corr_matrix = df[corr_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Correlation Matrix Heatmap"
            )
            fig.update_layout(width=800, height=700)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Price Distribution by Room Counts (BHK & Bathrooms)":
            st.markdown("#### Price by Room Counts")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.box(df, x="BHK", y="Price", color="BHK", title="Price vs. Bedrooms (BHK)")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.box(df, x="Bathroom", y="Price", color="Bathroom", title="Price vs. Bathrooms")
                st.plotly_chart(fig2, use_container_width=True)
                
        elif viz_option == "Top 10 Locality Price Trends":
            st.markdown("#### Top 10 Most Expensive Localities")
            top_localities = df.groupby('Locality')['Price'].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(
                top_localities, x="Price", y="Locality", 
                orientation="h", color="Price",
                color_continuous_scale="Viridis",
                title="Delhi's Top 10 Most Expensive Localities (Average Price)"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Price Distribution by Furnishing Status":
            st.markdown("#### Price Distribution by Furnishing Status")
            st.write("Box and points distribution of house prices based on whether they are furnished, semi-furnished, or unfurnished.")
            fig = px.box(
                df, x="Furnishing", y="Price", color="Furnishing",
                points="all",
                title="House Price by Furnishing Status",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Average Price by Property Age & Type":
            st.markdown("#### Property Age vs Average Price")
            st.write("Line plot showing the average price of houses grouped by Property Age and Type (Apartment vs Builder Floor).")
            age_df = df.groupby(['House_Age', 'Type'])['Price'].mean().reset_index()
            fig = px.line(
                age_df, x="House_Age", y="Price", color="Type", markers=True,
                title="Average House Price vs. Property Age",
                labels={"House_Age": "Age of the Property (Years)", "Price": "Average Price (INR)"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------
    # Page: Predict Price
    # ------------------
    elif page == "Predict Price":
        st.markdown("<h1 class='section-title'>🔮 Property Valuation Calculator</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Fill in the features below to predict the estimated valuation of the house in Delhi.</p>", unsafe_allow_html=True)
        
        localities, furnishings, status, transactions, types, conditions = get_dataset_dropdown_options()
        
        # Load predictor
        try:
            predictor = HousePricePredictor()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
            
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 📐 Area & Dimensions")
                area = st.number_input("Property Area (Square Feet)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0)
                lot_size = st.number_input("Lot Size (Square Feet) - Optional", min_value=100.0, max_value=25000.0, value=area, step=50.0)
                floors = st.number_input("Total Floors in Building", min_value=1, max_value=30, value=4, step=1)
                
            with col2:
                st.markdown("##### 🛏️ Rooms & Configuration")
                bhk = st.slider("Bedrooms (BHK)", min_value=1, max_value=10, value=3)
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=10, value=3)
                parking = st.slider("Parking Spaces", min_value=0, max_value=10, value=1)
                garage = st.slider("Private Garages", min_value=0, max_value=5, value=0)
                
            with col3:
                st.markdown("##### 🏷️ Property Attributes")
                locality = st.selectbox("Locality / Location", localities)
                property_type = st.selectbox("Property Type", types)
                furnishing = st.selectbox("Furnishing Status", furnishings)
                status_type = st.selectbox("Construction Status", status)
                transaction = st.selectbox("Transaction Type", transactions)
                condition = st.selectbox("Overall Property Condition", conditions)
                year_built = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015, step=1)
                
            # Submit button
            submit = st.form_submit_button("Calculate Property Value")
            
        if submit:
            input_dict = {
                'Area': area,
                'BHK': bhk,
                'Bathroom': bathrooms,
                'Locality': locality,
                'Parking': parking,
                'Furnishing': furnishing,
                'Status': status_type,
                'Transaction': transaction,
                'Type': property_type,
                'Year_Built': year_built,
                'Floors': floors,
                'Garage': garage,
                'Lot_Size': lot_size,
                'Condition': condition
            }
            
            with st.spinner("Processing preprocessors and predicting price..."):
                try:
                    predicted_price = predictor.predict(input_dict)
                    
                    # Short readable format
                    short_price = format_indian_currency_short(predicted_price)
                    full_price = format_indian_currency(predicted_price)
                    
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <div class='prediction-title'>Estimated Market Valuation</div>
                        <div class='prediction-val'>{short_price}</div>
                        <div class='prediction-desc'>Full valuation: <b>{full_price}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Contextual feedback
                    # Load training mean price
                    processed_path = "dataset/house_prices_processed.csv"
                    if os.path.exists(processed_path):
                        df_p = pd.read_csv(processed_path)
                        avg_price = df_p['Price'].mean()
                        diff = predicted_price - avg_price
                        percent_diff = (diff / avg_price) * 100
                        
                        comp_text = "higher" if diff > 0 else "lower"
                        st.info(f"💡 This valuation is **{abs(percent_diff):.1f}% {comp_text}** than the average house price in our Delhi dataset (Avg: {format_indian_currency_short(avg_price)}).")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ------------------
    # Page: Batch Prediction
    # ------------------
    elif page == "Batch Prediction":
        st.markdown("<h1 class='section-title'>📥 Bulk Valuation System (CSV)</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Upload a CSV file containing multiple property configurations to perform batch predictions.</p>", unsafe_allow_html=True)
        
        # Guide user on expected CSV columns
        st.markdown("""
        **Required CSV Columns:**
        `Area`, `BHK`, `Bathroom`, `Locality`, `Parking`, `Furnishing`, `Status`, `Transaction`, `Type`, `Year_Built`, `Floors`, `Garage`, `Lot_Size` (optional), `Condition`
        """)
        
        # Download template button
        template_df = pd.DataFrame([{
            'Area': 1500.0, 'BHK': 3, 'Bathroom': 3, 'Locality': 'Saket', 'Parking': 2,
            'Furnishing': 'Semi-Furnished', 'Status': 'Ready to Move', 'Transaction': 'Resale',
            'Type': 'Apartment', 'Year_Built': 2018, 'Floors': 4, 'Garage': 1, 'Lot_Size': 1500.0,
            'Condition': 'Good'
        }])
        
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV Template",
            data=template_csv,
            file_name="house_prediction_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.markdown("### Uploaded Data Preview:")
                st.dataframe(input_df.head())
                
                # Check predictor
                predictor = HousePricePredictor()
                
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch inputs..."):
                        predictions = []
                        formatted_prices = []
                        
                        for idx, row in input_df.iterrows():
                            # Reconstruct row into dictionary
                            row_dict = row.to_dict()
                            try:
                                pred = predictor.predict(row_dict)
                                predictions.append(pred)
                                formatted_prices.append(format_indian_currency_short(pred))
                            except Exception as ex:
                                predictions.append(np.nan)
                                formatted_prices.append(f"Failed: {ex}")
                                
                        input_df['Predicted_Price'] = predictions
                        input_df['Predicted_Price_Formatted'] = formatted_prices
                        
                        st.success("Batch predictions completed!")
                        st.dataframe(input_df)
                        
                        # Download results button
                        results_csv = input_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Prediction Report (CSV)",
                            data=results_csv,
                            file_name="house_predictions_output.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # ------------------
    # Page: Model Performance
    # ------------------
    elif page == "Model Performance":
        st.markdown("<h1 class='section-title'>⚙️ Model Training Performance & Metrics</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Compare standard regression metrics (MAE, RMSE, R² Score) for all models evaluated during the training phase.</p>", unsafe_allow_html=True)
        
        metrics_path = "dataset/model_metrics.json"
        
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_list = json.load(f)
                
            metrics_df = pd.DataFrame(metrics_list)
            
            # Format metrics
            metrics_df_display = metrics_df.copy()
            metrics_df_display['Test R2'] = metrics_df_display['Test R2'].apply(lambda x: f"{x:.4f}")
            metrics_df_display['Test MAE'] = metrics_df_display['Test MAE'].apply(lambda x: format_indian_currency_short(x))
            metrics_df_display['Test RMSE'] = metrics_df_display['Test RMSE'].apply(lambda x: format_indian_currency_short(x))
            metrics_df_display['Test MSE'] = metrics_df_display['Test MSE'].apply(lambda x: f"{x:,.0f}")
            
            st.markdown("### 📊 Metrics Comparison Table")
            st.dataframe(metrics_df_display.sort_values(by='Test R2', ascending=False), use_container_width=True)
            
            # Plot metrics comparison
            st.markdown("### 🏆 Test R² Score Comparison")
            fig = px.bar(
                metrics_df, x="Model", y="Test R2", 
                color="Model", color_discrete_sequence=px.colors.qualitative.Safe,
                title="Model Test R2 Scores (Higher is Better)"
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot MAE comparison
            st.markdown("### 📉 Test Mean Absolute Error (MAE) Comparison")
            fig_mae = px.bar(
                metrics_df, x="Model", y="Test MAE", 
                color="Model", color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Model Test MAE in INR (Lower is Better)"
            )
            st.plotly_chart(fig_mae, use_container_width=True)
            
            # Show static charts if they exist
            st.markdown("### 📈 Evaluation Plots")
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists("charts/actual_vs_predicted.png"):
                    st.image("charts/actual_vs_predicted.png", caption="Actual vs Predicted House Prices Scatter Plot")
                else:
                    st.info("Scatter plot chart not found. Run model training to generate.")
                    
            with col2:
                if os.path.exists("charts/residuals_distribution.png"):
                    st.image("charts/residuals_distribution.png", caption="Residuals Distribution KDE Plot")
                else:
                    st.info("Residuals distribution chart not found. Run model training to generate.")
                    
            st.markdown("### 🔍 Feature Importance")
            if os.path.exists("charts/feature_importance.png"):
                st.image("charts/feature_importance.png", caption="Relative Importances of Features")
            else:
                st.info("Feature importance chart not found. Run model training to generate.")
                
        else:
            st.info("No model metrics file found. Please run train_model.py to perform model comparisons.")

if __name__ == "__main__":
    main()
