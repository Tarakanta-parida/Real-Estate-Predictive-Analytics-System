import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from utils import logger, download_dataset
from feature_engineering import engineer_features

# Column configurations
NUMERICAL_COLS = ['Area', 'BHK', 'Bathroom', 'Floors', 'Garage', 'Parking', 'House_Age', 'Lot_Size', 'Luxury_Score', 'Total_Rooms']
CATEGORICAL_COLS = ['Locality', 'Furnishing', 'Status', 'Transaction', 'Type', 'Condition']
TARGET_COL = 'Price'

def load_raw_data(filepath="dataset/house_prices.csv"):
    """
    Loads raw housing data from the given filepath. Downloads it if missing.
    """
    if not os.path.exists(filepath):
        logger.info(f"Dataset file {filepath} not found locally. Initiating download...")
        download_dataset(dest_path=filepath)
    
    logger.info(f"Loading raw dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise e

def clean_data(df):
    """
    Cleans raw dataframe:
    - Removes duplicate records.
    - Standardizes column names (capitalization, whitespace).
    - Fills missing values in key columns.
    - Casts types properly.
    """
    df = df.copy()
    initial_rows = df.shape[0]
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - df.shape[0]} duplicate rows.")
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Handle missing/invalid numeric values
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')
    df['Bathroom'] = pd.to_numeric(df['Bathroom'], errors='coerce')
    df['Parking'] = pd.to_numeric(df['Parking'], errors='coerce')
    
    # Drop rows where critical target (Price) is missing or zero
    df = df.dropna(subset=['Price'])
    df = df[df['Price'] > 0]
    
    # Impute other missing values
    df['Area'] = df['Area'].fillna(df['Area'].median())
    df['BHK'] = df['BHK'].fillna(df['BHK'].mode()[0]).astype(int)
    df['Bathroom'] = df['Bathroom'].fillna(df['Bathroom'].mode()[0]).astype(int)
    df['Parking'] = df['Parking'].fillna(1).astype(int)
    
    # Impute categorical columns
    df['Furnishing'] = df['Furnishing'].fillna('Semi-Furnished')
    df['Status'] = df['Status'].fillna('Ready to Move')
    df['Transaction'] = df['Transaction'].fillna('Resale')
    df['Type'] = df['Type'].fillna('Apartment')
    df['Locality'] = df['Locality'].fillna('Unknown')
    
    logger.info(f"Data cleaning complete. Rows remaining: {df.shape[0]}")
    return df

def enrich_dataset(df):
    """
    Enriches the dataset with extra requested features in a reproducible way using a deterministic seed:
    - Year_Built and House_Age
    - Floors
    - Garage
    - Lot_Size
    - Condition
    """
    df = df.copy()
    rng = np.random.default_rng(seed=42)
    
    # 1. Year_Built based on Status
    year_built = []
    for status in df['Status']:
        if 'Under Construction' in status:
            year = 2026
        else:
            year = int(rng.integers(1996, 2025))
        year_built.append(year)
    df['Year_Built'] = year_built
    
    # 2. Floors based on Type
    floors = []
    for t in df['Type']:
        if 'Builder Floor' in t:
            floors.append(int(rng.integers(1, 5)))  # 1 to 4 floors
        else:
            floors.append(int(rng.integers(1, 16))) # 1 to 15 floors
    df['Floors'] = floors
    
    # 3. Garage correlated with Parking
    garage = []
    for p in df['Parking']:
        if p <= 1:
            garage.append(0)
        elif p == 2:
            garage.append(1)
        else:
            garage.append(int(rng.integers(1, p)))
    df['Garage'] = garage
    
    # 4. Lot Size correlated with Area and Type
    lot_size = []
    for _, row in df.iterrows():
        area = row['Area']
        t = row['Type']
        if 'Builder Floor' in t:
            factor = float(rng.uniform(1.1, 1.4))
        else:
            factor = 1.0
        lot_size.append(round(area * factor, 2))
    df['Lot_Size'] = lot_size
    
    # 5. Condition based on House_Age (which is 2026 - Year_Built)
    # Temporary calculate age for assigning condition
    temp_age = 2026 - df['Year_Built']
    condition = []
    for age in temp_age:
        if age <= 2:
            condition.append('New')
        elif age <= 8:
            condition.append('Excellent')
        elif age <= 15:
            condition.append('Good')
        elif age <= 25:
            condition.append('Average')
        else:
            condition.append('Fair')
    df['Condition'] = condition
    
    logger.info("Enriched dataset with synthetic Year_Built, Floors, Garage, Lot_Size, and Condition fields.")
    return df

def remove_outliers(df):
    """
    Removes outliers using the IQR method (with a 2.5 multiplier to retain high-end luxury data).
    """
    df = df.copy()
    initial_shape = df.shape[0]
    
    for col in ['Area', 'Price']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2.5 * iqr
        upper_bound = q3 + 2.5 * iqr
        
        # Sane lower bounds (Area must be positive; Price must be realistic)
        lower_bound = max(lower_bound, 100 if col == 'Area' else 500000)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    final_shape = df.shape[0]
    logger.info(f"Outlier removal complete. Removed {initial_shape - final_shape} rows.")
    return df

def fit_transform_pipeline(df, save_path="."):
    """
    Carries out the full preprocessing workflow:
    1. Splits features (X) and target (y)
    2. Performs train-test split (80/20)
    3. Fits categorical encoder (OneHotEncoder) on train features
    4. Fits numerical scaler (StandardScaler) on train features
    5. Saves fitted scaler.pkl and encoder.pkl
    6. Returns X_train_proc, X_test_proc, y_train, y_test, scaler, encoder, and feature names.
    """
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Train-test split (80% Train, 20% Test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Split data into train shape: {X_train.shape} and test shape: {X_test.shape}")
    
    # Fit encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[CATEGORICAL_COLS])
    X_test_cat = encoder.transform(X_test[CATEGORICAL_COLS])
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test_num = scaler.transform(X_test[NUMERICAL_COLS])
    
    # Concatenate numerical and categorical transformations
    X_train_proc = np.hstack((X_train_num, X_train_cat))
    X_test_proc = np.hstack((X_test_num, X_test_cat))
    
    # Save the encoder and scaler
    scaler_file = os.path.join(save_path, "scaler.pkl")
    encoder_file = os.path.join(save_path, "encoder.pkl")
    
    joblib.dump(scaler, scaler_file)
    joblib.dump(encoder, encoder_file)
    logger.info(f"Saved fitted scaler to {scaler_file} and encoder to {encoder_file}")
    
    # Retrieve final feature names
    cat_feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLS))
    feature_names = NUMERICAL_COLS + cat_feature_names
    
    return X_train_proc, X_test_proc, y_train, y_test, scaler, encoder, feature_names
