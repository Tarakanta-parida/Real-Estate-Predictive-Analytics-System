import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Creates new engineered features from the existing columns:
    - House_Age: Years since construction (relative to 2026)
    - Total_Rooms: Sum of Bedrooms (BHK) and Bathrooms
    - Luxury_Score: A custom calculated index from 1 to 5 based on amenities and property state.
    """
    df = df.copy()
    
    # 1. House Age (if Year_Built is present)
    if 'Year_Built' in df.columns:
        df['House_Age'] = 2026 - df['Year_Built']
    elif 'House_Age' not in df.columns:
        # Fallback default age
        df['House_Age'] = 10
        
    # 2. Total Rooms (BHK + Bathroom)
    bhk = df['BHK'].fillna(1).astype(float)
    bathroom = df['Bathroom'].fillna(1).astype(float)
    df['Total_Rooms'] = bhk + bathroom
    
    # 3. Luxury Score (1 to 5)
    # Calculate score based on parking, furnishing, status, and condition
    scores = np.ones(len(df)) # start at base 1.0
    
    # Furnishing contributions
    if 'Furnishing' in df.columns:
        scores += df['Furnishing'].map({
            'Furnished': 2.0,
            'Semi-Furnished': 1.0,
            'Unfurnished': 0.0
        }).fillna(0.0)
        
    # Parking contributions
    if 'Parking' in df.columns:
        parking_contrib = df['Parking'].fillna(0).astype(float).apply(
            lambda x: 1.5 if x >= 2 else (0.5 if x == 1 else 0.0)
        )
        scores += parking_contrib
        
    # Condition contributions
    if 'Condition' in df.columns:
        scores += df['Condition'].map({
            'New': 1.5,
            'Excellent': 1.5,
            'Good': 0.5,
            'Average': 0.0,
            'Fair': -0.5
        }).fillna(0.0)
        
    # Clip between 1.0 and 5.0
    df['Luxury_Score'] = np.clip(scores, 1.0, 5.0)
    
    # 4. Price per Square Foot (only for analysis, NOT for model training to prevent target leakage)
    if 'Price' in df.columns and 'Area' in df.columns:
        # Avoid division by zero
        area = df['Area'].replace(0, np.nan)
        df['Price_per_SqFt'] = df['Price'] / area
        # Fill NaNs with median if area was 0
        df['Price_per_SqFt'] = df['Price_per_SqFt'].fillna(df['Price_per_SqFt'].median())
        
    return df
