import os
import joblib
import pandas as pd
import numpy as np

from utils import logger
from feature_engineering import engineer_features
from preprocessing import NUMERICAL_COLS, CATEGORICAL_COLS

class HousePricePredictor:
    def __init__(self, model_dir="."):
        """
        Loads the trained model, scaler, and encoder pickles.
        """
        self.model_path = os.path.join(model_dir, "house_price_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.encoder_path = os.path.join(model_dir, "encoder.pkl")
        
        if not (os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.encoder_path)):
            raise FileNotFoundError(
                "Model files not found. Make sure to run train_model.py first "
                "to generate house_price_model.pkl, scaler.pkl, and encoder.pkl."
            )
            
        logger.info("Loading model, scaler, and encoder pickles...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)
        logger.info("Model files loaded successfully.")

    def predict(self, input_dict):
        """
        Predicts house price for a single input dictionary.
        input_dict keys:
            - Area: float
            - BHK: int (Bedrooms)
            - Bathroom: int
            - Locality: str (Location)
            - Parking: int
            - Furnishing: str ('Furnished', 'Semi-Furnished', 'Unfurnished')
            - Status: str ('Ready to Move', 'Under Construction')
            - Transaction: str ('Resale', 'New Property')
            - Type: str ('Apartment', 'Builder Floor')
            - Year_Built: int
            - Floors: int
            - Garage: int
            - Lot_Size: float (optional)
            - Condition: str ('New', 'Excellent', 'Good', 'Average', 'Fair')
        """
        # Create DataFrame from input dictionary
        df = pd.DataFrame([input_dict])
        
        # 1. Fill optional values
        if 'Lot_Size' not in df.columns or pd.isna(df['Lot_Size'][0]):
            df['Lot_Size'] = df['Area']
            
        # Validate inputs
        self._validate_inputs(df.iloc[0])
            
        # 2. Engineer features (Total_Rooms, Luxury_Score, etc.)
        df = engineer_features(df)
        
        # 3. Separate numerical and categorical columns
        df_num = df[NUMERICAL_COLS]
        df_cat = df[CATEGORICAL_COLS]
        
        # 4. Transform numerical features
        num_scaled = self.scaler.transform(df_num)
        
        # 5. Transform categorical features
        cat_encoded = self.encoder.transform(df_cat)
        
        # 6. Concatenate
        X_processed = np.hstack((num_scaled, cat_encoded))
        
        # 7. Model predict
        prediction = self.model.predict(X_processed)[0]
        
        # Ensure prediction is positive and realistic
        return max(float(prediction), 100000.0)

    def _validate_inputs(self, row):
        """
        Performs basic bounds checks to ensure input values are reasonable.
        """
        if row['Area'] <= 0:
            raise ValueError("Area must be a positive number.")
        if row['BHK'] < 1:
            raise ValueError("Number of bedrooms (BHK) must be at least 1.")
        if row['Bathroom'] < 1:
            raise ValueError("Number of bathrooms must be at least 1.")
        if row['Floors'] < 1:
            raise ValueError("Number of floors must be at least 1.")
        if row['Parking'] < 0:
            raise ValueError("Parking spaces cannot be negative.")
        if row['Garage'] < 0:
            raise ValueError("Garage spaces cannot be negative.")
        if row['Year_Built'] < 1800 or row['Year_Built'] > 2030:
            raise ValueError("Year Built must be between 1800 and 2030.")
        if row['Lot_Size'] <= 0:
            raise ValueError("Lot Size must be a positive number.")
