import os
import urllib.request
import logging

def setup_logging():
    """Sets up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("HousePricePrediction")

logger = setup_logging()

def format_indian_currency(amount):
    """
    Formats a numeric amount into the Indian currency layout (e.g., 8250000 -> ₹ 82,50,000).
    """
    try:
        amount = int(round(float(amount)))
        neg = "-" if amount < 0 else ""
        amount = abs(amount)
        s = str(amount)
        if len(s) <= 3:
            return f"₹ {neg}{s}"
        last_three = s[-3:]
        remaining = s[:-3]
        parts = []
        while remaining:
            parts.append(remaining[-2:])
            remaining = remaining[:-2]
        parts.reverse()
        formatted = ",".join(parts) + "," + last_three
        return f"₹ {neg}{formatted}"
    except Exception as e:
        logger.warning(f"Error formatting currency: {e}")
        return f"₹ {amount}"

def format_indian_currency_short(amount):
    """
    Formats a numeric amount into a shorter Indian currency layout (e.g., 8250000 -> ₹ 82.50 Lakhs).
    """
    try:
        val = float(amount)
        neg = "-" if val < 0 else ""
        val = abs(val)
        if val >= 10000000: # 1 Crore = 10,000,000
            crores = val / 10000000
            return f"₹ {neg}{crores:.2f} Cr"
        elif val >= 100000: # 1 Lakh = 100,000
            lakhs = val / 100000
            return f"₹ {neg}{lakhs:.2f} Lakhs"
        else:
            return format_indian_currency(val)
    except Exception as e:
        logger.warning(f"Error formatting currency short: {e}")
        return f"₹ {amount}"

def download_dataset(url=None, dest_path="dataset/house_prices.csv"):
    """
    Downloads the Delhi MagicBricks housing dataset from a public raw URL if not already present.
    """
    if url is None:
        url = "https://raw.githubusercontent.com/AshwaniTiwari664/Delhi-House-Price-Prediction/master/MagicBricks.csv"
    
    if os.path.exists(dest_path):
        logger.info(f"Dataset already exists at {dest_path}")
        return dest_path
    
    # Create directory if it doesn't exist
    dest_dir = os.path.dirname(dest_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.info(f"Created directory {dest_dir}")
        
    logger.info(f"Downloading dataset from {url} to {dest_path}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        logger.info("Dataset downloaded successfully.")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise e
