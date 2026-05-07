from src.features.feature_engineering import create_features
from src.data.load_data import load_data

# Load data
df = load_data("data/sales.xlsx")

# Create features
df_features = create_features(df)

print(df_features.head())