import pandas as pd

def preprocess(df):
    
    print("\n🔄 Starting Preprocessing...\n")

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Sort data
    df = df.sort_values(['state', 'date'])

    # ✅ FIXED HERE
    df = df.ffill()

    # Reset index
    df = df.reset_index(drop=True)

    # Save cleaned data
    df.to_csv("data/processed/cleaned_data.csv", index=False)

    print("✅ Preprocessing Completed!")
    print("📁 Cleaned data saved!\n")

    print(df.head())

    return df