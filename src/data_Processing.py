import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.median())

    # Encode categorical variables
    df = pd.get_dummies(df)

    # Feature scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled