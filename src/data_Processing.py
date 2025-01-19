import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables, and scaling features.
    """
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Encode categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

if __name__ == "__main__":
    # Load the training data
    train_data = load_data('./data/train.csv')

    # Preprocess the training data
    processed_train_data = preprocess_data(train_data)

    # Save the preprocessed data to a new CSV file
    processed_train_data.to_csv('./data/processed_train.csv', index=False)

    print("Data preprocessing completed and saved to './data/processed_train.csv'")