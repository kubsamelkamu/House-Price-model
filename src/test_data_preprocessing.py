from  data_Processing import load_data,preprocess_data;

def test_preprocess_data():
    train_data = load_data('./data/train.csv')
    processed_train_data = preprocess_data(train_data)
    print("Train data preprocessed successfully!")
    print(processed_train_data.head())

if __name__ == "__main__":
    test_preprocess_data()