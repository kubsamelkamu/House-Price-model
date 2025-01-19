from data_Processing import load_data;

def test_load_data():
    train_data = load_data('./data/train.csv')
    print("Train data loaded successfully!")
    print(train_data.head())

if __name__ == "__main__":
    test_load_data()