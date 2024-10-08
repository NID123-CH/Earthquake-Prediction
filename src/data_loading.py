import pandas as pd
    
# Load the data
def load_data():
    
    train_values = pd.read_csv(r'C:/Users/Puneet Makkar/Desktop/Earthquake Prediction/data/01_raw/train_values.csv')
    train_labels = pd.read_csv(r'C:/Users/Puneet Makkar/Desktop/Earthquake Prediction/data/01_raw/train_labels.csv')
    test_values = pd.read_csv(r'C:/Users/Puneet Makkar/Desktop/Earthquake Prediction/data/01_raw/test_values.csv')
    
    return train_values, train_labels, test_values

train_values, train_labels, test_values = load_data()
  


{
    "code-runner.ignoreSelection": True
}