import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from data_loading import load_data
from preprocess import preprocess_data
from evaluate import evaluate_model
from train import train_xgboost
from predict import make_predictions





def main():
    # Load data
    train_values, train_labels, test_values = load_data()
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, preprocessor = preprocess_data(train_values, train_labels, test_values)
    
    # Train model
    model = train_xgboost(X_train, y_train, X_val, y_val, preprocessor)
    
    # Make predictions on validation set
    val_predictions = make_predictions(model, X_val)
    
    # Evaluate model
    print("Validation Set Evaluation:")
    evaluate_model(y_val, val_predictions)
    
    # Make predictions on test data
    test_predictions = make_predictions(model, X_test)
    
    # You can save or further process the test predictions here

    submission_df = pd.DataFrame({'building_id': test_values['building_id'], 'damage_grade': test_predictions})
    submission_df.to_csv('submission.csv', index=False)
    
if __name__ == "__main__":
    main()