# Preprocess the data
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def preprocess_data(train_values, train_labels, test_values):
    # Merge train values and labels
    train_data = pd.merge(train_values, train_labels, on='building_id')
    
    # Identify numeric and categorical features
    # You may need to adjust these based on your specific dataset
    numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['building_id', 'damage_grade']]
    categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Prepare the data
    X = train_data.drop(['building_id', 'damage_grade'], axis=1)
    y = train_data['damage_grade']
    X_test = test_values.drop('building_id', axis=1)

    # Adjust labels to start from 0
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split the training data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, preprocessor