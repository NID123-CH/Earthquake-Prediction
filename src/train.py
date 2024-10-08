# Train the model
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

def train_xgboost(X_train, y_train, X_val, y_val, preprocessor):
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    return model

