# Evaluate the model
from sklearn.metrics import classification_report, f1_score
# Evaluate model
def evaluate_model(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}") 
    
    
    
   