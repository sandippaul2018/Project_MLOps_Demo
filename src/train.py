import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def run_training():
    print('Loading data...')
    try:
        # Load data from root directory
        data = pd.read_csv('curing_data.csv')
    except FileNotFoundError:
        print('Error: curing_data.csv not found. Run generate_data.py first.')
        sys.exit(1)

    print('Setting up environment...')
    # Prepare data
    X = data.drop('quality_grade', axis=1)
    y = data['quality_grade']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    
    print('Training and comparing models...')
    # Define models
    models = {
        'lr': LogisticRegression(max_iter=1000, random_state=123),
        'rf': RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=123),
        'svm': SVC(kernel='rbf', random_state=123, probability=True)
    }
    
    results_list = []
    best_model = None
    best_accuracy = 0
    
    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if len(model.classes_) == 2:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            else:
                auc = 0.0
        except:
            auc = 0.0
        
        results_list.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
    
    print(f'Best Model Found: {best_model_name}')
    
    print('Saving model and metrics...')
    # Save model to root directory
    with open('bat_curing_pipeline.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save metrics for the dashboard
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('model_metrics.csv', index=False)
    
    print('Training Complete. Model and metrics saved.')

if __name__ == '__main__':
    run_training()
