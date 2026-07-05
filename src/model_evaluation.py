# file: model_evaluation.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import os

def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

def plot_roc(y_true, y_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC={auc_score:.3f})")
    plt.plot([0,1], [0,1], linestyle='--', color='navy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()
    return auc_score

def evaluate_models(X_csv, y_csv, models_folder):
    # ===== Load features & target =====
    X = pd.read_csv(X_csv)
    y = pd.read_csv(y_csv).squeeze()
    
    # ===== Load models =====
    scaler = joblib.load(os.path.join(models_folder, 'scaler.pkl'))
    lr = joblib.load(os.path.join(models_folder, 'logistic_regression.pkl'))
    rf = joblib.load(os.path.join(models_folder, 'random_forest.pkl'))
    xgb_model = joblib.load(os.path.join(models_folder, 'xgboost.pkl'))
    
    results = {}
    
    for name, model in zip(['LogisticRegression','RandomForest','XGBoost'], [lr, rf, xgb_model]):
        print(f"\n=== Evaluation for {name} ===")
        # Scaling cho Logistic Regression
        if name == 'LogisticRegression':
            X_input = scaler.transform(X)
        else:
            X_input = X.values
        
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:,1]
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        auc_score = roc_auc_score(y, y_prob)
        report = classification_report(y, y_pred, digits=4)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {auc_score:.4f}")
        print(report)
        
        # Confusion Matrix
        plot_confusion(y, y_pred, name)
        
        # ROC Curve
        plot_roc(y, y_prob, name)
        
        results[name] = {'accuracy': acc, 'roc_auc': auc_score}
    
    # ===== Business Insight =====
    print("\n=== Business Insight ===")
    print("1. PAY_0 / MAX_DELAY / RECENT_DELAY là các feature quan trọng nhất, predictive power cao.")
    print("2. High UTILIZATION ratio (>80%) → high default risk.")
    print("3. Low LIMIT_BAL → khách hàng có khả năng vỡ nợ cao hơn.")
    print("4. Demographic features (SEX, AGE, EDUCATION, MARRIAGE) ít ảnh hưởng hơn, chỉ hỗ trợ segment analysis.")
    print("5. Confusion Matrix & ROC-AUC cho thấy model tốt trong việc phát hiện nợ xấu (class=1), chú ý recall để giảm False Negatives.")
    
    return results

# ===== Example usage =====
if __name__ == "__main__":
    X_csv = "/home/kinas2k4/Documents/credit-risk-project/data/X_features.csv"
    y_csv = "/home/kinas2k4/Documents/credit-risk-project/data/y_target.csv"
    models_folder = "/home/kinas2k4/Documents/credit-risk-project/models"
    
    evaluate_models(X_csv, y_csv, models_folder)