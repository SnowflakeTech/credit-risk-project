# file: model_training_advanced.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def plot_feature_importance(model, feature_names, model_name, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10,6))
        plt.title(f"{model_name} Feature Importance (Top {top_n})")
        plt.bar(range(top_n), importances[indices], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

def train_models_advanced(X_csv, y_csv, save_models=True):
    """
    Train Logistic Regression, Random Forest, XGBoost with feature importance and ROC curve
    """
    # ===== 1. Load dữ liệu =====
    print("=== Loading dataset ===")
    X = pd.read_csv(X_csv)
    y = pd.read_csv(y_csv).squeeze()  # convert to series
    
    # ===== 2. Train/test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ===== 3. Scaling cho Logistic Regression =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # ===== Logistic Regression =====
    print("=== Training Logistic Regression ===")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['LogisticRegression'] = lr
    
    # ===== Random Forest =====
    print("=== Training Random Forest ===")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    # ===== XGBoost =====
    print("=== Training XGBoost ===")
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # ===== 4. Evaluation =====
    print("\n=== Evaluation on Test Set ===")
    for name, model in models.items():
        if name == 'LogisticRegression':
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:,1]
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
        
        print(f"\n{name} Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
        print(classification_report(y_test, y_pred, digits=4))
        
        # Vẽ ROC curve
        plot_roc_curve(y_test, y_prob, name)
        
        # Vẽ feature importance cho tree-based
        if name in ['RandomForest', 'XGBoost']:
            plot_feature_importance(model, X.columns, name)
    
    # ===== 5. Lưu models =====
    if save_models:
        print("\n=== Saving models ===")
        model_folder = os.path.abspath("models")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        joblib.dump(models['LogisticRegression'], os.path.join(model_folder, 'logistic_regression.pkl'))
        joblib.dump(models['RandomForest'], os.path.join(model_folder, 'random_forest.pkl'))
        joblib.dump(models['XGBoost'], os.path.join(model_folder, 'xgboost.pkl'))
        joblib.dump(scaler, os.path.join(model_folder, 'scaler.pkl'))
        print(f"Models saved to folder '{model_folder}' successfully!")
    
    return models, X_test, y_test


# ===== Example usage =====
if __name__ == "__main__":
    train_models_advanced(
        X_csv="/home/kinas2k4/Documents/credit-risk-project/data/X_features.csv",
        y_csv="/home/kinas2k4/Documents/credit-risk-project/data/y_target.csv",
        save_models=True
    )