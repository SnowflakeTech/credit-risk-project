# file: model_explainability.py
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def explain_model(models_folder, X_csv, top_n=10):
    """
    Explain model predictions using SHAP
    models_folder: folder chứa model .pkl
    X_csv: file CSV chứa features
    top_n: số feature quan trọng hiển thị
    """
    # ===== Load dataset =====
    X = pd.read_csv(X_csv)
    
    # ===== Load model =====
    xgb_model = joblib.load(os.path.join(models_folder, 'xgboost.pkl'))
    
    print("=== Initializing SHAP Explainer ===")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    # ===== Global Feature Importance =====
    print("=== Global Feature Importance ===")
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n, show=True)
    
    # ===== Global SHAP Summary (beeswarm) =====
    print("=== SHAP Summary Plot (Beeswarm) ===")
    shap.summary_plot(shap_values, X, max_display=top_n, show=True)
    
    # ===== Local Explanation (sample) =====
    sample_index = 0  # bạn có thể thay đổi index
    print(f"=== Local Explanation for Sample Index {sample_index} ===")
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[sample_index,:], X.iloc[sample_index,:], matplotlib=True, show=True)
    
    print("=== SHAP analysis completed ===")

# ===== Example usage =====
if __name__ == "__main__":
    models_folder = "/home/kinas2k4/Documents/credit-risk-project/models"
    X_csv = "/home/kinas2k4/Documents/credit-risk-project/data/X_features.csv"
    
    explain_model(models_folder, X_csv, top_n=10)