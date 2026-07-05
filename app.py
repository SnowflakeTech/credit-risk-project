# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix

st.set_page_config(page_title="Dự án Dự đoán Nợ Xấu", layout="wide")
st.title("📊 Dự án Dự đoán Nợ Xấu Thẻ Tín Dụng")

# ===== Sidebar =====
st.sidebar.header("Cài đặt người dùng")
sample_index = st.sidebar.number_input("Chọn chỉ số khách hàng mẫu", min_value=0, max_value=29999, value=0)
selected_model = st.sidebar.selectbox("Chọn model dự đoán", ["Logistic Regression", "Random Forest", "XGBoost"])

# ===== Load dữ liệu =====
@st.cache_data
def load_data():
    X = pd.read_csv("data/X_features.csv")
    y = pd.read_csv("data/y_target.csv").squeeze()
    return X, y

@st.cache_resource
def load_models():
    lr = joblib.load("models/logistic_regression.pkl")
    rf = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return lr, rf, xgb_model, scaler

X, y = load_data()
lr, rf, xgb_model, scaler = load_models()

# ===== EDA: Distribution =====
st.header("1️⃣ Phân phối Dữ liệu và EDA")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(X['UTILIZATION'], bins=30, kde=True, ax=ax, color="skyblue")
ax.set_title("Phân phối Tỷ lệ Sử dụng Hạn mức (UTILIZATION)")
ax.set_xlabel("Tỷ lệ sử dụng hạn mức")
ax.set_ylabel("Số lượng khách hàng")
st.pyplot(fig)

# ===== Model Prediction =====
st.header("2️⃣ Dự đoán Rủi ro Khách hàng")
sample = X.iloc[[sample_index]]

if selected_model == "Logistic Regression":
    sample_scaled = scaler.transform(sample)
    pred = lr.predict(sample_scaled)[0]
elif selected_model == "Random Forest":
    pred = rf.predict(sample)[0]
else:
    pred = xgb_model.predict(sample)[0]

st.markdown(f"**Dự đoán rủi ro (0=Không nợ xấu, 1=Nợ xấu):** {pred}")

# ===== SHAP Feature Importance =====
st.header("3️⃣ Giải thích mô hình (SHAP Feature Importance)")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

fig2, ax2 = plt.subplots(figsize=(10,6))
shap.summary_plot(shap_values, X, plot_type="bar", max_display=10, show=False)
st.pyplot(fig2)

# ===== ROC Curve =====
st.header("4️⃣ ROC Curve (XGBoost)")
y_prob = xgb_model.predict_proba(X)[:,1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

fig3, ax3 = plt.subplots(figsize=(6,5))
ax3.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.3f}')
ax3.plot([0,1], [0,1], linestyle='--', color='navy')
ax3.set_xlabel("Tỷ lệ False Positive")
ax3.set_ylabel("Tỷ lệ True Positive")
ax3.set_title("ROC Curve")
ax3.legend()
st.pyplot(fig3)

# ===== Confusion Matrix =====
st.header("5️⃣ Confusion Matrix (XGBoost)")
y_pred = xgb_model.predict(X)
cm = confusion_matrix(y, y_pred)

fig4, ax4 = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_xlabel("Dự đoán")
ax4.set_ylabel("Thực tế")
ax4.set_title("Confusion Matrix")
st.pyplot(fig4)

# ===== Business Insight =====
st.header("6️⃣ Business Insight")
st.markdown("""
- **PAY_0, MAX_DELAY, RECENT_DELAY** là các feature quan trọng nhất, ảnh hưởng trực tiếp đến khả năng nợ xấu.
- **Tỷ lệ sử dụng hạn mức (UTILIZATION)** cao → rủi ro nợ xấu cao.
- **Hạn mức tín dụng thấp (LIMIT_BAL)** → khả năng khách nợ xấu tăng.
- Các thông tin demographic (SEX, EDUCATION, MARRIAGE) ít ảnh hưởng, chủ yếu để phân nhóm khách hàng.
""")