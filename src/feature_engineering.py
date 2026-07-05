# file: feature_engineering.py
import pandas as pd
import numpy as np

def print_phase3_steps():
    steps = [
        "1. Payment Behavior Features:",
        "   - MAX_DELAY: trễ hạn tối đa trong 6 tháng",
        "   - PAY_AVG: trung bình trễ hạn trong 6 tháng (đúng hạn = 0)",
        "2. Credit Utilization:",
        "   - BILL_AVG: trung bình dư nợ 6 tháng",
        "   - UTILIZATION: tỷ lệ sử dụng hạn mức = BILL_AVG / LIMIT_BAL",
        "3. Payment Ratio:",
        "   - PAY_AMT_AVG: trung bình tiền đã trả 6 tháng",
        "   - PAY_RATIO: PAY_AMT_AVG / BILL_AVG",
        "4. Recent behavior:",
        "   - RECENT_DELAY = PAY_0 (trễ hạn gần nhất)",
        "5. Change in debt:",
        "   - BILL_CHANGE = BILL_AMT1 - BILL_AMT6",
        "6. Encode categorical features:",
        "   - SEX, EDUCATION, MARRIAGE → one-hot encoding",
        "7. Select features for model:",
        "   - MAX_DELAY, PAY_AVG, UTILIZATION, PAY_RATIO, RECENT_DELAY, BILL_CHANGE + categorical dummies",
        "8. Save processed features and target to CSV."
    ]
    print("=== PHASE 3: Feature Engineering Steps ===")
    for s in steps:
        print(s)
    print("========================================")

def feature_engineering(input_csv, X_output_csv, y_output_csv):
    """
    Feature engineering cho dataset credit card.
    """
    print_phase3_steps()  # in ra quy trình
    
    df = pd.read_csv(input_csv)
    
    # ===== 1. Payment Behavior =====
    pay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    df["MAX_DELAY"] = df[pay_cols].max(axis=1)
    df["PAY_AVG"] = df[pay_cols].replace([-2,-1,0],0).mean(axis=1)
    
    # ===== 2. Credit Utilization =====
    bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    df["BILL_AVG"] = df[bill_cols].mean(axis=1)
    df["UTILIZATION"] = df["BILL_AVG"] / (df["LIMIT_BAL"] + 1)
    
    # ===== 3. Payment Ratio =====
    pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    df["PAY_AMT_AVG"] = df[pay_amt_cols].mean(axis=1)
    df["PAY_RATIO"] = df["PAY_AMT_AVG"] / (df["BILL_AVG"] + 1)
    
    # ===== 4. Recent behavior =====
    df["RECENT_DELAY"] = df["PAY_0"]
    
    # ===== 5. Change in debt =====
    df["BILL_CHANGE"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    
    # ===== 6. Encode categorical =====
    categorical_cols = ["SEX","EDUCATION","MARRIAGE"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # ===== 7. Chọn feature =====
    features = ["MAX_DELAY", "PAY_AVG", "UTILIZATION", "PAY_RATIO", "RECENT_DELAY", "BILL_CHANGE"] + \
               [c for c in df.columns if c.startswith("SEX_") or c.startswith("EDUCATION_") or c.startswith("MARRIAGE_")]
    
    X = df[features]
    y = df["default.payment.next.month"]
    
    # ===== 8. Lưu file =====
    X.to_csv(X_output_csv, index=False)
    y.to_csv(y_output_csv, index=False)
    
    print(f"Feature engineering completed. Features saved to {X_output_csv}, target saved to {y_output_csv}")


# ===== Example usage =====
if __name__ == "__main__":
    feature_engineering(
    input_csv="/home/kinas2k4/Documents/credit-risk-project/data/UCI_Credit_Card_cleaned.csv",
    X_output_csv="/home/kinas2k4/Documents/credit-risk-project/data/X_features.csv",
    y_output_csv="/home/kinas2k4/Documents/credit-risk-project/data/y_target.csv"
)