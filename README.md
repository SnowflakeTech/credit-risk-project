# Credit Card Default Prediction Project

## Dự án Dự đoán Nợ Xấu Thẻ Tín Dụng

Dự án này xây dựng một hệ thống Machine Learning nhằm dự đoán khả năng **khách hàng thẻ tín dụng trở thành nợ xấu trong tháng tiếp theo** dựa trên dữ liệu lịch sử thanh toán, dư nợ, hạn mức tín dụng và thông tin nhân khẩu học.

Mục tiêu cuối cùng của dự án là hỗ trợ ngân hàng hoặc tổ chức tài chính:

* Giảm thiểu rủi ro tín dụng
* Phát hiện sớm khách hàng có nguy cơ nợ xấu
* Tối ưu quản lý hạn mức thẻ tín dụng
* Hỗ trợ nhắc nợ sớm
* Đưa ra quyết định dựa trên dữ liệu

---

## 1. Tổng quan dự án

* **Dataset:** UCI Credit Card Dataset
* **Số lượng dữ liệu:** Hơn 30,000 khách hàng
* **Bài toán:** Binary Classification
* **Target:**

  * `1`: Khách hàng có khả năng nợ xấu
  * `0`: Khách hàng không nợ xấu
* **Ngôn ngữ:** Python
* **Triển khai giao diện:** Streamlit Web App

---

## 2. Các giai đoạn thực hiện

### Phase 1: Exploratory Data Analysis

Trong giai đoạn EDA, dự án tập trung phân tích dữ liệu ban đầu để hiểu rõ đặc điểm khách hàng và hành vi thanh toán.

Các nội dung chính:

* Phân tích phân phối biến mục tiêu
* Phân tích thông tin nhân khẩu học
* Phân tích lịch sử thanh toán
* Trực quan hóa default rate theo:

  * Giới tính
  * Học vấn
  * Tình trạng hôn nhân
  * Độ tuổi
  * Hạn mức thẻ tín dụng

---

### Phase 2: Data Cleaning

Giai đoạn này xử lý dữ liệu thiếu, dữ liệu bất thường và chuẩn hóa các biến đầu vào.

Các công việc chính:

* Kiểm tra giá trị thiếu
* Xử lý giá trị bất thường
* Làm sạch các biến categorical:

  * `SEX`
  * `EDUCATION`
  * `MARRIAGE`
* Chuẩn hóa các biến số:

  * `BILL_AMT`
  * `PAY_AMT`
* Xuất dataset đã làm sạch:

```bash
data/UCI_Credit_Card_cleaned.csv
```

---

### Phase 3: Feature Engineering

Giai đoạn này tạo thêm các biến đặc trưng mới giúp mô hình dự đoán tốt hơn.

Các feature được tạo thêm:

* `MAX_DELAY`: Số tháng trễ hạn lớn nhất
* `PAY_AVG`: Trung bình trạng thái thanh toán
* `UTILIZATION`: Tỷ lệ sử dụng hạn mức thẻ
* `PAY_RATIO`: Tỷ lệ thanh toán so với dư nợ
* `RECENT_DELAY`: Trạng thái trễ hạn gần nhất
* `BILL_CHANGE`: Biến động dư nợ

Sau đó, các biến categorical được one-hot encoding để phù hợp với mô hình Machine Learning.

File đầu ra:

```bash
data/X_features.csv
data/y_target.csv
```

---

### Phase 4: Model Training

Dự án huấn luyện và so sánh nhiều mô hình Machine Learning khác nhau.

Các mô hình sử dụng:

* Logistic Regression
* Random Forest
* XGBoost

Do dữ liệu có hiện tượng mất cân bằng giữa nhóm nợ xấu và không nợ xấu, dự án sử dụng `class_weight='balanced'` hoặc các cơ chế tương đương để cải thiện khả năng nhận diện nhóm rủi ro cao.

Model và scaler được lưu tại thư mục:

```bash
models/
```

---

### Phase 5: Evaluation & Business Insight

Các mô hình được đánh giá bằng nhiều chỉ số khác nhau:

* Accuracy
* ROC-AUC
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve

Một số insight quan trọng:

* `PAY_0`, `MAX_DELAY`, `UTILIZATION` là những feature có ảnh hưởng mạnh đến khả năng nợ xấu.
* Khách hàng có hạn mức thấp nhưng tỷ lệ sử dụng thẻ cao thường có rủi ro cao hơn.
* Lịch sử trễ hạn gần đây là tín hiệu rất quan trọng trong dự đoán nợ xấu.
* Các biến demographic như giới tính, học vấn, hôn nhân có ảnh hưởng thấp hơn so với hành vi thanh toán.

---

### Phase 6: Explainability

Dự án sử dụng SHAP để giải thích kết quả dự đoán của mô hình.

Các nội dung giải thích bao gồm:

* Global Feature Importance
* SHAP Summary Plot
* Local Explanation cho từng khách hàng
* Force Plot để minh họa vì sao một khách hàng được dự đoán là rủi ro cao hoặc thấp

Explainability giúp mô hình trở nên minh bạch hơn, đặc biệt trong các bài toán tài chính có yêu cầu giải thích quyết định.

---

### Phase 7: Deployment / Streamlit Web App

Dự án được triển khai thành một ứng dụng web tương tác bằng Streamlit.

Các chức năng chính của web app:

* Chọn mẫu khách hàng
* Chọn mô hình dự đoán
* Hiển thị xác suất nợ xấu
* Hiển thị kết quả phân loại rủi ro
* Hiển thị SHAP explanation
* Hiển thị ROC Curve
* Hiển thị Confusion Matrix
* Giao diện tiếng Việt, trực quan, phù hợp trình bày portfolio

Chạy ứng dụng bằng lệnh:

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại:

```bash
http://localhost:8501
```

---

## 3. Folder Structure

```text
credit-risk-project/
│
├── data/
│   ├── UCI_Credit_Card.csv
│   ├── UCI_Credit_Card_cleaned.csv
│   ├── X_features.csv
│   └── y_target.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_DataCleaning.ipynb
│   ├── 03_FeatureEngineering.ipynb
│   ├── 04_ModelTraining.ipynb
│   ├── 05_Evaluation.ipynb
│   └── 06_Explainability.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_training_advanced.py
│   ├── model_evaluation.py
│   └── model_explainability.py
│
├── app.py
├── README.md
└── requirements.txt
```

---

## 4. Cài đặt môi trường

Clone project từ GitHub:

```bash
git clone https://github.com/your-username/credit-risk-project.git
cd credit-risk-project
```

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:

Trên Windows:

```bash
venv\Scripts\activate
```

Trên macOS/Linux:

```bash
source venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

---

## 5. Requirements

File `requirements.txt`:

```txt
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
shap>=0.42
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.20
joblib>=1.3
```

---

## 6. Hướng dẫn chạy project

### Bước 1: Data Cleaning & Feature Engineering

```bash
python src/feature_engineering.py
```

### Bước 2: Train models

```bash
python src/model_training_advanced.py
```

### Bước 3: Evaluate models

```bash
python src/model_evaluation.py
```

### Bước 4: Explainability

```bash
python src/model_explainability.py
```

### Bước 5: Chạy Streamlit Web App

```bash
streamlit run app.py
```

Mở trình duyệt tại:

```bash
http://localhost:8501
```

---

## 7. Kết quả mô hình

Các mô hình được đánh giá bằng ROC-AUC, Precision, Recall và F1-score.

Kết quả tổng quan:

* Logistic Regression: mô hình baseline, dễ giải thích
* Random Forest: hiệu quả tốt, có khả năng đánh giá feature importance
* XGBoost: hiệu quả cao, phù hợp với dữ liệu dạng bảng

Trong dự án này, Random Forest và XGBoost thường cho kết quả tốt hơn so với Logistic Regression.

---

## 8. Business Insight

Một số kết luận có giá trị nghiệp vụ:

* Khách hàng có `PAY_0 > 0` thường có rủi ro nợ xấu cao hơn.
* `MAX_DELAY` càng lớn thì khả năng nợ xấu càng cao.
* `UTILIZATION` cao cho thấy khách hàng đang sử dụng gần hết hạn mức, đây là dấu hiệu rủi ro.
* Khách hàng có `LIMIT_BAL` thấp thường dễ rơi vào nhóm rủi ro cao hơn.
* Các biến nhân khẩu học chỉ nên dùng để phân nhóm, không nên là yếu tố quyết định chính.
* Mô hình có thể hỗ trợ ngân hàng ưu tiên nhắc nợ, kiểm soát hạn mức và đánh giá rủi ro sớm.

---

## 9. Streamlit Web App

Ứng dụng Streamlit giúp người dùng tương tác trực tiếp với mô hình.

Các thành phần chính:

* Dashboard tổng quan
* Lựa chọn khách hàng
* Lựa chọn mô hình
* Dự đoán xác suất nợ xấu
* Phân loại mức rủi ro
* Biểu đồ ROC
* Confusion Matrix
* SHAP explanation

Ứng dụng phù hợp để trình bày trong:

* GitHub Portfolio
* CV Data Analyst / Data Scientist / AI Engineer Fresher
* Demo phỏng vấn
* Báo cáo môn học hoặc đồ án cá nhân

---

## 10. Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## 11. Project Highlights

Dự án này thể hiện đầy đủ quy trình của một bài toán Machine Learning thực tế:

* Phân tích dữ liệu
* Làm sạch dữ liệu
* Feature engineering
* Huấn luyện nhiều mô hình
* Đánh giá mô hình
* Giải thích mô hình bằng SHAP
* Triển khai thành web app bằng Streamlit
* Rút ra insight nghiệp vụ từ kết quả mô hình

---

## 12. Portfolio Ready

Dự án đã sẵn sàng để đưa lên GitHub và sử dụng trong portfolio.

Điểm nổi bật:

* Notebook có giải thích rõ ràng bằng Markdown
* Code được tách thành các file Python trong thư mục `src/`
* Có model đã train trong thư mục `models/`
* Có Streamlit app để demo trực quan
* Có SHAP plots giúp giải thích mô hình
* Có business insight phù hợp với bài toán tài chính
* Cấu trúc project rõ ràng, dễ trình bày với nhà tuyển dụng

---

## 13. Future Improvements

Một số hướng phát triển thêm:

* Thêm hyperparameter tuning cho Random Forest và XGBoost
* Thử thêm LightGBM hoặc CatBoost
* Thêm cross-validation
* Lưu kết quả prediction vào database
* Deploy app lên Streamlit Cloud
* Thêm Dockerfile để đóng gói project
* Tạo API bằng FastAPI để phục vụ prediction

---

## 14. Author

**Phạm Tiến Sơn**

Data Science / AI Engineer Fresher

---

## 15. License

This project is for educational and portfolio purposes.
