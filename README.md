# 🧪 Diabetes Patient Prediction using Machine Learning

Welcome to the **Diabetes Patient Prediction** project! This repository demonstrates how machine learning can be applied to predict whether an individual is likely to have diabetes based on diagnostic features. It utilizes the **PIMA Indians Diabetes dataset** and explores multiple classification models to find the best performing one.

---

## ✅ Objective

Early diagnosis of diabetes can significantly improve patient health outcomes. In this project, we build ML models that can classify whether a patient is diabetic (1) or not (0) based on various health metrics. The goal is to:

- Clean and preprocess real-world medical data
- Visualize patterns in the dataset
- Apply and evaluate various ML classification algorithms
- Build a model suitable for deployment or further development

---

## 📁 Project Structure

diabetes-prediction/
├── diabetes_pateint_prediction.ipynb
├── README.md
├── requirements.txt


---

## 📊 Dataset

- **Source**: [Kaggle – PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size**: 768 entries × 9 columns
- **Target Column**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

| Feature | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes genetic function |
| Age | Age in years |
| Outcome | Target class (0 or 1) |

---

## 🛠️ Technologies Used

| Category | Tools |
|---------|-------|
| Language | Python 3.10+ |
| Libraries | NumPy, Pandas, Matplotlib, Seaborn |
| ML Models | Scikit-learn (Logistic Regression, SVM, KNN, Decision Tree, Random Forest) |
| Evaluation | Confusion Matrix, Accuracy Score, Classification Report |

---

## 🧠 ML Pipeline

### 1. Data Cleaning & Preprocessing
- Handle zero entries as missing values (e.g., Glucose, Insulin)
- Impute missing values using mean strategy
- Normalize input features (MinMaxScaler)

### 2. Exploratory Data Analysis (EDA)
- Visualize distributions (histograms, countplots)
- Analyze class imbalance
- Generate correlation heatmaps

### 3. Model Training
- Train-Test split: 80/20
- Trained on 5 models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Random Forest Classifier

### 4. Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1)

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~77% |
| SVM | ~79% |
| KNN | ~75% |
| Decision Tree | ~74% |
| **Random Forest** | **~81%** ✅ |

**Random Forest** performed best with an accuracy of ~81%, making it the most suitable for further deployment.

---

## 💻 How to Run

1. **Clone the Repository**
   
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction

2. Create Virtual Environment (optional)
   
python -m venv venv
source venv/bin/activate 

3. Install Dependencies
   
pip install -r requirements.txt

4. Launch Jupyter Notebook
   
jupyter notebook diabetes_pateint_prediction.ipynb

---

⭐ Support & Contribute
If you found this project helpful, consider giving it a ⭐ on GitHub!
Feel free to fork, raise issues, or submit PRs to contribute.

---

