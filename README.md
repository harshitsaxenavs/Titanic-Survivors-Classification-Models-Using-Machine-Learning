# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project predicts the **survival of passengers** on the Titanic using supervised machine learning **classification models**.

The model is trained on the famous Kaggle **Titanic dataset**, containing demographic and travel details such as passenger class, gender, age, fare, siblings/spouses aboard, parents/children aboard, and embarkation port.

This project includes a complete end-to-end ML workflow:
**Data Cleaning → Feature Engineering → Preprocessing → Model Training → Evaluation**

## 📂 Dataset Overview
The dataset contains passenger records with the following attributes:

| Feature | Description | Type |
| :--- | :--- | :--- |
| **survived** | Survival status (0 = No, 1 = Yes) | **Target** |
| **pclass** | Ticket class (1st, 2nd, 3rd) | Categorical (Ordinal) |
| **sex** | Sex of the passenger | Categorical |
| **age** | Age in years | Numerical |
| **sibsp** | # of siblings/spouses aboard | Numerical |
| **parch** | # of parents/children aboard | Numerical |
| **fare** | Ticket fare | Numerical |
| **embarked** | Port of Embarkation (C, Q, S) | Categorical |
| **alone** | Whether passenger was traveling alone | Boolean |

## 🛠️ Tech Stack
* **Python**
* **Pandas, NumPy** – Data processing
* **Matplotlib, Seaborn** – Data visualization
* **Scikit-Learn** – Preprocessing & ML models
* **Jupyter Notebook**

## 📊 Key Workflow Steps

### 1. Data Cleaning & EDA
* Checked missing values and duplicates.
* Dropped redundant columns: `deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`.
* Filled missing `age` values using mean.
* Dropped rows with missing `embarked` values.
* Visualized correlations between numerical features and target.
* **Key Insight:**  
  👉 Women and passengers in 1st class had the highest survival rate.

### 2. Feature Engineering
* **Label Encoding** applied to:
  - `sex`
  - `embarked`
* Created “alone” feature based on `sibsp` + `parch`.
* Converted categorical values into ML-friendly numeric format.

### 3. Data Preprocessing
* **Scaling:** Applied `StandardScaler` on numerical features (`age`, `fare`, `sibsp`, `parch`).
* **Train-Test Split:** 80% training and 20% testing.

### 4. Model Building & Evaluation
Multiple models were trained and evaluated.  
Performance was measured using **Accuracy**, **Confusion Matrix**, and **Classification Report**.

### ✅ Model Accuracy Comparison

| Model | Accuracy |
| :--- | :--- |
| **Support Vector Machine (SVM)** | **81.46%** 🥇 |
| **Logistic Regression** | 80.34% |
| **Decision Tree Classifier** | 80.34% |
| **K-Nearest Neighbors (KNN)** | 79.21% |
| **Gaussian Naive Bayes** | 77.52% |

### 🔍 Detailed Performance Insights

#### ✔ Support Vector Machine (Best Model)
* Highest accuracy (81.46%).
* Performs well on linear + non-linear boundaries.
* Stable performance without overfitting.

#### ✔ Logistic Regression
* Very close performance to SVM.
* Best for linearly separable patterns.
* Fast and interpretable.

#### ✔ Decision Tree
* Simple and easy to interpret.
* Slight overfitting observed.
* Accuracy tied with Logistic Regression.

#### ✔ KNN
* Sensitive to scaling and choice of K.
* Performs decently but slower on large datasets.

#### ✔ Naive Bayes
* Assumes independence between features.
* Fastest but least accurate due to independence violations.

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/harshitsaxenavs/Titanic-Survival-Prediction.git

# Navigate to the directory
cd Titanic-Survival-Prediction

# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn

# Run the notebook
jupyter notebook Titanic.ipynb
```

## 📈 Future Improvements

* Hyperparameter tuning (GridSearchCV).
* Extract titles from passenger names (Mr, Mrs, Miss, etc.).
* Build ensemble models like **Random Forest**, **Gradient Boosting**, **XGBoost**.
* Deploy the model using **Streamlit** or **Flask**.

## 👨‍💻 Author  
**Harshit Saxena**  
Machine Learning & AI Enthusiast  
📧 harshitsaxenavs@gmail.com  
🔗 GitHub: https://github.com/harshitsaxenavs

## ⚠️ License
This project is for educational purposes.
