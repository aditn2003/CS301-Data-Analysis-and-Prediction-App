# CS301 Data Analysis and Prediction App

## 📌 Overview

This project was developed for my **CS301: Data Science** class. The goal was to analyze healthcare insurance data and build predictive models for medical insurance charges. Over four milestones, we progressed from exploratory data analysis (EDA) to regression modeling, ensemble learning, and finally deploying an interactive web application using **Dash**.

The application allows users to upload datasets, explore correlations and averages, train regression models and make predictions on new data.

## 🔗 **Live Demo:** [Data Analysis App on Render](https://dash-app2.onrender.com/)

## 📂 Dataset

- **Source:** [Kaggle – Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)
- **Size:** 1,338 rows × 7 columns
- **Features:**
  - `age` – Age of the insured individual
  - `sex` – Gender
  - `bmi` – Body Mass Index
  - `children` – Number of dependents
  - `smoker` – Smoking status
  - `region` – Geographical location
  - `charges` – Medical insurance cost (target variable)

---

## 🚀 Milestones

### 🔹 Milestone 1 – Dataset Exploration

- Explored dataset structure and features.
- Verified data quality (no missing values, outliers checked).
- Identified key features affecting insurance costs (age, BMI, smoking status, region, gender).
- Established application domain: predicting healthcare insurance charges for fairer premium setting.

---

### 🔹 Milestone 2 – Multiple Regression Model

- Built a **Multiple Linear Regression** model using one-hot encoding for categorical variables.
- Split dataset: 80% training, 20% testing.
- **Performance:**
  - R² = **0.758**
- Feature impact: Smoking, BMI, and Age were strongest predictors of insurance charges.

---

### 🔹 Milestone 3 – Model Comparison & Ensembles

- Tested base models: Linear Regression, KNN, Decision Tree.
- Introduced **Bagging** and **Stacking** for improved performance.
- **Best Model:** Random Forest with Bagging.
  - R² = **0.8792** (highest accuracy)
  - Strong ability to capture non-linear relationships and reduce variance.

---

### 🔹 Milestone 4 – Deployment & App

- Built an interactive **Dash web app** (`app1.py`).
- Key Features:
  - Upload CSV datasets.
  - Explore average values and correlation charts.
  - Select target variable & features for training.
  - Train regression model (with preprocessing pipelines).
  - Predict target values from user input.
- **Deployment:** Hosted on Render.

---

## 🛠️ Technologies Used

- **Python**
- **Dash (Plotly)** – Web application framework
- **Pandas, NumPy** – Data preprocessing
- **Scikit-learn** – Regression models, pipelines, preprocessing
- **Plotly Express** – Interactive visualizations
- **Render** – Deployment

---

## 📊 Results

- Regression R² scores improved across milestones (0.758 → 0.8792).

- Random Forest with Bagging gave the best performance.

- Interactive app enables hands-on exploration of healthcare datasets and predictions.
