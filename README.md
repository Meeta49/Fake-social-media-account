# Social Media Fake Account Detection using Machine Learning

## Project Overview

Fake accounts on social media platforms are widely used for spam, scams, misinformation, and malicious activities. Detecting such accounts is important for maintaining platform security and user trust.

This project builds a machine learning system that can automatically detect whether a social media account is **real or fake** using profile-related features. Multiple supervised learning algorithms were implemented and compared to evaluate their performance.

---

## Objectives

* Identify fake social media accounts using machine learning.
* Perform data preprocessing and exploratory analysis.
* Train and evaluate multiple classification models.
* Compare model performance and accuracy.
* Build a prediction system for new accounts.

---

## Dataset Description

The dataset contains **profile information of social media users**. Each record represents a user account with multiple features.

### Features

* **profile pic** – Whether the account has a profile picture (1 = Yes, 0 = No)
* **nums/length username** – Ratio of numbers in username
* **fullname words** – Number of words in full name
* **nums/length fullname** – Ratio of numbers in full name
* **name == username** – Whether the username matches the full name
* **description length** – Length of the profile description
* **external URL** – Presence of external URL
* **private** – Whether the account is private
* **#posts** – Number of posts
* **#followers** – Number of followers
* **#follows** – Number of accounts followed

### Target Variable

* **fake**

  * 0 → Real account
  * 1 → Fake account

The dataset is balanced with an equal number of real and fake accounts.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Google Colab

---

## Project Workflow

### 1. Data Loading

Training and testing datasets were loaded using Pandas and combined into a single dataset for processing.

### 2. Data Exploration

Initial exploration was performed to understand:

* Dataset structure
* Feature distributions
* Class balance
* Missing values

### 3. Data Preprocessing

* Combined training and testing datasets
* Checked and handled missing values
* Selected input features and target variable

### 4. Feature Selection

The dataset was divided into:

* **X** → Input features
* **y** → Target variable (fake account label)

### 5. Train-Test Split

The dataset was split into:

* **75% Training data**
* **25% Testing data**

This allows the model to learn patterns from training data and evaluate performance on unseen data.

---

## Machine Learning Models Used

### 1. Logistic Regression

Logistic Regression is a baseline classification model used for binary classification problems.

Advantages:

* Simple and interpretable
* Fast training time
* Works well for linearly separable data

---

### 2. Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

Advantages:

* High accuracy
* Handles nonlinear relationships
* Reduces overfitting

Results:

* **Accuracy: ~91.95%**

Confusion Matrix:

```
[[82  5]
 [ 9 78]]
```

---

### 3. XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is an advanced ensemble algorithm known for its high performance and efficiency.

Advantages:

* Handles large datasets
* Strong predictive performance
* Widely used in machine learning competitions

---

## Model Evaluation Metrics

The models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* AUC Score

These metrics help measure classification performance and prediction reliability.

---

## Prediction System

A prediction system was implemented where a new social media account’s features can be entered and the model predicts whether the account is **real or fake**.

Example output:

```
⚠️ This account is FAKE
```

or

```
✅ This account is REAL
```

---

## Model Saving and Loading

The trained model is saved using **Joblib** so it can be reused later without retraining.

Example:

```
from joblib import dump
dump(model, "fake_account_model.joblib")
```

The saved model can be loaded later for predictions.

---

## Visualization

Several visualizations were used to analyze model performance:

* Confusion Matrix
* ROC Curve
* Feature relationships
* Data distribution plots

These visualizations help in understanding model predictions and classification behavior.

---

## Results

Among the implemented models, **Random Forest and XGBoost showed better performance** compared to Logistic Regression due to their ability to capture complex patterns in the data.

The Random Forest model achieved approximately **92% accuracy**, making it a reliable model for fake account detection.

---

## Future Improvements

* Feature engineering to improve model accuracy
* Hyperparameter tuning
* Use deep learning models
* Deploy the model as a web application
* Integrate with real-time social media data

---

## Author

Meeta
B.Tech CSE (AI & ML)
Lovely Professional University
