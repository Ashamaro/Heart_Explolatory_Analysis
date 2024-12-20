# Heart_Explolatory_Analysis
Heart Disease Analysis and Prediction

Overview
This project aims to analyze the Heart Disease dataset and build a predictive model to determine the likelihood of heart disease based on various health and lifestyle factors. By utilizing data exploration, preprocessing, and machine learning, the project seeks to provide insights into the dataset and develop an accurate predictive system.

Features of the Project:
Data Exploration: Comprehensive analysis to understand the structure and distribution of the data.

Data Cleaning: Handling missing values, removing duplicates, and ensuring consistency.

Feature Engineering: Encoding categorical variables, scaling numerical features, and feature selection.

Model Training: Implementation of machine learning models to predict heart disease.

Evaluation: Performance evaluation using metrics like accuracy, precision, recall, and F1-score.
Dataset Information

The dataset used in this project contains the following attributes:

Age: Age of the individual.

Sex: Gender (1 = male; 0 = female).

Chest Pain Type: Types of chest pain experienced (e.g., typical angina, atypical angina).

Resting Blood Pressure: Resting blood pressure (in mm Hg).

Cholesterol: Serum cholesterol (in mg/dl).

Fasting Blood Sugar: Whether fasting blood sugar > 120 mg/dl (1 = true; 0 = false).

Resting ECG Results: Results of electrocardiographic measurements.

Maximum Heart Rate Achieved.

Exercise Induced Angina: Presence of angina induced by exercise (1 = yes; 0 = no).

Oldpeak: Depression induced by exercise relative to rest.

Slope of the Peak Exercise ST Segment.

Number of Major Vessels (0–3): Colored by fluoroscopy.

Thalassemia: A blood disorder type (e.g., normal, fixed defect, reversible defect).

Target Variable:

Heart Disease: Whether the individual is likely to have heart disease (1 = yes; 0 = no).

Steps Performed

1. Data Exploration

Univariate Analysis: Distribution and statistical summary of individual features.

Bivariate Analysis: Relationship between features and the target variable.

Correlation Analysis: Identifying key relationships between numerical features.

2. Data Cleaning

Handled missing values and inconsistencies.

Removed duplicate records.

Standardized data formats for numerical and categorical features.

3. Preprocessing
Feature Scaling: Applied normalization techniques to scale numerical features.

Categorical Encoding: Converted categorical features into numerical formats using one-hot encoding and label encoding.

Feature Selection: Selected features with the highest correlation and predictive potential.
TWO STEPS PERFORMED:
i.Feature Importance technique for numerical features
ii.Croos Tab Analysis and Chi_squared test for categorical features

4. Model Training

Implemented machine learning models including Logistic Regression, Decision Tree, Random Forest, and others.

Split data into training and testing sets.

5. Model Evaluation

Compared models based on evaluation metrics:

Accuracy

Precision

Recall

F1-Score

Chose the best-performing model for prediction.

Results

Best Model: [LogisticRegression Model].

Accuracy: Achieved an accuracy of [82.46%%].

Key Insights:
Chest pain types and maximum heart rate achieved are significant indicators of heart disease.

Cholesterol levels showed a weak correlation with heart disease.
fbs(Fasting Blood Sugar) had not affected target variable so it was dropped out.

Regular exercise reduces the likelihood of heart disease.

Usage

How to Run the Project

1.Clone this repository:

git clone <https://github.com/Ashamaro/Heart_Explolatory_Analysis/blob/main/Untitled.ipynb>
2.Navigate to the project directory:
  ```bash
cd heart-disease-analysis

3.Ensure the dataset is placed in the data/ folder or update the script with the dataset path.

4. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy
5.Run the analysis and prediction script:

python src/main.py
--Requirements

Python 3.8+

Libraries:

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

SciPy
--Future Enhancements

.Build a user-friendly Streamlit app for predictions.

.Experiment with deep learning models for improved accuracy.

.Integrate additional data sources for a more comprehensive analysis.

Repository Structure:

data/: Contains the dataset.

notebooks/: Jupyter notebooks for exploration and model development.

src/: Source code for preprocessing, training, and evaluation.

README.md: Project documentation.







