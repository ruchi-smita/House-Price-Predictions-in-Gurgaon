🏠 House Price Prediction System
📌 Project Overview

The House Price Prediction System is a machine learning project that predicts the median house value based on different housing features such as income, location, population, and number of rooms.

The system uses data preprocessing pipelines and machine learning models to train on housing data and generate accurate price predictions.

This project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and prediction.

⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Joblib

Machine Learning Pipelines

📊 Dataset

The project uses a housing dataset containing multiple features related to housing and demographics.

Some key features include:

longitude

latitude

housing_median_age

total_rooms

total_bedrooms

population

households

median_income

ocean_proximity

The target variable is:

median_house_value

🧠 Machine Learning Workflow
1️⃣ Data Loading

The dataset is loaded using Pandas for analysis and preprocessing. 

main_old

2️⃣ Stratified Sampling

To maintain the distribution of income levels, the dataset is split using StratifiedShuffleSplit based on income categories. 

main_old

This ensures:

balanced training data

unbiased model evaluation

3️⃣ Data Preprocessing Pipeline

A ColumnTransformer pipeline is used to preprocess both numerical and categorical features.

Numerical Pipeline

Missing value handling using median imputation

Feature scaling using StandardScaler

Categorical Pipeline

Encoding categorical values using OneHotEncoder

These pipelines automate preprocessing during both training and prediction. 

main

4️⃣ Model Training

Multiple regression models were trained and evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

After evaluation, Random Forest Regressor was selected as the final model due to better performance. 

main

5️⃣ Model Evaluation

The models were evaluated using:

Root Mean Squared Error (RMSE)

and Cross Validation to ensure model stability.

6️⃣ Model Saving

The trained model and preprocessing pipeline are saved using Joblib for later use in prediction. 

main

Saved files:

model.pkl

pipeline.pkl

7️⃣ Prediction / Inference

When new input data is provided:

Data is loaded from input.csv

The preprocessing pipeline transforms the data

The trained model predicts house prices

Predictions are saved to output.csv 

main

🚀 Project Features

End-to-end ML workflow

Automated preprocessing using pipelines

Multiple model comparison

Cross-validation for performance evaluation

Model persistence using Joblib

Batch prediction system using CSV input

📂 Project Structure
project/
│
├── housing.csv
├── main.py
├── model.pkl
├── pipeline.pkl
├── input.csv
├── output.csv
🎯 Future Improvements

Deploy the model using Flask / FastAPI

Build a web interface for predictions

Add hyperparameter tuning

Use advanced models (XGBoost / Gradient Boosting)
