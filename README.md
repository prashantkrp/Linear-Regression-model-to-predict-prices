Linear Regression Model to Predict Prices:
This project implements a Linear Regression model using Python to predict house prices based on features such as square footage, number of bedrooms, and number of bathrooms.

Project Structure:
The project includes steps for loading and preprocessing the data, training the Linear Regression model, evaluating its performance, and making predictions on new data points.

Libraries Used:
numpy
pandas
scikit-learn
Getting Started
Prerequisites

Ensure you have the following libraries installed:

bash code:
pip install numpy pandas scikit-learn

Project Files:
train.csv: The dataset containing training data.
test.csv: The dataset containing test data (optional).
data_description.txt: A file describing the dataset (optional).
sample_submission.csv: A sample submission file (optional).
Steps to Run the Project
Load the Dataset
The script loads the house price data from a CSV file.

python code:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
train_data = pd.read_csv(r'C:\Users\Prashant kumar\PycharmProjects\pythonProject5\train.csv')

# Display the first few rows of the dataset
print("Train Data Head:")
print(train_data.head())
Check and Handle Missing Values
Check for missing values and drop rows with missing values if any.

python code:
# Check for missing values
print("Missing Values in Train Data:")
print(train_data.isnull().sum())

# Drop rows with missing values (if any)
train_data.dropna(inplace=True)
Define Features and Target Variable
Define the features (X) and the target variable (y).

python code:
# Define the features (X) and the target variable (y)
X = train_data[['square_footage', 'bedrooms', 'bathrooms']]
y = train_data['price']
Split the Data
Split the data into training and testing sets.

python code:
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Create and Train the Model
Create the Linear Regression model and train it.

python code:
# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
Make Predictions and Evaluate the Model
Make predictions on the test set and evaluate the model's performance.

python code:
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
Predict on New Data
Make predictions on new data points.

python code:
# Example new data point
new_data = pd.DataFrame({
    'square_footage': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})

# Predict the price
predicted_price = model.predict(new_data)
print(f"Predicted price: {predicted_price[0]}")
Optional: Load and Display Additional Files
Load and display data_description.txt, sample_submission.csv, and test.csv if they exist.

python code:
# Load and display data_description.txt (if it exists)
try:
    with open(r'C:\Users\Prashant kumar\PycharmProjects\pythonProject5\data_description.txt', 'r') as file:
        data_description = file.read()
        print("Data Description:")
        print(data_description)
except FileNotFoundError:
    print("data_description.txt not found.")

# Load and display sample_submission.csv (if it exists)
try:
    sample_submission = pd.read_csv(r'C:\Users\Prashant kumar\PycharmProjects\pythonProject5\sample_submission.csv')
    print("Sample Submission Head:")
    print(sample_submission.head())
except FileNotFoundError:
    print("sample_submission.csv not found.")

# Load and display test.csv (if it exists)
try:
    test_data = pd.read_csv(r'C:\Users\Prashant kumar\PycharmProjects\pythonProject5\test.csv')
    print("Test Data Head:")
    print(test_data.head())
except FileNotFoundError:
    print("test.csv not found.")
    
How It Works:
Data Loading and Preprocessing: The dataset is loaded, missing values are handled, and features and target variables are defined.
Data Splitting: The data is split into training and testing sets.
Model Training: A Linear Regression model is created and trained using the training data.
Model Evaluation: The model's performance is evaluated using Mean Squared Error and R-squared score.
Prediction: The trained model is used to predict house prices for new data points.

Acknowledgements:
pandas: An open-source data analysis and manipulation tool.
NumPy: A fundamental package for scientific computing with Python.
scikit-learn: A machine learning library for Python.
