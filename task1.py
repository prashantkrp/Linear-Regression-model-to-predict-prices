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

# Check for missing values
print("Missing Values in Train Data:")
print(train_data.isnull().sum())

# Drop rows with missing values (if any)
train_data.dropna(inplace=True)

# Define the features (X) and the target variable (y)
X = train_data[['square_footage', 'bedrooms', 'bathrooms']]
y = train_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example new data point
new_data = pd.DataFrame({
    'square_footage': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})

# Predict the price
predicted_price = model.predict(new_data)
print(f"Predicted price: {predicted_price[0]}")

# Load and display data_description.txt (if it exists)
try:
    with open(r'C:\Users\Prashant kumar\PycharmProjects\pythonProject5\data_description.txt', 'r') as file:
        # Your code to read the file contents
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
