

# Weather Data Analysis & Prediction

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

## 2. Load Dataset
df = pd.read_csv('weather.csv')
df.head()

## 3. Data Cleaning
df = df.dropna()

## 4. Feature Selection
X = df[['MinTemp', 'MaxTemp', 'Humidity3pm']]
y = df['Temp3pm']

## 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

## 7. Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

## 8. Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted")
plt.show()
