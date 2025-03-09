import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv('dataset.csv')

# Step 2: Print column names to see what's available in the dataset
print("Columns in dataset:", df.columns.tolist())

# Step 3: Define the input (X) and target (y) data
# Use multiple features instead of just one
X = df[['Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction']]  # Multiple features
y = df['Glucose']  # Glucose is still the target

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the model - this doesn't change
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set - this doesn't change
y_pred = model.predict(X_test)

# Step 7: Calculate MSE and RMSE - this doesn't change
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Step 8: Print results and model coefficients
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Step 9: Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Glucose Levels')
plt.ylabel('Predicted Glucose Levels')
plt.title('Actual vs Predicted Glucose Levels')
plt.grid(True)


# Show the plots
plt.tight_layout()
plt.show()
