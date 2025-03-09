import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
# Replace 'diabetes.csv' with your actual filename
df = pd.read_csv('dataset.csv')

# Step 2: Print column names to see what's available in the dataset
print("Columns in dataset:", df.columns.tolist())

# Step 3: Define the input (X) and target (y) data
X = df[['Age']]  # Age is the feature (input)
y = df['Glucose']  # Glucose is the target (output)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Step 8: Print results and model coefficients
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Slope coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Step 9: Plot the actual data points
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

# Labels and title
plt.xlabel('Age')
plt.ylabel('Glucose Level')
plt.title('Age vs. Glucose Level')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
