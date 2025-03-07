import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset from the CSV file
df = pd.read_csv('dataset.csv')

# Print column names to see what's available in the dataset
print("Columns in dataset:", df.columns.tolist())

# Step 2: Manual encoding of danceability values
danceability_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Danceability'] = df['Danceability'].map(danceability_map)

# Step 3: Define the input (X) and target (y) data
X = df[['Tempo (BPM)']]  # Tempo is the feature (input)
y = df['Danceability']   # Danceability is the target (output)

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

# Step 9: Calculate accuracy using rounded predictions
y_pred_rounded = np.round(y_pred).astype(int)
y_pred_rounded = np.clip(y_pred_rounded, 0, 2)  # Ensure predictions stay within valid range
accuracy = (y_pred_rounded == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")

# Step 10: Visualize the results
plt.figure(figsize=(10, 6))

# Plot actual values
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')

# Plot predicted values
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')

# Create and plot the regression line (Fixed to avoid warning)
x_values = np.linspace(X['Tempo (BPM)'].min(), X['Tempo (BPM)'].max(), 100)
x_range = pd.DataFrame(x_values, columns=['Tempo (BPM)'])
y_pred_line = model.predict(x_range)
plt.plot(x_range, y_pred_line, color='green', linewidth=2, label='Regression Line')

# Add danceability level labels to the y-axis
plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])

# Add chart details
plt.xlabel('Tempo (BPM)')
plt.ylabel('Danceability')
plt.title('Tempo vs. Danceability Prediction Model')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()
