import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('dataset.csv')

# Convert categorical Danceability to numerical
# We'll use an ordinal encoding: Low=0, Medium=1, High=2
danceability_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Danceability_Numeric'] = danceability_encoder.fit_transform(df[['Danceability']])

# Similarly encode Energy
energy_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Energy_Numeric'] = energy_encoder.fit_transform(df[['Energy']])

# Select features and target
X = df[['Tempo (BPM)', 'Energy_Numeric', 'Genre']]
y = df['Danceability_Numeric']

# Identify categorical columns
categorical_cols = ['Genre']
numerical_cols = ['Tempo (BPM)', 'Energy_Numeric']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'R² Score: {r2:.4f}')

# Create visualizations
plt.figure(figsize=(15, 10))

# Actual vs Predicted values
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Danceability')
plt.ylabel('Predicted Danceability')
plt.title('Actual vs Predicted Danceability')

# Residual plot
plt.subplot(2, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.xlabel('Predicted Danceability')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Tempo vs Danceability
plt.subplot(2, 2, 3)
plt.scatter(X_test['Tempo (BPM)'], y_test, alpha=0.5, label='Actual')
plt.scatter(X_test['Tempo (BPM)'], y_pred, alpha=0.5, label='Predicted')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Danceability')
plt.title('Tempo vs Danceability')
plt.legend()

# Energy vs Danceability
plt.subplot(2, 2, 4)
plt.scatter(X_test['Energy_Numeric'], y_test, alpha=0.5, label='Actual')
plt.scatter(X_test['Energy_Numeric'], y_pred, alpha=0.5, label='Predicted')
plt.xlabel('Energy (Numeric)')
plt.ylabel('Danceability')
plt.title('Energy vs Danceability')
plt.legend()

plt.tight_layout()
plt.show()
