import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('dataset.csv')

# Encode the target variable (Danceability)
label_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Danceability_Encoded'] = label_encoder.fit_transform(df[['Danceability']])
y = df['Danceability_Encoded'].astype(int)  # Convert to int for classification

# Similarly encode Energy and Mood if using them
energy_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Energy_Numeric'] = energy_encoder.fit_transform(df[['Energy']])

mood_encoder = OrdinalEncoder()
df['Mood_Numeric'] = mood_encoder.fit_transform(df[['Mood']])

# Select features
X = df[['Tempo (BPM)', 'Energy_Numeric', 'Mood_Numeric', 'Genre']]

# Identify categorical and numerical columns
categorical_cols = ['Genre']
numerical_cols = ['Tempo (BPM)', 'Energy_Numeric', 'Mood_Numeric']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipeline with KNN classifier
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Find optimal k using cross-validation
param_grid = {'classifier__n_neighbors': range(1, 21)}
grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best model
best_k = grid_search.best_params_['classifier__n_neighbors']
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Best k: {best_k}')
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, 
                           target_names=label_encoder.categories_[0]))

# Create visualizations
plt.figure(figsize=(15, 10))

# Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=label_encoder.categories_[0],
           yticklabels=label_encoder.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# K vs Accuracy plot (from grid search)
plt.subplot(2, 2, 2)
k_values = param_grid['classifier__n_neighbors']
cv_results = grid_search.cv_results_
plt.plot(k_values, cv_results['mean_test_score'])
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('K Value vs Accuracy')
plt.axvline(x=best_k, color='r', linestyle='--')

# Feature importance visualization (based on energy and tempo)
plt.subplot(2, 2, 3)
# Extracting transformed features for visualization
preprocessed_X_test = best_model.named_steps['preprocessor'].transform(X_test)

# Only using Energy and Tempo for 2D scatter plot
# We need to know the column positions after transformation
numerical_features = preprocessed_X_test[:, :len(numerical_cols)]
tempo_idx, energy_idx = 0, 1  # Adjust if order is different

# Create scatter plot
scatter = plt.scatter(numerical_features[:, tempo_idx], 
                     numerical_features[:, energy_idx],
                     c=y_pred, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Predicted Danceability')
plt.xlabel('Tempo (Standardized)')
plt.ylabel('Energy (Standardized)')
plt.title('Classification Boundaries by Tempo and Energy')

# Decision boundary plot (simplified 2D)
plt.subplot(2, 2, 4)
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Use just two features for visualization
    simple_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', knn)
    ])
    # Use only numeric features for this plot
    simple_X = X_train[numerical_cols].iloc[:, :2]  # Tempo and Energy
    simple_pipeline.fit(simple_X, y_train)
    pred = simple_pipeline.predict(X_test[numerical_cols].iloc[:, :2])
    error = 1 - accuracy_score(y_test, pred)
    error_rates.append(error)

plt.plot(k_range, error_rates)
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K Value (Simplified 2D Model)')

plt.tight_layout()
plt.show()
