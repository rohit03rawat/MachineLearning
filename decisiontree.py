import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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

# Similarly encode Energy and Mood
energy_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Energy_Numeric'] = energy_encoder.fit_transform(df[['Energy']])

mood_encoder = OrdinalEncoder()
df['Mood_Numeric'] = mood_encoder.fit_transform(df[['Mood']])

# Select features
X = df[['Tempo (BPM)', 'Energy_Numeric', 'Mood_Numeric', 'Genre']]

# Identify categorical and numerical columns
categorical_cols = ['Genre']
numerical_cols = ['Tempo (BPM)', 'Energy_Numeric', 'Mood_Numeric']

# Create preprocessor WITHOUT standardization to keep original scales
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),  # Keep original values
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipeline with Decision Tree classifier
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter tuning
param_grid = {
    'classifier__max_depth': [None, 3, 5, 7, 10],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(f'Best parameters: {best_params}')

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, 
                           target_names=label_encoder.categories_[0]))

# Determine feature names post-transformation
# Extract the one-hot encoded feature names
ohe = best_model.named_steps['preprocessor'].transformers_[1][1]
genre_features = ohe.get_feature_names_out(['Genre'])
feature_names = numerical_cols + list(genre_features)

# Create visualizations
plt.figure(figsize=(20, 15))

# Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=label_encoder.categories_[0],
           yticklabels=label_encoder.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Feature Importance
plt.subplot(2, 2, 2)
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')

# Decision Tree Visualization
plt.subplot(2, 2, 3)
# Get the decision tree from the pipeline
tree = best_model.named_steps['classifier']
# Plot a simplified version of the tree
plot_tree(tree, 
          max_depth=3,  # Limit depth for visibility
          feature_names=feature_names,
          class_names=label_encoder.categories_[0],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree (Simplified)')

# Tempo vs Danceability visualization with original values
plt.subplot(2, 2, 4)
tempo_danceability = pd.DataFrame({
    'Tempo': df['Tempo (BPM)'],
    'Danceability': y,
    'Genre': df['Genre']
})
sns.scatterplot(data=tempo_danceability, x='Tempo', y='Danceability', 
                hue='Genre', style='Genre', s=100, alpha=0.7)
plt.xlabel('Tempo (BPM)')
plt.ylabel('Danceability Level')
plt.yticks([0, 1, 2], label_encoder.categories_[0])
plt.title('Tempo vs Danceability by Genre')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Plot a larger, separate decision tree for better visibility
plt.figure(figsize=(25, 15))
plot_tree(tree, 
          feature_names=feature_names,
          class_names=label_encoder.categories_[0],
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title('Complete Decision Tree')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')  # Save high-resolution image
plt.show()
