import pandas as pd
from scipy.stats import gaussian_kde

# Original DataFrame
df_train = pd.DataFrame({
    'feature1': ['A', 'A', 'B', 'B', 'C', 'D'],
    'feature2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'feature3': [1, 2, 3, 4, 5, 6],
    'target': [0, 0, 0, 0, 1, 1]
})

# New DataFrame with the combination of values you want to find the probability for
df_test = pd.DataFrame({
    'feature1': ['A'],
    'feature2': ['X'],
    'feature3': [2],
})

# Initialize a dictionary to hold KDE estimators for each feature
kde_estimators = {}

# Calculate KDE for each feature in df_train
for column in df_train.columns:
    # Skip non-numeric columns
    if df_train[column].dtype != 'object':
        kde_estimators[column] = gaussian_kde(df_train[column])

# Calculate the probability density for each feature in df_test
probabilities = {}
for column in df_test.columns:
    # Skip non-numeric columns
    if column in kde_estimators:
        probabilities[column] = kde_estimators[column].evaluate(df_test[column])

# Combine probabilities from all features
overall_probability = 1.0
for column in probabilities:
    overall_probability *= probabilities[column][0]  # Assuming only one row in df_test

print("Overall Probability:", overall_probability)