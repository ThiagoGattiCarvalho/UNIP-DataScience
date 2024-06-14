import pandas as pd
from scipy.stats import multivariate_hypergeom
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Original DataFrame
df_train = pd.DataFrame({
    'feature1': ['A', 'A', 'B', 'B', 'C', 'D'],
    'feature2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'feature3': [1, 2, 3, 4, 5, 6],
    'target': [0, 0, 0, 0, 1, 1]
})

# New DataFrame with the combination of values you want to find the joint probability for
df_test = pd.DataFrame({
    'feature1': ['A', 'B'],
    'feature2': ['X', 'Y'],
    'feature3': [2, 3],
})

# Convert rows of categorical columns in the training dataset to tuples and count occurrences
train_tuples = [tuple(row) for row in df_train[df_train.select_dtypes(include='object').columns].values]
train_tuple_counts = Counter(train_tuples)

# Total number of tuples in the training dataset
total_train_tuples = len(train_tuples)

# Calculate probabilities of each tuple occurring in the training dataset
train_tuple_probabilities = {tuple_: count / total_train_tuples for tuple_, count in train_tuple_counts.items()}

# Convert rows of categorical columns in the test dataset to tuples
test_tuples = [tuple(row) for row in df_test[df_test.select_dtypes(include='object').columns].values]

# Count occurrences of test tuples in the training dataset
test_counts_in_train = {tuple_: train_tuple_probabilities.get(tuple_, 0) for tuple_ in test_tuples}

print("Probabilities of test tuples happening in the training dataset:")
print(test_counts_in_train)