import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': ['A', 'B', 'A', 'B', 'A', 'B'],
    'feature3': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'target': [0, 0, 0, 0, 1, 1]
})

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

categorical_features = ['feature2', 'feature3']
numerical_features = ['feature1']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE()),
    ('scaler', MinMaxScaler()),
    ('classifier', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(predictions)

# Assuming you have another DataFrame with similar columns
new_data = pd.DataFrame({
    'feature1': [7, 8, 9],
    'feature2': ['A', 'B', 'A'],
    'feature3': ['X', 'Y', 'X']
})

# Make predictions on the new data
new_predictions = pipeline.predict(new_data)

print(new_predictions)