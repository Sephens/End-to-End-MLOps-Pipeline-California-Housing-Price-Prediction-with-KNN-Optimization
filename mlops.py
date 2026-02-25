# 📦 Import libraries
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the California Housing dataset
# return_X_y=True returns the data as (features, target) instead of a Bunch object
# as_frame=True returns pandas DataFrames instead of numpy arrays for better data exploration
housing_data = fetch_california_housing(return_X_y=True, as_frame=True)

# Unpack the data into features (X) and target (y)
X, y = housing_data

# Display basic dataset information
print("\n=== Dataset Features (First 5 Rows) ===")
print(X.head())  # Show first 5 rows of features

print("\n=== Target Variable (First 5 Values) ===")
print(y.head())  # Show first 5 target values

print("\n=== Dataset Shape ===")
print(f"Features shape: {X.shape}")  # (20640, 8)
print(f"Target shape: {y.shape}")    # (20640,)

print("\n=== Feature Names ===")
print(X.columns.tolist())  # List all feature names

# Define numerical features - all columns in this dataset are numerical
numeric_features = X.columns.tolist()  # ['MedInc', 'HouseAge', 'AveRooms', ...]

# Create numerical preprocessing pipeline
# This handles two sequential operations:
# 1. Imputation: Fills missing values with the mean of each column
# 2. Scaling: Standardizes features to have mean=0 and variance=1
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardize feature scales
])

# Create ColumnTransformer to apply preprocessing
# This allows applying different transformations to different columns
# In this case, we apply the same numeric_transformer to all features
preprocessor = ColumnTransformer(
    transformers=[
        # ('name', transformer, columns)
        ('num', numeric_transformer, numeric_features)  # Apply to all numeric features
    ],
    remainder='drop'  # Explicitly drop any columns not specified
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline: preprocessing + KNN
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor())
])

# Define hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

# Apply GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("Best CV R² Score:", grid_search.best_score_)
print("Test R² Score:", r2)
print("Test MSE:", mse)
# print("Test RMSE:", rmse)

# Save the pipeline
with open('california_knn_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("📦 Final pipeline saved to 'california_knn_pipeline.pkl'")