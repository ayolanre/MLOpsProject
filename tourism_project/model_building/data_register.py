### Register and Validate the Raw Dataset
'''
This script loads the raw tourism dataset, validates that all required columns exist,
prints useful summary information, and re-saves the dataset to ensure consistency
for downstream pipeline steps.
'''

import pandas as pd
import os

# Path to the raw dataset inside the project structure.
# Using a relative path ensures compatibility across Colab, GitHub Actions, and local machines.
RAW_PATH = "tourism_project/data/tourism.csv"

# Check that the dataset exists before attempting to load it.
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at: {RAW_PATH}")

# Load the dataset into a pandas DataFrame.
df = pd.read_csv(RAW_PATH)

# Define the columns that must be present for the pipeline to work correctly.
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Identify any missing columns by comparing the dataset to the expected list.
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Print dataset summary information for quick inspection.
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())

# Save the validated dataset back to disk.
# This ensures downstream steps always operate on a clean, verified dataset.
df.to_csv(RAW_PATH, index=False)
print(f"Dataset saved and ready for pipeline at: {RAW_PATH}")
