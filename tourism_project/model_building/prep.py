### Data Preparation: Train/Test Split
'''
This script loads the validated dataset, removes non‑predictive columns,
splits the data into training and testing sets, and saves the resulting files
for use in the model‑training pipeline.
'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the registered dataset.
# Using a relative path ensures compatibility across Colab, GitHub Actions, and local machines.
DATA_PATH = "tourism_project/data/tourism.csv"

# Check that the dataset exists before loading it.
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# Load the dataset into a pandas DataFrame.
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")

# Remove the CustomerID column since it does not help with prediction.
df.drop(columns=["CustomerID"], inplace=True)

# Define the target variable for classification.
target = "ProdTaken"

# Separate features (X) from the target (y).
X = df.drop(columns=[target])
y = df[target]

# Split the dataset into training and testing sets.
# Stratification ensures the target class distribution remains consistent.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the resulting splits to CSV files.
# These files will be consumed by the training script in the next pipeline stage.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
