# ---------------------------------------------
# ONE-HOT ENCODING IN MACHINE LEARNING
# ---------------------------------------------

# Import required libraries
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# ---------------------------------------------
# Step 1: Create Sample Data
# ---------------------------------------------
# We reshape because encoder expects 2D input
data = np.array(["Apple", "Banana", "Cherry", "Apple", "Cherry"]).reshape(-1, 1)

# ---------------------------------------------
# Step 2: Initialize OneHotEncoder
# ---------------------------------------------
# sparse=False → returns normal array instead of sparse matrix
encoder = OneHotEncoder(sparse=False)

# ---------------------------------------------
# Step 3: Fit and Transform Data
# ---------------------------------------------
# Fit → learn categories
# Transform → convert into one-hot encoded format
one_hot_encoded = encoder.fit_transform(data)

# ---------------------------------------------
# Step 4: Print Learned Categories
# ---------------------------------------------
print("Original Categories:")
print(encoder.categories_)

# ---------------------------------------------
# Step 5: Print One-Hot Encoded Output
# ---------------------------------------------
print("\nOne-Hot Encoded Data:")
print(one_hot_encoded)

# ---------------------------------------------
# Step 6: Show Data with Labels (Better Understanding)
# ---------------------------------------------
print("\nReadable Format:")

categories = encoder.categories_[0]

for i in range(len(data)):
    print(f"{data[i][0]} -> {one_hot_encoded[i]}")