# ---------------------------------------------
# LABEL ENCODING IN MACHINE LEARNING
# ---------------------------------------------

# Import required library
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------
# Step 1: Define Custom Order
# ---------------------------------------------
# Note:
# LabelEncoder sorts values alphabetically internally.
# So actual encoding will be:
# High -> 0
# Low  -> 1
# Medium -> 2

order = ["Low", "Medium", "High"]

# ---------------------------------------------
# Step 2: Initialize LabelEncoder
# ---------------------------------------------
encoder = LabelEncoder()

# ---------------------------------------------
# Step 3: Fit Encoder with Categories
# ---------------------------------------------
# This learns the unique classes
encoder.fit(order)

# ---------------------------------------------
# Step 4: Sample Data
# ---------------------------------------------
data = ["Low", "Medium", "High", "Low",
        "Low", "Medium", "High", "Low"]

# ---------------------------------------------
# Step 5: Transform Data (Convert to Numbers)
# ---------------------------------------------
encoded_data = encoder.transform(data)

# ---------------------------------------------
# Step 6: Output Results
# ---------------------------------------------
print("Original Data:")
print(data)

print("\nEncoded Data:")
print(encoded_data)

# ---------------------------------------------
# Step 7: Show Mapping
# ---------------------------------------------
print("\nLabel Mapping:")
for label, value in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"{label} -> {value}")