import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# ---------------------------------------------
# Page Config
# ---------------------------------------------
st.set_page_config(page_title="Encoder Tool", layout="wide")

st.title("🚀 Categorical Data Encoder Tool")
st.write("Apply Label Encoding & One-Hot Encoding easily!")

# ---------------------------------------------
# Sidebar Options
# ---------------------------------------------
st.sidebar.header("📌 Choose Input Method")

option = st.sidebar.radio(
    "Select Data Source:",
    ("Default Data", "Manual Input", "Upload CSV")
)

# ---------------------------------------------
# Default Data
# ---------------------------------------------
if option == "Default Data":
    data = ["Apple", "Banana", "Cherry", "Apple", "Cherry"]

# ---------------------------------------------
# Manual Input
# ---------------------------------------------
elif option == "Manual Input":
    user_input = st.text_input("Enter values (comma separated):", "Apple,Banana,Cherry")
    data = [x.strip() for x in user_input.split(",")]

# ---------------------------------------------
# File Upload
# ---------------------------------------------
else:
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("📄 Uploaded Data:")
        st.dataframe(df)

        column = st.selectbox("Select Column for Encoding", df.columns)
        data = df[column].dropna().astype(str).tolist()
    else:
        st.warning("Please upload a file.")
        data = []

# ---------------------------------------------
# Process Data
# ---------------------------------------------
if len(data) > 0:

    st.subheader("📊 Original Data")
    st.write(data)

    # Convert to numpy array
    data_array = np.array(data).reshape(-1, 1)

    # ---------------------------------------------
    # Label Encoding
    # ---------------------------------------------
    st.subheader("🔢 Label Encoding")

    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(data)

    df_label = pd.DataFrame({
        "Original": data,
        "Label Encoded": label_encoded
    })

    st.dataframe(df_label)

    # ---------------------------------------------
    # One-Hot Encoding (FIXED)
    # ---------------------------------------------
    st.subheader("🔥 One-Hot Encoding")

    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(data_array)

    feature_names = onehot_encoder.get_feature_names_out()

    df_onehot = pd.DataFrame(onehot_encoded, columns=feature_names)

    st.dataframe(df_onehot)

    # ---------------------------------------------
    # Visualization (MEDIUM SIZE FIXED)
    # ---------------------------------------------
    st.subheader("📈 Data Distribution")

    value_counts = pd.Series(data).value_counts()

    # ✅ Medium size chart (perfect for screenshot)
    fig, ax = plt.subplots(figsize=(6, 4))

    value_counts.plot(kind='bar', ax=ax)

    ax.set_title("Category Distribution", fontsize=12)
    ax.set_xlabel("Categories", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    plt.xticks(rotation=0)

    st.pyplot(fig)

else:
    st.info("👈 Please provide data to proceed.")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.write("💡 Built with ❤️ using Streamlit | By Sen")