#!/data/data/com.termux/files/usr/bin/python

import os
import pandas as pd
from sklearn.datasets import make_classification

# Install scikit-learn
os.system("pkg install -y python python-dev clang")
os.system("pip install scikit-learn")

# Function to generate random data
def generate_random_data():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    df["Issue"] = [f"Issue_{i+1}" for i in range(len(df))]
    df["Resolution"] = [f"Resolution_{i+1}" for i in range(len(df))]
    return df

# Generate example data
general_ai_df = generate_random_data()
narrow_ai_df = generate_random_data()
theory_of_mind_df = generate_random_data()
self_awareness_df = generate_random_data()

# Save data to CSV
general_ai_df.to_csv("General-AI(AGI).csv", index=False)
narrow_ai_df.to_csv("Narrow-AI.csv", index=False)
theory_of_mind_df.to_csv("Theory-of.Mind.csv", index=False)
self_awareness_df.to_csv("Self-awareness.csv", index=False)

# Calculate LOV-e.Alhg module
lov_e_df = pd.DataFrame()
lov_e_df["General AI (AGI)"] = general_ai_df["Feature_1"]
lov_e_df["Narrow AI"] = narrow_ai_df["Feature_2"]
lov_e_df["Theory of Mind"] = theory_of_mind_df["Feature_3"]
lov_e_df["Tolerance"] = lov_e_df["Narrow AI"] / theory_of_mind_df["Feature_4"]
lov_e_df["Resistance"] = 1 - lov_e_df["Tolerance"]
lov_e_df["LOV-e.Alhg"] = 0.7 * lov_e_df["General AI (AGI)"] + 0.3 * lov_e_df["Resistance"]

# Save LOV-e.Alhg module to CSV
lov_e_df.to_csv("LOV-e.Alhg.csv", index=False)

print("Scikit-learn installed and data generated. CSV files created.")
