import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import PROCESSED_DATA_PATH, PROCESSED_TRAIN_CSV, PROCESSED_TEST_CSV

# Ensure the processed folder exists
os.makedirs(os.path.dirname(PROCESSED_TRAIN_CSV), exist_ok=True)

# Load the full processed dataset
df = pd.read_csv(PROCESSED_DATA_PATH)
print(f"Total documents: {len(df)}")

# Split into 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save the splits (always overwrite)
train_df.to_csv(PROCESSED_TRAIN_CSV, index=False)
test_df.to_csv(PROCESSED_TEST_CSV, index=False)

print(f"Train set: {len(train_df)} documents saved to {PROCESSED_TRAIN_CSV}")
print(f"Test set: {len(test_df)} documents saved to {PROCESSED_TEST_CSV}")
