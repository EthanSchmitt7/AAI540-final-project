import joblib
import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
TARGET_PARAM = "pm25"

base_dir = "/opt/ml/processing"

df_location = pd.read_csv(f"{base_dir}/input/sensor_data.csv")

# Split training
df_param = df_location[df_location["parameter"] == TARGET_PARAM]  # Filter data for this parameter
train_data = df_param.iloc[: int(len(df_param) * TRAIN_SPLIT)]
train_data = train_data.reset_index(drop=True)

# Split validation
val_data = df_param.iloc[
    int(len(df_param) * TRAIN_SPLIT) : int(len(df_param) * TRAIN_SPLIT)
    + int(len(df_param) * VAL_SPLIT)
]
val_data = val_data.reset_index(drop=True)

# Split testing
test_data = df_param.iloc[
    int(len(df_param) * TRAIN_SPLIT) + int(len(df_param) * VAL_SPLIT) : int(
        len(df_param) * TRAIN_SPLIT
    )
    + int(len(df_param) * VAL_SPLIT)
    + int(len(df_param) * TEST_SPLIT)
]
test_data = test_data.reset_index(drop=True)

# Normalize the training dataset
scaler = StandardScaler()
train_data.loc[:, "value"] = scaler.fit_transform(train_data["value"].values.reshape(-1, 1))
val_data.loc[:, "value"] = scaler.transform(val_data["value"].values.reshape(-1, 1))
test_data.loc[:, "value"] = scaler.transform(test_data["value"].values.reshape(-1, 1))

print("Train Data Shape:", train_data.shape)
print("Validation Data Shape:", val_data.shape)
print("Test Data Shape:", test_data.shape)

pd.DataFrame(train_data).to_csv(f"{base_dir}/train/train.csv")
pd.DataFrame(val_data).to_csv(f"{base_dir}/validation/validation.csv")
pd.DataFrame(test_data).to_csv(f"{base_dir}/test/test.csv")

scaler_dir = f"{base_dir}/scaler"
scaler_output_path = f"{scaler_dir}/scaler.pkl"
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(scaler, scaler_output_path)
print(f"Scaler saved to {scaler_output_path}")
