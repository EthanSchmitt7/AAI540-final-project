
import joblib
import json
import os
import time
import boto3
import torch
import numpy as np
from torch import nn

# Initialize CloudWatch client
session = boto3.Session()
region_name = session.region_name  # Dynamically get region from the AWS environment
cloudwatch = boto3.client("cloudwatch", region_name=region_name)  # Change region as needed


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def model_fn(model_dir):
    print("Running model_fn")
    model = LSTM(input_size=1, hidden_size=80, num_layers=2)
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    global scaler
    scaler = joblib.load(scaler_path)

    return model


def input_fn(request_body, request_content_type):
    print("Running input_fn")
    if request_content_type == "application/json":
        data = json.loads(request_body)["data"]

        raw_inputs = np.array(data).reshape(-1, 1)
        scaled_inputs = scaler.transform(raw_inputs)
        inputs = torch.tensor(scaled_inputs, dtype=torch.float32)

        # Log raw input data to CloudWatch
        cloudwatch.put_metric_data(
            Namespace="AirQualityMonitoring",
            MetricData=[{"MetricName": "RawQueriedPM25Value", "Unit": "None", "Value": data[0]}],
        )
        return inputs
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_object, model):
    print("Running predict_fn")
    start_time = time.time()
    with torch.no_grad():
        preds = model(input_object)
    inference_time = time.time() - start_time
    input_value = input_object.squeeze().tolist()

    # Log prediction latency and input to model
    cloudwatch.put_metric_data(
        Namespace="AirQualityMonitoring",
        MetricData=[
            {
                "MetricName": "InputPM25Value",
                "Unit": "None",
                "Value": input_value[0] if isinstance(input_value, list) else input_value,
            },
            {
                "MetricName": "InferenceLatency",
                "Unit": "Milliseconds",
                "Value": inference_time * 1000,  # Convert to ms
            },
        ],
    )
    return preds


def output_fn(prediction, response_content_type):
    print("Running output_fn")
    if response_content_type == "application/json":
        scaled_result = prediction.squeeze().item()
        # Convert prediction to original scale
        result = scaler.inverse_transform(np.array(scaled_result).reshape(-1, 1)).item()

        # Log prediction value
        cloudwatch.put_metric_data(
            Namespace="AirQualityMonitoring",
            MetricData=[{"MetricName": "PredictedPM25", "Unit": "None", "Value": result}],
        )
        return json.dumps({"prediction": result})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
