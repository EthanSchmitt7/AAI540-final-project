
import json
import os
import torch
import numpy as np
from torch import nn

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
        # out is (batch_size, seq_len, hidden_size)
        # We want the last timestep
        out = out[:, -1, :]
        out = self.fc(out)  # shape (batch_size, 1)
        return out

def model_fn(model_dir):
    """
    Loads the model from the model_dir. This is invoked by SageMaker once at
    container startup to initialize your model. The returned object is passed
    to `predict_fn` for every inference request.
    """
    # Create model with the same architecture/hyperparams as training
    model = LSTM(input_size=1, hidden_size=50, num_layers=1)
    # Load state dict from model.pt
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Deserializes the incoming request body into a PyTorch tensor.
    - If you expect JSON, parse it.
    - If you expect CSV, parse differently, etc.
    """
    if request_content_type == "application/json":
        # Example: request_body = '{"data": [12.3, 45.6, 78.9, ...]}'
        data = json.loads(request_body)["data"]
        # Convert to a float32 tensor. Suppose it's a 1D series (seq_len).
        inputs = torch.tensor([data], dtype=torch.float32)
        return inputs
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_object, model):
    """
    Performs prediction on the deserialized input.
    """
    with torch.no_grad():
        # input_object shape = (batch=1, seq_len)
        preds = model(input_object)  # shape = (batch=1, 1)
    return preds

def output_fn(prediction, response_content_type):
    """
    Serializes the prediction output.
    """
    if response_content_type == "application/json":
        # Convert the tensor to a Python float
        result = prediction.squeeze().item()  # single float
        return json.dumps({"prediction": result})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
