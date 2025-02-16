
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------
# 1) Dataset definition
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=30, pred_len=1):
        """
        Args:
        - data (pd.DataFrame or np.array): Time series with a column 'value'
        - seq_len (int): Number of timesteps in the input sequence
        - pred_len (int): Number of timesteps in the output sequence
        """
        # Expecting a df with at least one column named "value"
        self.data = np.array(data["value"]) if isinstance(data, pd.DataFrame) else np.array(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # e.g. if data has 100 points, and seq_len=30, pred_len=1 -> total sequences ~ 69
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")

        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        # Return tensors for PyTorch
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------------------
# 2) Model definition
# ---------------------------
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        """
        A simple LSTM for univariate time series prediction.
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len)
        We'll unsqueeze(-1) to get (batch_size, seq_len, input_size=1).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
        # out is (batch_size, seq_len, hidden_size)
        # We want the last timestep
        out = out[:, -1, :]
        out = self.fc(out)  # shape (batch_size, 1)
        return out

# ---------------------------
# 3) Main training function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    # Channels for data paths (SageMaker will populate these automatically)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Hyperparameters
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    # ---------------------------
    # 3a) Load data
    # ---------------------------
    # We assume your processor wrote "train.csv", "validation.csv", and "test.csv"
    # into the respective directories: /opt/ml/input/data/train, etc.
    train_csv = os.path.join(args.train, "train.csv")
    val_csv   = os.path.join(args.validation, "validation.csv")
    test_csv  = os.path.join(args.test, "test.csv")

    train_data = pd.read_csv(train_csv)
    val_data   = pd.read_csv(val_csv)
    test_data  = pd.read_csv(test_csv)

    # ---------------------------
    # 3b) Create PyTorch datasets & loaders
    # ---------------------------
    train_dataset = TimeSeriesDataset(train_data, seq_len=args.seq_len, pred_len=args.pred_len)
    val_dataset   = TimeSeriesDataset(val_data, seq_len=args.seq_len, pred_len=args.pred_len)
    test_dataset  = TimeSeriesDataset(test_data, seq_len=args.seq_len, pred_len=args.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------
    # 3c) Initialize model, loss, optimizer
    # ---------------------------
    model = LSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------
    # 3d) Training Loop
    # ---------------------------
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")

    # ---------------------------
    # 3e) (Optional) Test evaluation
    # ---------------------------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_outputs = model(test_inputs)
            test_loss += criterion(test_outputs, test_targets).item()
    test_loss /= len(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")

    # ---------------------------
    # 3f) Save the model
    # ---------------------------
    # SageMaker automatically uploads /opt/ml/model to S3 after training
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
