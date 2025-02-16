
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import optuna

# ---------------------------
# 1) Dataset definition
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=30, pred_len=1):
        self.data = np.array(data["value"]) if isinstance(data, pd.DataFrame) else np.array(data) 
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len) 

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")

        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------------------
# 2) Model definition
# ---------------------------
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        # unsqueeze(-1) => (batch, seq_len, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
        return self.fc(out[:, -1, :])

# ---------------------------
# 3) Optuna objective
# ---------------------------
def build_objective(train_loader, val_loader):
    def objective(trial):
        # Hyperparameter search
        hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        # Model, loss, optimizer
        model = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = 20  # you can adjust

        for epoch in range(epochs):
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

            # (Optional) Pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_loss
    return objective

# ---------------------------
# 4) Main training function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    # SageMaker will populate these with directory paths:
    # e.g. /opt/ml/input/data/train, /opt/ml/input/data/validation
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Some example hyperparameters
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    # 4a) Load data from CSVs inside the train/val/test directories
    # We assume your processor wrote CSVs named train.csv, validation.csv, test.csv
    train_csv = os.path.join(args.train, "train.csv")       # e.g. /opt/ml/input/data/train/train.csv
    val_csv   = os.path.join(args.validation, "validation.csv")
    test_csv  = os.path.join(args.test, "test.csv")

    train_data = pd.read_csv(train_csv)
    val_data   = pd.read_csv(val_csv)
    test_data  = pd.read_csv(test_csv)

    # 4b) Create datasets/loaders
    train_dataset = TimeSeriesDataset(train_data, seq_len=args.seq_len, pred_len=args.pred_len)
    val_dataset   = TimeSeriesDataset(val_data, seq_len=args.seq_len, pred_len=args.pred_len)
    test_dataset  = TimeSeriesDataset(test_data, seq_len=args.seq_len, pred_len=args.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4c) Run Optuna
    objective = build_objective(train_loader, val_loader)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params
    print("Best Params:", best_params)

    # 4d) Train final model with best params
    model = LSTM(input_size=1,
                 hidden_size=best_params["hidden_size"],
                 num_layers=best_params["num_layers"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

    # 4e) Save model to /opt/ml/model (SageMaker will upload to S3 automatically)
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
