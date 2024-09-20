import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Define the Lightning Module
class TransformerRegressor(pl.LightningModule):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, lr=1e-3):
        super(TransformerRegressor, self).__init__()
        self.save_hyperparameters()

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, 1)
        self.criterion = nn.MSELoss()  # For regression tasks

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, input_dim)
        """
        x = self.embedding(x)  # (batch_size, sequence_length, d_model)
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
        x = self.fc_out(x[:, -1, :])  # Predict using the last time step
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


# Data Module for Handling CSV Time Series Data
class TimeSeriesCSVDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, target_column, seq_length=10, batch_size=32, test_size=0.2, val_size=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.target_column = target_column
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        df = df.dropna()

        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values.reshape(-1, 1)

        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

        # Prepare sequences for time series
        X_seq, y_seq = self.create_sequences(X, y, self.seq_length)

        # Split the data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=self.test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=42)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Create TensorDatasets for each split
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

    def create_sequences(self, X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3)


# Main Function to Train and Test the Model
if __name__ == "__main__":
    csv_path = "./csvData/timeseries_data.csv"  # Replace with your time series CSV file path
    target_column = "target_column_name"  # Replace with your actual target column name

    data_module = TimeSeriesCSVDataModule(csv_path=csv_path, target_column=target_column, batch_size=32, seq_length=10)
    data_module.setup()

    input_dim = data_module.train_dataset.tensors[0].shape[2]

    model = TransformerRegressor(input_dim=input_dim)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=80,
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=7
    )

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

    # Access scalers
    scaler_X = data_module.scaler_X
    scaler_y = data_module.scaler_y

    # Get predictions for the test set
    test_loader = data_module.test_dataloader()
    all_preds = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)

            preds = model(x)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Rescale predictions and targets
    all_preds_rescaled = scaler_y.inverse_transform(all_preds)
    all_targets_rescaled = scaler_y.inverse_transform(all_targets)

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets_rescaled, label='Actual Values', color='blue', linestyle='--', marker='o')
    plt.plot(all_preds_rescaled, label='Predicted Values', color='red', linestyle='-', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()












# Example sequence of features (e.g., 10 time steps, each with 2 features)
new_data = np.array([
    [0.2, 0.5],   # x1
    [0.3, 0.6],   # x2
    [0.4, 0.7],   # x3
    [0.5, 0.8],   # x4
    [0.6, 0.9],   # x5
    [0.7, 1.0],   # x6
    [0.8, 1.1],   # x7
    [0.9, 1.2],   # x8
    [1.0, 1.3],   # x9
    [1.1, 1.4]    # x10
])

# Reshape the data for a single sample: (1, seq_length, num_features)
new_data = new_data[np.newaxis, :, :]  # Shape: (1, 10, 2)

# Convert to PyTorch tensor (dtype must match model's expected dtype)
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)

# Make prediction
model.eval()  # Ensure the model is in evaluation mode
with torch.no_grad():
    prediction = model(new_data_tensor)

# Output prediction
print("Predicted value:", prediction)


