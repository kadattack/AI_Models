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


# Define the Lightning Module for CNN on Time Series Data
class CNNTimeSeriesRegressor(pl.LightningModule):
    def __init__(self, input_dim, seq_length, num_filters=64, kernel_size=3, dropout=0.1, lr=1e-3):
        super(CNNTimeSeriesRegressor, self).__init__()
        self.save_hyperparameters()

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)

        # Flatten after convolution
        conv_output_size = self._get_conv_output_size(seq_length, kernel_size)

        # Fully connected layer to get the output for regression
        self.fc_out = nn.Linear(conv_output_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.MSELoss()  # For regression tasks

    def _get_conv_output_size(self, seq_length, kernel_size):
        # Since we are using padding, the size after conv layers will remain the same
        return seq_length * (64 * 2)  # 64 filters from conv1, and 64*2 from conv2

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, input_dim)
        """
        # Apply convolution over the sequence
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length * input_dim) - needed for Conv1d
        x = self.conv1(x)  # (batch_size, num_filters, seq_length)
        x = nn.functional.relu(x)
        x = self.conv2(x)  # (batch_size, num_filters * 2, seq_length)
        x = nn.functional.relu(x)

        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        # Apply dropout and the fully connected layer
        x = self.dropout(x)
        x = self.fc_out(x)  # (batch_size, 1)

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


# Data Module for Handling Time Series CSV Data
class TimeSeriesCSVDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, target_column, batch_size=32, test_size=0.2, val_size=0.1, seq_length=10):
        super().__init__()
        self.csv_path = csv_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.seq_length = seq_length
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

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=42)

        # Create sequences for the CNN
        self.train_dataset = self.create_sequences(X_train, y_train)
        self.val_dataset = self.create_sequences(X_val, y_val)
        self.test_dataset = self.create_sequences(X_test, y_test)

    def create_sequences(self, X, y):
        sequences = []
        targets = []
        for i in range(len(X) - self.seq_length):
            seq_X = X[i:i + self.seq_length]
            target_y = y[i + self.seq_length]
            sequences.append(seq_X)
            targets.append(target_y)
        return TensorDataset(torch.tensor(sequences, dtype=torch.float32),
                             torch.tensor(targets, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3)


# Main Function to Train and Test the Model
if __name__ == "__main__":
    csv_path = "./csvData/sumdata.csv"
    target_column = "target"  # Replace with your actual target column name

    # Initialize DataModule
    data_module = TimeSeriesCSVDataModule(csv_path=csv_path, target_column=target_column, batch_size=32, seq_length=10)

    # Setup the data
    data_module.setup()

    # Determine input dimension based on the dataset
    input_dim = data_module.train_dataset.tensors[0].shape[2]
    seq_length = data_module.seq_length

    # Initialize Model
    model = CNNTimeSeriesRegressor(input_dim=input_dim, seq_length=seq_length)

    # Define Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=80,
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=7
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
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
