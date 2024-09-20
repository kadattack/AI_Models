import torch
import numpy as np
import pandas as pd
import torch.nn as nn

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from DataModuleHandler import CSVDataModule
from torch.utils.data import DataLoader


class FeedforwardNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[32, 16], output_dim=1, lr=1e-3):
        super(FeedforwardNN, self).__init__()
        self.save_hyperparameters()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

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


if __name__ == "__main__":
    # Path to your CSV file and target column
    csv_path = "./csvData/sumdata.csv"
    target_column = "c"  # Replace with your actual target column name

    # Initialize DataModule
    data_module = CSVDataModule(csv_path=csv_path, target_column=target_column, batch_size=32)

    # Setup the data
    data_module.setup()

    # Determine input dimension based on the dataset
    input_dim = data_module.train_dataset.tensors[0].shape[1]

    # Initialize Model
    model = FeedforwardNN(input_dim=input_dim)

    # Define Early Stopping Callback (optional but recommended)
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

    # Extract test data
    test_loader = data_module.test_dataloader()
    all_preds = []
    all_targets = []

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)

            # Perform inference
            preds = model(x)

            # Collect predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Rescale predictions if necessary
    scaler_y = data_module.scaler_y
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
