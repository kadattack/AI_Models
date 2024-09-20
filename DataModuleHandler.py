import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch


class CSVDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, target_column, batch_size=32, test_size=0.2, val_size=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.target_column = target_column
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3)
