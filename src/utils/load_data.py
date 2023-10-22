"""
Data Loading for Algorithmic Trading
Takumi Sudo
"""

import numpy as np
import pandas as pd
from datetime import date
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
import calendar

class StockDataset(Dataset):
    def __init__(self, ticker, start_date, end_date, window_size=60):
        # Fetch data
        self.data = yf.download(ticker, start=start_date, end=end_date)
        self.prices = self.data['Close'].values
        self.window_size = window_size

    def __len__(self):
        # Return number of samples
        return len(self.prices) - self.window_size

    def __getitem__(self, idx):
        # Given an index, return the window_size samples before it as features (X) 
        # and the sample at the index as target (y)
        x = self.prices[idx: idx+self.window_size]
        y = self.prices[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

def stock_dataloader(ticker, start_date, end_date, window_size=60, batch_size=32, test_split=0.2):
    dataset = StockDataset(ticker, start_date, end_date, window_size)
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


