"""
Recreating , “Deep Learning and Time Series-to-Image Encoding for Financial Forecasting” paper
This code utilizes the gaf and deep learning techiniques to achieve ~52 % of the market pricings
"""

import pandas as pd
import os
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from utils import load_data



def preprocess():
    ticker = "^GSPC"
    trainloader, testloader = load_data.stock_dataloader(ticker, "1990-01-01", "2020-12-10", window_size=5)
    return trainloader, testloader




