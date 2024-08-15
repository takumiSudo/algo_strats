# Novel Chart Patterns
**Author:** takumiSudo
**Date:** <2024-07-23 Tue>

## Objective:
To find patterns in historical data, identifying patterns from the **Perceptually Important Points (PIP)**. This is the pattern finding algorithm that finds the 3 most distant points within a timeframe. In the video, the author chooses 24 candles apart for a 1-hour time period, and the algorithm will choose the 3 points that are most distant.

With PIP, the dimension of candles is reduced significantly as the 24 candles can now be represented by 5 points of data.

After the PIP runs through the whole section of the data using a **Rolling Window**, which slides the period of contention over time (a common technique utilized in time series analysis), it will then use **Clustering Algorithms** to group the patterns into different groups. The author uses K-means clusters for simplicity's sake, but essentially, given the PIP over a certain period of time, these PIPs will be clustered into groups resulting in the creation of patterns.

## 1. Data Intake
The first section is the easy part. Import the data and set the index values to the right time, which in this case is Date.

```bash
pip install pandas
```

```python
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

data = pd.read_csv('../data/BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')
data = np.log(data)
plt.style.use('dark_background')
```

## 2. PIP Analyzer Initialization
The next section goes over the anatomy of the PIP analyzer. The most important components of this section are `n_pips` and `lookback`.

```python
class PIPPatternMiner:
    def __init__(self, n_pips: int, lookback: int, hold_period: int):
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period

        self._unique_pip_patterns = []
        self._unique_pip_indices = []
        self._cluster_centers = []
        self._pip_clusters = []

        self._cluster_signals = []
        self._cluster_objs = []

        self._long_signal = None
        self._short_signal = None

        self._selected_long = []
        self._selected_short = []

        self._fit_martin = None
        self._perm_martins = []

        self._data = None  # Array of log closing prices to mine patterns
```

This `__init__` section goes over the hold period, which is going to be 24 for this project.

## 3. PIP Iterator
This section explains how the PIPs are calculated while on a sliding window with 24 candles. In the class `pip analyzer`, there is the `_find_unique_patterns()` function that finds the pip patterns.
```python
def find_pips(data: np.array, n_pips: int, dist_measure: int):

    """
    Distance Measure
    1. Euclidean Distance
    2. Perpindicular
    3. Vertical Distance
    """

    pips_x = [0, len(data) - 1] # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max Distance
        md_i = -1 # Max Distance Index
        insert_index = -1

        for k in range(0, curr_point -1):

            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]

            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope;

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                d = 0.0
                if dist_measure == 1:
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                elif dist_measure == 2:
                    d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                else:
                    d = abs((slope * i + intercept) - data[i])


            if d > md:
                md = d
                md_i = i
                insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y
```

Given the 2 dimensions of a time series x and y which equivalent to the index(time stamp) and the price of the stock, we can iterate through the set time frame to find the PIPs. 

### Initialization
The function starts off with initializing a list call `pips` with the first 0 and the last data `len(data) - 1`. 

### Iteration 
The main loop for this function goes through the `2` to `n_pips - 2` points since the first and last points are already taken.

### Calculating the Distance
- For each point in the time series, the function can calculate the following measurements: `Euclidean`, `perpendicular`, and `Vertical` distances from the nearest PIPs

### Updating the list of PIPs
The point with the maximum distance is inserted into the `pips` list, by updating if the `d > md`


## PIP Pattern miner
Now lets take a look at the main code for the pip pattern finder.
The first function that the class calls is `train`, which loads the data as the arr and initializes the returns for the script.

Then the function goes into the `find_unique_pips` function that utilizes the above mentioned calculating pips functions and iterates the whole data array to find the pips during the set timeframe.


