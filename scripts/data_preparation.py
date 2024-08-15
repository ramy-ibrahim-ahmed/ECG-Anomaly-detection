import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_ECG():
    # Load the data
    data_normal = pd.read_csv(
        r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\data\ptbdb_normal.csv"
    )
    data_abnormal = pd.read_csv(
        r"D:\ramy\Omnetrex\Heartbeat anomaly detection on time series\data\ptbdb_abnormal.csv"
    )

    # Remove targets
    data_normal = data_normal.iloc[:, :-1]
    data_abnormal = data_abnormal.iloc[:, :-1]

    # Name the columns
    data_normal.columns = [f"feature_{i}" for i in range(data_normal.shape[1])]
    data_abnormal.columns = [f"feature_{i}" for i in range(data_abnormal.shape[1])]

    # Set the target
    data_normal["label"] = 0
    data_abnormal["label"] = 1

    # Concatinate normal and abnormal
    data = pd.concat([data_normal, data_abnormal], axis=0).reset_index(drop=True)

    # Seperate labels before handeling NaNs
    labels = data["label"].values
    data = data.drop("label", axis=1)

    # Handling NaNs
    # make any 0 nan as it is a padding befor nan handeling
    # forward and backward fill
    """
        By chaining these two fillna operations, 
        the code attempts to fill all missing values in the dataset. 
        First, it tries to fill missing values by propagating the last valid observation forward.
        If there are still missing values after this, 
        it then propagates the next valid observation backward to fill the remaining gaps.
    """
    data = data.replace(0, np.nan)
    data = data.ffill().bfill()
    data = data.dropna(axis=1, how="all")

    xtrain, xtest, ytrain, ytest = train_test_split(
        data,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=123,
    )

    return xtrain, xtest, ytrain, ytest
