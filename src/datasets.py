
## create the data sets

import torch
import pandas as pd 
import os

from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader , TensorDataset
from sklearn.preprocessing import StandardScaler

cpu_count = os.cpu_count()

def load_data(
    path = '../data/creditcard.csv' ,
    batch_size = 1024 , 
    test_size = 0.3 ,
    random_state = 42 ,
    cpu_count = cpu_count
):
    df  = pd.read_csv(path)

    X = df.drop('Class' , axis = 1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    X_train = torch.tensor(X_train , dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy() , dtype=torch.float32)

    X_test = torch.tensor(X_test , dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy() , dtype=torch.float32)

    train_dataset = TensorDataset(X_train , y_train)
    test_dataset = TensorDataset(X_test , y_test)

    train_loader = DataLoader(
        train_dataset ,
        batch_size=batch_size ,
        shuffle=True ,
        num_workers=cpu_count,
        pin_memory=True

    )

    test_loader = DataLoader(
        test_dataset ,
        batch_size= batch_size ,
        shuffle= False ,
        num_workers=cpu_count,
        pin_memory=True
    )

    input_dim = X_train.shape[1]

    return train_loader , test_loader , input_dim
