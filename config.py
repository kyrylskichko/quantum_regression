import pandas as pd
import numpy as np

val_size = 0.2
lr = 0.05
epochs = 60
batch_size = 64

train_data = pd.read_csv('data/internship_train.csv')
test_data = pd.read_csv('data/internship_hidden_test.csv')

train_data = train_data[['6', '7', '8', 'target']]
test_data = test_data[['6', '7', '8']]

X_train = train_data.to_numpy(dtype=np.float32)[:, :-1]
Y_train = train_data.to_numpy(dtype=np.float32)[:, -1]
X_test = test_data.to_numpy(dtype=np.float32)

