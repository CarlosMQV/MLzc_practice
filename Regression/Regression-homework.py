import pandas as pd
import numpy as np
import seaborn as sns

#Import data into a panda's dataframe
data = 'data_houses.data'
feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "Price"
    ]
df = pd.read_csv(data, delimiter=',')
df.columns = feature_names
df = df[~(df['Price'] == 500001)]
sns.histplot(df.Price, bins=50)

#Data processing

n = len(df)
n_val = int(0.2*n)
n_test = int(0.2*n)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.Price.values
y_val = df_val.Price.values
y_test = df_test.Price.values

del df_train['Price']
del df_val['Price']
del df_test['Price']

#Functions

def train_linear_regression_reg(X,y,r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])
    XTX = X.T.dot(X)
    XTX = XTX + r*np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0],w_full[1:]

def prepare_X(df):
    df = df.copy()
    features = feature_names.copy()    
    df_num = df[features].fillna(0)
    X = df_num.values
    return X

def rmse(y,y_pred):
    error = (y - y_pred)
    se = error**2
    mse = se.mean()
    return np.sqrt(mse)





