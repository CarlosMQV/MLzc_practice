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
base = [feature for feature in feature_names if feature != "Price"]
df = pd.read_csv(data, delimiter=',')
df.columns = feature_names
df = df[~(df['Price'] == 500001)]

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

y_train = np.log1p(df_train.Price.values)
y_val = np.log1p(df_val.Price.values)
y_test = np.log1p(df_test.Price.values)

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
    features = base.copy()
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def rmse(y,y_pred):
    error = (y - y_pred)
    se = error**2
    mse = se.mean()
    return np.sqrt(mse)

#Training

X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train,r=0.001)
y_pred_train = w0 + X_train.dot(w)
rmse_train = rmse(y_train,y_pred_train)

#Validation

X_val = prepare_X(df_val)
y_pred_val = w0 + X_val.dot(w)
rmse_val = rmse(y_val,y_pred_val)

#Full trained model

df_full_train = pd.concat([df_train,df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train,y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train,r=0.001)

#Testing

X_test = prepare_X(df_test)
y_pred_test = w0 + X_test.dot(w)
rmse_test = rmse(y_test,y_pred_test)

#Using the model with a prediction

number = 200
house = df_test.iloc[number].to_dict()
df_small = pd.DataFrame([house])
X_small = prepare_X(df_small)
y_pred_small = w0 + X_small.dot(w)
y_pred_small = y_pred_small[0]
predicted_price = np.expm1(y_pred_small)
price = np.expm1(y_test[number])

sns.histplot(y_pred_test, color ='red', bins=100, alpha=0.5)
sns.histplot(y_test, color ='blue', bins=100, alpha=0.5)

