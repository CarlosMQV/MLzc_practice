import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#define the location of the csv
data = 'data.csv'

#clean columns data
df = pd.read_csv(data)
df.columns = df.columns.str.lower().str.replace(' ','_')
strings = list(df.dtypes[df.dtypes == 'object'].index)

#erase spaces on strings in the data
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ','_')

#Get the log of prices to change the plot
price_logs = np.log1p(df.msrp)

#Plot of the price logarithms
#sns.histplot(price_logs, bins=50)

#resume of the null values by columns
#unexisting=df.isnull().sum()

n = len(df) #size of the dataset
n_val = int(n*0.2) #size of the validation dataset (20%)
n_test = int(n*0.2) #size of the test dataset (20%)
n_train = n - n_val - n_test #size of the training dataset (~60%)

idx = np.arange(n) #create a 0 to n-1 list of number
np.random.seed(2)
np.random.shuffle(idx) #shuffle those numbers

df_train = df.iloc[idx[:n_train]] #dataframe for training
df_val = df.iloc[idx[n_train:n_train+n_val]] #dataframe for validation
df_test = df.iloc[idx[n_train+n_val:]] #dataframe for test

#Rest index of dataframes
df_train = df_train.reset_index(drop=True) 
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#Set y as array
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

#Not gonna use them, so we erase for security
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']




