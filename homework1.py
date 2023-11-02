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
#price_logs = np.log1p(df.msrp)

#Plot of the price logarithms
#sns.histplot(price_logs, bins=50)

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

#Reset index of dataframes
df_train = df_train.reset_index(drop=True) 
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#Set y as array (we get the logarithms)
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

#Not gonna use msrp from this datasets, so we erase them
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

#Create a function for linear regression
def train_linear_regression(X,y):
    
    ones = np.ones(X.shape[0]) #vector of ones
    X = np.column_stack([ones,X]) #add ones to X columns
    
    XTX = X.T.dot(X) #gram matrix (trasposed X dot X)
    XTX_inv = np.linalg.inv(XTX) #inverse of gram
    w_full = XTX_inv.dot(X.T).dot(y) #X.w = y -> XT.X.w = XT.y -> w = (XT.X)-1.XT.y
    
    return w_full[0],w_full[1:]

#columns we want
base = ['engine_hp','engine_cylinders','highway_mpg', 'city_mpg', 'popularity']
makes = list(df.make.value_counts().head(5).index)
#function for preparation of X filling missing values
def prepare_X(df):
    df = df.copy() #for not to modify the original dataframe
    features = base.copy() #copy for using append and not modify base
    
    df['age']=2017-df.year #Explained in feature engineering section
    features.append('age')
    
    for v in [2,3,4]: #appending categorical values. Explained further below
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)
    
    for v in makes:
        df['make_%s' % v] = (df.make == v).astype('int')
        features.append('make_%s' % v)
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train) #We prepare X with the function 
w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w) #predicted y

sns.histplot(y_pred, color ='red', bins=100, alpha=0.5)
sns.histplot(y_train, color ='blue', bins=100, alpha=0.5)

#Function for RMSE

def rmse(y,y_pred):
    error = (y - y_pred)
    se = error**2
    mse = se.mean()
    return np.sqrt(mse)

rmse_train = rmse(y_train,y_pred)

#Validation

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse_val = rmse(y_val, y_pred)

#Simple feature engineering

#If we calculate the max year with:
#df_train.year.max()
#Using a small number is better than the year
#So we can use 2017 - df_train.year
#So I modified the prepare_X function for this purpose

#Categorical variables

#for v in [2,3,4]:
    #df_train['num_doors_%s' % v] = (df_train.number_of_doors == v).astype('int')
#Here we create three binary columns from each category
#Also we add this feature in the prepare_X function

#We do the same with makes
#makes = list(df.make.value_counts().head().index)






    