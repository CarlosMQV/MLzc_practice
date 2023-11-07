import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

data = 'churn_data.csv'

#Processing data

#Importing and pre-processing data
df = pd.read_csv(data)
df.columns = df.columns.str.lower().str.replace(' ','_')
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ','_')

#Converted strings to float and replaced Nan with 0
df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')
df.totalcharges = df.totalcharges.fillna(0)

#Converted "yes" or "no" values to 0 or 1.
df.churn = (df.churn == 'yes').astype('int')

#Use of sklearn
df_full_train, df_test = train_test_split(df,test_size=0.2,random_state=1)
df_train, df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)

#Reset index of dataframes
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True) 
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#Obtaining y values
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

#Erasing churn values from the dataframes
del df_train['churn']
del df_val['churn']
del df_test['churn']

#Searching for specific values
#df_full_train.isnull().sum() #-> There are no missing values
#count = df_full_train.churn.value_counts(normalize=True) #-> For counting the number of values of each type
global_churn_rate = df_full_train.churn.mean()

#Define the numerical variables

numerical = ['tenure','monthlycharges','totalcharges']

categorical = ['customerid', 'gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice','onlinesecurity',
               'onlinebackup', 'deviceprotection', 'techsupport','streamingtv',
               'streamingmovies', 'contract', 'paperlessbilling','paymentmethod']

unique_values = df_full_train[categorical].nunique()









