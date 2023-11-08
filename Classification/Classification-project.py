import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

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

#Exploratory data analysis

#df_full_train.isnull().sum() #-> There are no missing values
#count = df_full_train.churn.value_counts(normalize=True) #-> For counting the number of values of each type
global_churn = df_full_train.churn.mean() #Global churn rate
numerical = ['tenure','monthlycharges','totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice','onlinesecurity',
               'onlinebackup', 'deviceprotection', 'techsupport','streamingtv',
               'streamingmovies', 'contract', 'paperlessbilling','paymentmethod']
unique_values = df_full_train[categorical].nunique() #For unique values into each categorical variable

"""Churn rate by gender
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
global_churn - churn_female
global_churn - churn_male"""

#Partner count

"""churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
global_churn - churn_partner
global_churn - churn_no_partner"""

#This analysis for feature importance.

#We can measure it by subtracting the group index minus the global index
#If the difference is > 0, the group is more likely to churn
#If the difference is < 0, the group is less likely to churn
#The greater the difference, the more influential it is

#We can also divide the group index by the global index (risk ratio)
#If the risk ratio is > 1, the group is more likely to churn
#If the risk ratio is < 1, the group is less likely to churn

#With a ratio of, for example, 1.01, the churn risk is practically the same as the global one.
#With a ratio of, for example, 0.75 or less, the risk of churn is low.
#With a ratio of, for example, 1.22 or more, the risk of churn is high.
#Each one defines the classification of risks according to what is most convenient.

#Fast way of obtaining the churn substraction
"""for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['ratio'] = df_group['mean'] / global_churn
    print(df_group)
    print()
    print()"""

#Feature importance: Mutual information
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)
mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi = mi.sort_values(ascending = False)

