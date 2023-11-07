import pandas as pd
import numpy as np
import seaborn as sns

data = 'churn_data.csv'

#Processing data

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

types = df.dtypes

