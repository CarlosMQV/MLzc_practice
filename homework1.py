import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#define the location of the csv
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'

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
sns.histplot(price_logs, bins=50)

#resume of the null values by columns
unexisting=df.isnull().sum()

