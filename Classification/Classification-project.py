import pandas as pd
import numpy as np
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

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

#Feature importance: Correlation

#This is a way to measure dependency between two variables
#The correlation coefficient "r" can take values between -> -1 < r < 1
#When r < 0, if x grows, y decreases
#When r > 0, if x grows, y grows too

#When correlation is between -0.2 and 0 or 0 and 0.2, is almost unexisting
#In that case, when one grows or decreases, the other rarely grows or decreases
#When correlation is between -0.2 and -0.5 or 0.2 and 0.5, there is moderate correlation
#In that case, when one grows or decreases, the other sometimes grows or decreases
#When correlation is between -0.5 and -1 or 0.5 and 1, there is strong correlation
#In that case, when one grows or decreases, the other often/always grows or decreases
#If correlation is 0, there are no effects on the other variable

#Correlation between the numerical values and churn
def correlation_plots(daf, variables, target, absl):
    corr = list(daf[variables].corrwith(daf[target]))
    corr = [(abs(x) if absl else x) * 100 for x in corr]
    X = variables
    Y = corr
    pl.bar(X, Y, facecolor='#ff9999', edgecolor='white')
    text_y_values = [y + 0.05 if y >= 0 else y - 12 for y in Y]
    for x, y, text_y in zip(X, Y, text_y_values):
        pl.text(x, text_y, f'{y:.2f}', ha='center', va='bottom')
    pl.ylim((0 if absl else -110), 110)
#correlation_plots(df_full_train, numerical, 'churn', True)

#Correlation between one part of a variable and churn
def correlation_intervals_plot(var, lim1, lim2, color):
    corr1 = 100*round(df_full_train[df_full_train[var] < lim1].churn.mean(), 2)
    corr2 = 100*round(df_full_train[(df_full_train[var] > lim1)
                      & (df_full_train[var] < lim2)].churn.mean(), 2)
    corr3 = 100*round(df_full_train[df_full_train[var] > lim2].churn.mean(), 2)
    X = ['< %s' % lim1, '%s - %s' % (lim1, lim2), '> %s' % lim2]
    Y = [corr1, corr2, corr3]
    pl.bar(X, Y, facecolor=color, edgecolor='white', alpha=0.5)
    for x, y in zip(X, Y):
        pl.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom')
    pl.ylim(0, +120)
#correlation_intervals_plot('tenure', 2, 12,'#9999ff') #Negative correlation
#correlation_intervals_plot('monthlycharges', 20, 50,'#ff9999') #Positive correlation

#One-hot encoding (using Scikit-Learn)

#We create a dictionary
train_dicts = df_train[categorical + numerical].to_dict(orient = 'records')
#We create a new instance of this class
dv = DictVectorizer(sparse=False)
#Then we train our DictVectorizer. It means that we show it the info so it can process it.
dv.fit(train_dicts)
#We can get the feature names:
dv_fn = dv.get_feature_names_out()
#Then we transform it. We get columns of ones and zeros.
#If there is a numerical variable, dosen't change it.
X_train = dv.transform(train_dicts)

#For the others
val_dicts = df_val[categorical + numerical].to_dict(orient = 'records')
X_val = dv.transform(val_dicts)
test_dicts = df_val[categorical + numerical].to_dict(orient = 'records')
X_test = dv.transform(test_dicts)

#Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

#Values of w
w_values = model.coef_[0]

#Model bias
m_bias = model.intercept_[0]

#Using the model. These are hard predictions, no probabilities are shown.
hard_p = model.predict(X_train)

#Using the model and getting probabilities. Soft predictions.
soft_p = model.predict_proba(X_train)

#We select just the 1 probabilities.
y_pred = model.predict_proba(X_train)[:,1]

#Analysis for y_val
y_pred = model.predict_proba(X_val)[:,1]
churn_decision = (y_pred >= 0.5)
ids = df_val[churn_decision].customerid

#Measuring the accuracy
accuracy = (y_val == churn_decision).mean()

#Model interpretation
#Join featueres with their weights
a1 = dict(zip(dv_fn, w_values))

#Using the model
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient = 'records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
y_full_train = df_full_train.churn.values
model = LogisticRegression()
model.fit(X_full_train, y_full_train)

#Testing

dicts_test = df_test[categorical+numerical].to_dict(orient = 'records')
X_test = dv.transform(dicts_test)

y_pred = model.predict_proba(X_test)[:,1]
churn_decision = (y_pred >= 0.5)
a2 = (churn_decision == y_test).mean()

#Particular example

a3 = 15     #customer number
customer = dicts_test[a3]
X_small = dv.transform([customer])
customer_pred = model.predict_proba(X_small)[0,1]
customer_real = y_test[a3]