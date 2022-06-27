import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import  statsmodels.regression.linear_model as lm




path = r'C:\Users\santo\Desktop\ML Course\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv'

df = pd.read_csv(path)
rd = df.iloc[:,0]
adm = df.iloc[:,1]
Mark = df.iloc[:,2]
St = df.iloc[:,3]
y = df.iloc[:,-1]
x = df.iloc[:,:-1]
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
x_tr,x_te,y_tr,y_te = train_test_split(x,y, test_size=0.2,random_state=0)

#no feature scaling

model = LinearRegression()
model = model.fit(x_tr,y_tr)
y_pred = np.array(model.predict(x_te))
y_te = np.array(y_te)
np.set_printoptions(precision=2)
a=y_pred.reshape(len(y_pred),1)
b=y_te.reshape(len(y_te),1)

print(np.concatenate((a,b),1))
print(model.coef_)
print(model.intercept_)

#Backward elimination
x=np.append(arr= np.ones((50,1)).astype(int), values=x, axis=1)
x_opt= np.array(x[:,[0,1,2,3,4,5]])
mod_ols= lm.OLS(endog=y,exog= x_opt).fit()
print(mod_ols.summary())

x_opt= np.array(x[:,[0,1,3,4,5]])
mod_ols= lm.OLS(endog=y,exog= x_opt).fit()
print(mod_ols.summary())

x_opt= np.array(x[:,[0,3,4,5]])
mod_ols= lm.OLS(endog=y,exog= x_opt).fit()
print(mod_ols.summary())

x_opt= np.array(x[:,[0,4,5]])
mod_ols= lm.OLS(endog=y,exog= x_opt).fit()
print(mod_ols.summary())

x_opt= np.array(x[:,[0,5]])
mod_ols= lm.OLS(endog=y,exog= x_opt).fit()
print(mod_ols.summary())
