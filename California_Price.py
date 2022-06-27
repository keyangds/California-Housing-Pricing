import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import ResidualsPlot
from sklearn.preprocessing import MinMaxScaler as ms

data = pd.read_csv('C:\\Users\\xuank\\OneDrive\\桌面\\Fall2021\\IE3013\\housing.csv', parse_dates = True, index_col = 0, encoding = 'gbk')

data = data.reset_index()

## remove some outliers
data = data.loc[data['Median_House_Value']<500001,:]
data = data.loc[data['Population']<25000]

data_clean = data.dropna()  ## drop NA in dataset if exist

## check collinearity
x = add_constant(data_clean)
vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif_info['Column'] = x.columns
vif_info.sort_values('VIF', ascending=False)

## feature selection
data_clean['income per working population']=data_clean['Median_Income']/(data_clean['Population']-data_clean['Households'])
data_clean['bed per house']=data_clean['Tot_Bedrooms']/data_clean['Tot_Rooms']
data_clean['h/p']=data_clean['Households']/data_clean['Population']
data_clean['Avg_Distance']=(data_clean['Distance_to_SanDiego']+data_clean['Distance_to_SanJose']+data_clean['Distance_to_SanFrancisco'])/3

##data_clean = data_clean.drop(['Median_Income', 'Tot_Rooms', 'Tot_Bedrooms', 'Households'],axis=1)

X = data_clean.iloc[:,1:].astype(float)
Y = data_clean.iloc[:,0:1].astype(float)



## split training set and test set 
train_data = data_clean.iloc[0:6192,:]
test_data = data_clean.iloc[6192:20640,:]

test_data = test_data.reset_index(drop=True)


## training set
X_train = train_data.iloc[:,1:].astype(float)
Y_train = train_data.iloc[:,0:1].astype(float)

## testing set
X_test = test_data.iloc[:,1:].astype(float)
Y_test = test_data.iloc[:,0:1].astype(float)


## Normal Linear Regression
linear = LinearRegression()
linear.fit(X_train,Y_train)


## Selection of lambda based on R^2
lasso_alpha = pd.DataFrame(columns=['alpha', 'R^2'])
ridge_alpha = pd.DataFrame(columns=['alpha', 'R^2'])

for i in range(2000):
    lasso = Lasso(alpha=i)
    lasso.fit(X_train,Y_train)
    lasso_alpha = lasso_alpha.append({'alpha':i, 'R^2':r2_score(Y_test, lasso.predict(X_test))},ignore_index=True)
    
for i in range(2000):
    ridge = Ridge(alpha=i)
    ridge.fit(X_train,Y_train)
    ridge_alpha = ridge_alpha.append({'alpha':i, 'R^2':r2_score(Y_test, ridge.predict(X_test))},ignore_index=True)
    
    
    
## Selection of lambda based on MSE
lasso_alpha_1 = pd.DataFrame(columns=['alpha', 'MSE'])
ridge_alpha_1 = pd.DataFrame(columns=['alpha', 'MSE'])


for i in range(2000):
    lasso = Lasso(alpha=i)
    lasso.fit(X_train,Y_train)
    lasso_alpha_1 = lasso_alpha_1.append({'alpha':i, 'MSE': mean_squared_error(Y_test, lasso.predict(X_test))},ignore_index=True)
    
for i in range(2000):
    ridge = Ridge(alpha=i)
    ridge.fit(X_train,Y_train)
    ridge_alpha_1 = ridge_alpha_1.append({'alpha':i, 'MSE': mean_squared_error(Y_test, ridge.predict(X_test))},ignore_index=True)
  
    

## Lasso Regression
lasso = Lasso(alpha = 1300)
lasso.fit(X_train,Y_train)
    

## Ridge Regression
ridge = Ridge(alpha = 800)
ridge.fit(X_train,Y_train)


## expected value from regression

linear_y = linear.predict(X_test)
lasso_y = lasso.predict(X_test)
lasso_y = np.reshape(lasso_y,(-1,1))
ridge_y = ridge.predict(X_test)


## check performance on training set (R^2)
s1 = [[r2_score(Y_train, linear.predict(X_train)), r2_score(Y_train, lasso.predict(X_train)), r2_score(Y_train, ridge.predict(X_train))]]
train_perform = pd.DataFrame(s1, columns = ['linear', 'lasso', 'ridge'])

## check performance on testing set (R^2)
s2 = [[r2_score(Y_test, linear.predict(X_test)), r2_score(Y_test, lasso.predict(X_test)), r2_score(Y_test, ridge.predict(X_test))]]
test_perform = pd.DataFrame(s2, columns = ['linear', 'lasso', 'ridge'])

## print coefficients
print(linear.coef_)
print(lasso.coef_)
print(ridge.coef_)


plt.plot(Y_test,lw=2, label = 'true')
plt.plot(linear_y, lw=2, label = 'linear')
plt.plot(lasso_y, lw=2, label = 'lasso')
plt.plot(ridge_y, color='brown', lw=1, label = 'ridge')
plt.xlim([1000,1200])
plt.ylim([-0.5,500001])
plt.ylabel('House_Price')
plt.legend()



'''
plt.plot(lasso_alpha['alpha'],lasso_alpha['R^2'])
plt.xlabel("Lasso_Lambda")
plt.ylabel("R^2")

plt.plot(ridge_alpha['alpha'],ridge_alpha['R^2'])
plt.xlabel("Ridge_Lambda")
plt.ylabel("R^2")

plt.plot(lasso_alpha_1['alpha'],lasso_alpha_1['MSE'])
plt.xlabel("Lasso_Lambda")
plt.ylabel("MSE")

plt.plot(ridge_alpha_1['alpha'],ridge_alpha_1['MSE'])
plt.xlabel("Ridge_Lambda")
plt.ylabel("MSE")
'''

## barchart for R^2
y_pos = np.arange(len(test_perform.columns))
plt.bar(y_pos, height = [test_perform.iloc[0,0], test_perform.iloc[0,1], test_perform.iloc[0,2]] )
plt.xticks(y_pos, ['linear', 'lasso', 'ridge'])
plt.ylabel('R^2')

y_pos = np.arange(len(train_perform.columns))
plt.bar(y_pos, height = [train_perform.iloc[0,0], train_perform.iloc[0,1], train_perform.iloc[0,2]] )
plt.xticks(y_pos, ['linear', 'lasso', 'ridge'])
plt.ylabel('R^2')

## histogram for residual 
r1 = Y_test-linear_y
r2 = Y_test-lasso_y
r3 = Y_test-ridge_y
r1 = r1.to_numpy()
r2 = r2.to_numpy()
r3 = r3.to_numpy()

plt.hist(r1, bins = 40)
plt.xlabel('Ytest-Ypred')
plt.ylabel('frequency')
plt.title('linear residual plot')

plt.hist(r2, bins = 40)
plt.xlabel('Ytest-Ypred')
plt.ylabel('frequency')
plt.title('lasso residual plot')

plt.hist(r3, bins = 40)
plt.xlabel('Ytest-Ypred')
plt.ylabel('frequency')
plt.title('ridge residual plot')

## coefficients
coef = pd.DataFrame(columns=['variable','linear_coef','lasso_coef','ridge_coef'])
coef['variable'] = ['Median_Income', 'Median_Age', 'Tot_Rooms',
       'Tot_Bedrooms', 'Population', 'Households', 'Latitude', 'Longitude',
       'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego',
       'Distance_to_SanJose', 'Distance_to_SanFrancisco',
       'income per working population', 'bed per house', 'h/p',
       'Avg_Distance']
coef['linear_coef']=linear.coef_[0]
coef['lasso_coef']=lasso.coef_
coef['ridge_coef']=ridge.coef_[0]
coef = coef.round(3)





