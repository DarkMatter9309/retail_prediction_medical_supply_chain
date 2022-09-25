import numpy as np
import matplotlib.pyplot as plot 
import pandas as pd  
import seaborn as sb
import matplotlib.image as mpimg
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import scale

# Reading data from cloud
data_set_path = './sales.csv'
df = pd.read_csv(data_set_path, low_memory=False)
#print dataset
print(df.head())

#find corr
print(df.corr())

#remove columns that are not impacting
df=df[['Product', 'Date', 'Sales','Customers']]

#Simple linear regression model to beign
def monthly_sales(data):
    monthly_data = data.copy()
    monthly_data.Date = monthly_data.Date.apply(lambda x: str(x)[3:])
    # print(monthly_data.Date)
    monthly_data = monthly_data.groupby(['Product','Date'])['Sales'].sum().reset_index()
    # monthly_data.head()
    # monthly_data.Date = pd.to_datetime(monthly_data.Date)
    return monthly_data

monthly_df = monthly_sales(df)
#filter for product
monthly_df = monthly_df[monthly_df['Product']==1]
monthly_df[['Month','Year']] = monthly_df['Date'].str.split('/',expand=True)
monthly_df['Year'] = '20'+monthly_df['Year']
monthly_df = monthly_df.drop('Date',axis=1)
y = monthly_df['Sales']
y=y.values
new_df = monthly_df.drop(['Sales'],axis=1)
new_df = new_df.values
new_df_train, new_df_test, value_train, value_test = train_test_split(new_df, y, test_size=0.1, random_state=1)

#simplest supervised machine learning model
#linear regression
#apply ols regression algorithm on training data
olsmodel = linear_model.LinearRegression()
olsmodel.fit(new_df_train, value_train)
print(olsmodel.intercept_)
print(olsmodel.coef_)

print("")
print("")
#Caluculate predictions
print("Linear Regression on training data:")
print("")
value_train_prediction = olsmodel.predict(new_df_train)
print("Mean squared error = ")
print(mean_squared_error(value_train, value_train_prediction,squared=False))
print("")

print("R2 score")
print(abs(r2_score(value_train, value_train_prediction)))
print("")

print("Absolute percentage error")
print(mean_absolute_percentage_error(value_train, value_train_prediction))
print("")
print("")
#Caluculate predictions
print("Linear Regression on testing data:")
value_prediction = olsmodel.predict(new_df_test)
print("Mean squared error = ")
print(mean_squared_error(value_test, value_prediction,squared=False))
print("")

print("R2 score")
print(abs(r2_score(value_test, value_prediction)))
print("")

print("Absolute percentage error")
print(mean_absolute_percentage_error(value_test, value_prediction))

print("")
print("")
#This problem is a multivariate time series model:
    #The prediction depends on multiple variables other than time

#Time series decomposition:
    #Seasonality: recurring movement
    #Trend: upward or downward
    #Noise: cannot be explained by trend or seasonality
#decomposition methods available to observe the above properties

#Auto correlation:
    #Positive autocorrelation: High value now will give a high value in future
    #Negative autocorrelation: opposite
#ACF function can be used to determine


#Stationary vs Non-Stationary:
    #Stationary: A time series is stationary if it has no trend
#Dicky-Fullers test

#Differencing:
    #Remove seasonal data to identify trends

#Random forest
from sklearn.ensemble import RandomForestRegressor

# fit the model
my_rf = RandomForestRegressor()
my_rf.fit(new_df_train, value_train)

# predict on the same period
preds_train = my_rf.predict(new_df_train)

test_preds = my_rf.predict(new_df_test)

#Caluculate predictions
print("Random forest on training data:")
print("Mean squared error = ")
print(mean_squared_error(value_train, preds_train,squared=False))
print("")

print("R2 score")
print(abs(r2_score(value_train, preds_train)))
print("")

print("Absolute percentage error")
print(mean_absolute_percentage_error(value_train, preds_train))

print("")
print("")
#Caluculate predictions for test data
print("Random forest on testing data:")
print("Mean squared error = ")
print(mean_squared_error(value_test, test_preds,squared=False))
print("")

print("R2 score")
print(abs(r2_score(value_test, test_preds)))
print("")

print("Absolute percentage error")
print(mean_absolute_percentage_error(value_test, test_preds))


print("")
print("")
print("")







