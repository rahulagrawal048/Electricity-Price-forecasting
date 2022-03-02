
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().magic('matplotlib inline')
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10


#  Load the Data

# In[2]:

#specify the columns to use
cols_use_load = ['Date', 'Hour', 'DA_DEMD', 'DryBulb', 'DewPnt', 'Day']
cols_use_price = ['Date', 'Hour', 'DA_DEMD', 'DryBulb', 'DewPnt', 'DA_LMP', 'Day', 'DA_EC', 'DA_CC', 'DA_MLC']

#2012-14 dataset
data1 = pd.read_csv('../loadData1.csv', usecols=cols_use_price, skiprows=range(1415,1439))
data2 = pd.read_csv('../loadData2.csv', usecols=cols_use_price)
data3 = pd.read_csv('../loadData3.csv', usecols=cols_use_price)
# Helper Functions

# In[4]:

#functions for converting string to datetime objects
def convert_string_to_datetime(string):
    datetime_obj = datetime.strptime(string, "%d-%b-%y")
    return datetime_obj.date()

#format is different for 2012,2013,2014
def convert_string_to_datetime2(string):
    datetime_obj = datetime.strptime(string, "%m/%d/%Y")
    return datetime_obj.date()

#format is different for 2015
def convert_string_to_datetime3(string):
    datetime_obj = datetime.strptime(string, "%d/%m/%Y")
    return datetime_obj.date()

#format for natural Gas prices
def convert_string_to_datetime4(string):
    datetime_obj = datetime.strptime(string, "%Y-%m-%d")
    return datetime_obj.date()


# ### Load Complete Data

# In[45]:

tot_data = pd.read_csv('../final_complete_data.csv')


# In[56]:

#tot_data['Date'] = tot_data['Date'].apply(convert_string_to_datetime4)
tot_data['Date'] = pd.to_datetime(tot_data['Date'])


# Deal with missings

# In[58]:

X = tot_data.copy()


# In[59]:

for i in range(len(X['TARGET_DA_LMP'])):
    if(X['TARGET_DA_LMP'][i] < 1):
        X['TARGET_DA_LMP'][i] = (X['TARGET_DA_LMP'][i-1]+X['TARGET_DA_LMP'][i+1])/2


# Train Test Split

# In[60]:

X.set_index('Date', drop=True, inplace=True)


# In[61]:

train = X['2012-01-02':'2012-12-31']


# In[62]:

test = X['2013-01-01':'2013-12-31']


# In[63]:

plt.plot(train['TARGET_DA_LMP'])


# ### Stationarity test

# In[64]:

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


# In[65]:

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#function to calculate mape
def mean_absolute_percentage_error(y_pred, y_true): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def point_absolute_percentage_error(y_pred, y_true):
    return np.abs((y_true - y_pred) / y_true) * 100

#GIVE Everything you can imagine of function - SD, VAR, MIN and MAX ERROR, CI
def get_CI_VAR_SD(error_list):
    t = stats.t.ppf(1-0.025, len(error_list)-1)
    max_err = np.mean(error_list) + (t * (np.std(error_list)/np.sqrt(len(error_list))))
    min_err = np.mean(error_list) - (t * (np.std(error_list)/np.sqrt(len(error_list))))
    ci = ((max_err - np.mean(error_list))/np.mean(error_list))*100
    sd = np.std(error_list)
    var = np.var(error_list)
    return (max_err, min_err, ci, sd, var)


# In[173]:

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=100)
    rolstd = pd.rolling_std(timeseries, window=100)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, maxlag=21)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[174]:

test_stationarity(train['TARGET_DA_LMP'])


# In[175]:

test_stationarity(train['TARGET_DA_LMP'][0:24*31])


# ### MODEL

# In[78]:

cond_to_use = ['DA_LMP_actual_value', 'chg_in_da_demd', 'chg_in_da_lmp', 'prev_day_da_lmp', 'DA_LMP_daily_mean',
              'Hour', 'DA_DEMD_actual_value', 'DryBulb_actual_value', 'DryBulb_daily_mean', 'Crude_Oil_Price']
train_x = train[cond_to_use]
test_x = test[cond_to_use]
train_y = train['TARGET_DA_LMP']
test_y = test['TARGET_DA_LMP']


# ### TRAIN

# In[79]:

model = ARIMA(train_y, exog=train_x, order=(0,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[80]:

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()


# In[81]:

residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# TRAIN RESULTS

# In[82]:

print('MAPE:', mean_absolute_percentage_error(train['TARGET_DA_LMP'].values, model_fit.fittedvalues))
print('MAE:', mean_absolute_error(train['TARGET_DA_LMP'].values, model_fit.fittedvalues))


# ### Iterative Train - Test Process
# Daily basis <br>
# NEED to convert this to a monthly method -> This will allow trendline comparison

# In[86]:

from collections import defaultdict


# In[141]:

history_x, history_y = train_x, train_y
prediction_results_history = []
day_wise_mape_history = []
day_wise_mae_history = []


# In[142]:

monthly_preds = defaultdict(list)
monthly_mape =  defaultdict(list)

weekly_preds = defaultdict(list)
weekly_mape =  defaultdict(list)


# In[143]:

day_count = 1
for i in range(0, len(test_x), 24):
    print(i)
    print(test_x.index[i])
    model = ARIMA(history_y, exog=history_x, order=(0,0,0))
    model_fit = model.fit(disp=0)
    if len(test_x[i:i+24]) == 23:
        output = model_fit.forecast(exog=test_x[i:i+23], steps=23)
    else:
        output = model_fit.forecast(exog=test_x[i:i+24], steps=24)
    mape_day = mean_absolute_percentage_error(test_y[i:i+24].values, output[0])
    mae_day = mean_absolute_error(test_y[i:i+24].values, output[0])
    print('DAY '+str(day_count)+' MAPE: ', mape_day)
    print('DAY '+str(day_count)+' MAE: ', mae_day, end='\n\n\n')
    ### save results to list
    prediction_results_history.extend(output[0])
    history_x.append(test_x[i:i+24])
    history_y.append(test_y[i:i+24])
    day_wise_mape_history.append(mape_day)
    day_wise_mae_history.append(mae_day)
    
    month = test_x.index[i].month
    week = test_x.index[i].week
    
    monthly_mape[month].append(mape_day)
    weekly_mape[week].append(mape_day)
    
    monthly_preds[month].append(output[0])
    weekly_preds[week].append(output[0])
    
    day_count+=1
    if day_count==366:
        break


# In[144]:

len(prediction_results_history)


# In[155]:

test['Predicted_Values'] = prediction_results_history


# In[158]:

test.shape


# In[157]:

test.head()


# In[160]:

test.to_csv('prediction_results_exogenous_arima.csv', index=True)


#  ### TEST Performances

# In[85]:

np.mean(day_wise_mape_history)


# In[32]:

np.mean(day_wise_mape_history[len(day_wise_mape_history)-31:-1])


# In[33]:

np.mean(day_wise_mape_history[0:30])


# In[34]:

np.mean(day_wise_mae_history)


# In[122]:

len(prediction_results_history)/2


# In[123]:

test_x.shape


# In[164]:

[0 for i in range(12)]+[np.mean(i) for i in monthly_mape.values()]


# Average Monthly MAPE

# In[153]:

plt.bar(np.arange(12),[np.mean(i) for i in monthly_mape.values()])
plt.xticks(np.arange(12), ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                          'November', 'December'))
plt.ylabel('MAPE')
plt.show()

