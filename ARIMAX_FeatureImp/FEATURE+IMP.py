
# coding: utf-8

# In[485]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import re

get_ipython().magic('matplotlib inline')
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10


# In[86]:

#specify the columns to use
cols_use_load = ['Date', 'Hour', 'DA_DEMD', 'DryBulb', 'DewPnt', 'Day']
cols_use_price = ['Date', 'Hour', 'DA_DEMD', 'DryBulb', 'DewPnt', 'DA_LMP', 'Day', 'DA_EC', 'DA_CC', 'DA_MLC']


# ### Load Data

# In[87]:

#2012-14 dataset
data1 = pd.read_csv('loadData1.csv', usecols=cols_use_price, skiprows=range(1415,1439))
data2 = pd.read_csv('loadData2.csv', usecols=cols_use_price)
data3 = pd.read_csv('loadData3.csv', usecols=cols_use_price)


# ### Feature Engineering

# In[88]:

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


# In[89]:

#append year wise data together to one single file
tot_data = data1.append([data2], ignore_index=True)


# In[90]:

tot_data['Date'] = tot_data['Date'].apply(convert_string_to_datetime2)
tot_data['Date'] = pd.to_datetime(tot_data['Date'])


# In[91]:

#function to get day from date
def get_day_from_date(date_time_obj):
    return int(date_time_obj.date().day)

#function to get month from date
def get_month_from_date(date_time_obj):
    return int(date_time_obj.date().month)


# In[92]:

tot_data['Day_Date'] = tot_data['Date'].apply(get_day_from_date)


# In[93]:

tot_data.head()


# #### Add Target Labels - Next Hour Data

# In[94]:

tot_data_labels = list(tot_data['DA_LMP'][1:])
tot_data_labels.append(data3['DA_LMP'][0])


# In[95]:

tot_data = tot_data.reset_index(drop=True)
print("Done")


# In[96]:

tot_data['TARGET_DA_LMP'] = tot_data_labels


# #### Add previous day same hour price
# This should be more useful for very short term than prev week

# In[97]:

prev_day_da_lmp_targ = tot_data['DA_LMP'][1:-24]
prev_day_da_lmp_targ = prev_day_da_lmp_targ.reset_index(drop=True)

prev_day_da_demd_targ = tot_data['DA_DEMD'][1:-24]
prev_day_da_demd_targ = prev_day_da_demd_targ.reset_index(drop=True)

prev_day_dry_bulb_targ = tot_data['DryBulb'][1:-24]
prev_day_dry_bulb_targ = prev_day_dry_bulb_targ.reset_index(drop=True)

temp_df = pd.DataFrame({
    'prev_day_da_lmp': prev_day_da_lmp_targ,
    'prev_day_da_demd': prev_day_da_demd_targ,
    'prev_day_dry_bulb': prev_day_dry_bulb_targ,
    'actual_da_lmp': tot_data['DA_LMP'][0:-25],
    'actual_da_demd': tot_data['DA_DEMD'][0:-25],
    'actual_dry_bulb': tot_data['DryBulb'][0:-25]
})

temp_df['chg_in_da_demd'] = temp_df['actual_da_demd'] - temp_df['prev_day_da_demd']
temp_df['chg_in_dry_bulb'] = temp_df['actual_dry_bulb'] - temp_df['prev_day_dry_bulb']
temp_df['chg_in_da_lmp'] = temp_df['actual_da_lmp'] - temp_df['prev_day_da_lmp']


# In[98]:

tot_data = tot_data[24:-1]
tot_data = tot_data.reset_index(drop=True)

hours_df = pd.DataFrame({
    'hour0': list(tot_data['DA_LMP'][0:-4]),
    'hour1': list(tot_data['DA_LMP'][1:-3]),
    'hour2': list(tot_data['DA_LMP'][2:-2]),
    'hour3': list(tot_data['DA_LMP'][3:-1])
})

hours_df = hours_df.reset_index(drop=True)

tot_data = tot_data[4:]
tot_data = tot_data.reset_index(drop=True)
tot_data['Hour0'] = hours_df['hour0']
tot_data['Hour1'] = hours_df['hour1']
tot_data['Hour2'] = hours_df['hour2']
tot_data['Hour3'] = hours_df['hour3']

tot_data = tot_data.reset_index(drop=True)
# In[99]:

tot_data = pd.merge(tot_data, temp_df[['prev_day_da_demd','prev_day_da_lmp', 'prev_day_dry_bulb',
                           'chg_in_da_demd', 'chg_in_da_lmp', 'chg_in_dry_bulb']], how='inner', 
                    left_index=True, right_index=True)
len(tot_data)


# #### Add Previous Week Next Hour Data
# Maybe useful for short-term, not so much for very short term forecasting
#da-lmp prev week
prev_week_da_lmp = list(tot_data['DA_LMP'][1:-168])

#da-demd prev week
prev_week_da_demd = list(tot_data['DA_DEMD'][1:-168])len(prev_week_da_lmp)tot_data = tot_data[168:-1]
tot_data = tot_data.reset_index(drop=True)

tot_data['Prev_Week_DA_DEMD'] = prev_week_da_demd
tot_data['Prev_Week_DA_LMP'] = prev_week_da_lmp
# #### Add daily means to the data

# In[100]:

mean_values = tot_data.groupby('Date', as_index=False).mean()[['Date','DA_DEMD', 'DA_LMP', 'DryBulb', 'DewPnt']]
temp = pd.merge(tot_data, mean_values, how='inner', left_on='Date', right_on='Date', 
                suffixes=('_actual_value','_daily_mean'))
tot_data = temp


# #### Add crude oil prices

# In[101]:

crude_oil = pd.read_csv('CurdeOil.csv', delimiter='\t')
crude_oil = crude_oil[::-1]
crude_oil = crude_oil.reset_index(drop=True)
crude_oil = crude_oil[2:]
crude_oil = crude_oil.reset_index(drop=True)

def get_num_from_vol_str(string):
    match = re.search(r"(\d+)", string)
    if match!=None:
        return float(match.group(1))
    
crude_oil = crude_oil[['Date', 'Price', 'Vol.']]
crude_oil['Date'] = crude_oil['Date'].apply(convert_string_to_datetime)
crude_oil['Date'] = pd.to_datetime(crude_oil['Date'])

crude_oil['Crude_Oil_Price'] = crude_oil['Price']
crude_oil['Crude_Oil_Vol'] = crude_oil['Vol.']

crude_oil = crude_oil.drop('Price', axis=1)
crude_oil = crude_oil.drop('Vol.', axis=1)

temp = pd.merge(tot_data, crude_oil, how='outer', left_on='Date', right_on='Date', suffixes=('_X','_crude_oil'))
temp.reset_index(drop=True)
temp['Crude_Oil_Price'] = temp['Crude_Oil_Price'].fillna(method='bfill')
temp['Crude_Oil_Vol'] = temp['Crude_Oil_Vol'].fillna(method='bfill')
temp = temp[0:len(tot_data)]
tot_data = temp

tot_data['Crude_Oil_Vol'] = tot_data['Crude_Oil_Vol'].apply(get_num_from_vol_str)

'''
temp2 = pd.merge(test, crude_oil, how='outer', left_on='Date', right_on='Date', suffixes=('_X','_crude_oil'))
temp2.reset_index(drop=True)
temp2['Crude_Oil_Price'] = temp2['Crude_Oil_Price'].fillna(method='bfill')
temp2['Crude_Oil_Vol'] = temp2['Crude_Oil_Vol'].fillna(method='bfill')
test = temp2[0:4271]

test['Crude_Oil_Vol'] = test['Crude_Oil_Vol'].apply(get_num_from_vol_str)
'''


# #### Add Natural Gas Price

# In[102]:

natural_gas = pd.read_csv('natural_gas.csv')


# In[103]:

natural_gas.tail()


# In[104]:

natural_gas['Date'] = natural_gas['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())


# In[105]:

natural_gas.rename(columns={
    'Open': 'Natural_Gas_Price_Open',
    'Last': 'Natural_Gas_Price_Close'
}, inplace=True)


# In[106]:

natural_gas['Date'] = pd.to_datetime(natural_gas['Date'])


# In[107]:

tot_data = pd.merge(left=tot_data, right=natural_gas[['Date', 'Natural_Gas_Price_Open', 'Natural_Gas_Price_Close']],
        how='left', on='Date')


# In[108]:

tot_data['Natural_Gas_Price_Open'].isnull().value_counts().plot(kind='bar')


# In[109]:

tot_data.fillna(method='bfill', inplace=True)


# In[110]:

tot_data.head()


# In[111]:

tot_data.to_csv('final_complete_data.csv', index=False)


# In[ ]:




#  # Load the Data
#  just load the saved csv
#  no need to run the previous blocks again

# In[970]:

tot_data = pd.read_csv('../final_complete_data.csv')


# In[971]:

tot_data.head()


# #### Helper Functions

# In[972]:

tot_data['Date'] = tot_data['Date'].apply(convert_string_to_datetime4)
tot_data['Date'] = pd.to_datetime(tot_data['Date'])


# In[973]:

tot_data.columns.values


# In[979]:

tot_data['Month'] = tot_data['Date'].apply(lambda x: x.month)
tot_data['Week'] = tot_data['Date'].apply(lambda x: x.week)


# #### Deal with -ve target values

# In[980]:

X = tot_data


# In[981]:

for i in range(len(X['TARGET_DA_LMP'])):
    if(X['TARGET_DA_LMP'][i] < 1):
        X['TARGET_DA_LMP'][i] = (X['TARGET_DA_LMP'][i-1]+X['TARGET_DA_LMP'][i+1])/2


# Date Time Index

# In[982]:

X.set_index(['Date'], drop=True, inplace=True)


# In[983]:

X.head()


# #### ------------- DATA PREP OVER --------------------

# ## FEATURE IMPORTANCE STUDY

# In[984]:

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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


# In[985]:

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression


# In[986]:

train_feature_imp = X[X.Day_Date<24]
test_feature_imp = X[X.Day_Date>=24]

#train1 = train.query('Day_Date < 2')
#train2 = train.query('Day_Date > 9 & Day_Date < 12')
#train3 = train.query('Day_Date > 18 & Day_Date < 22')
#train = train1.append(train2)
#train = train.append(train3)
#test = test1.append(test2)

train_feature_imp = train_feature_imp.reset_index(drop=True)
test_fetaur_imp = test_feature_imp.reset_index(drop=True)

print(len(train_feature_imp))
print(len(test_feature_imp))


# In[987]:

train_feature_imp.head()


# In[988]:

cols_feature_imp = X.columns.difference(['TARGET_DA_LMP']).values


# In[1074]:

cols_feature_imp2 = ['DA_LMP_actual_value', 'chg_in_da_demd', 'chg_in_da_lmp', 'prev_day_da_lmp', 'DA_LMP_daily_mean',
 'Hour', 'DA_DEMD_actual_value', 'DryBulb_actual_value', 'DryBulb_daily_mean', 
 'Crude_Oil_Price', 'Day', 'Day_Date', 'Month']


# In[1075]:

train_feature_imp_x = train_feature_imp[cols_feature_imp2]
train_feature_imp_y = pd.DataFrame(train_feature_imp['TARGET_DA_LMP'])
test_feature_imp_x = test_feature_imp[cols_feature_imp2]
test_feature_imp_y = pd.DataFrame(test_feature_imp['TARGET_DA_LMP'])


# In[1076]:

scaler = StandardScaler()
train_feature_imp_scaled = scaler.fit_transform(train_feature_imp_x)
test_feature_imp_scaled = scaler.fit_transform(test_feature_imp_x)


# ##### ElasticNet

# In[1101]:

enet_model = ElasticNet(alpha=4, l1_ratio=0.3)
enet_model = enet_model.fit(train_feature_imp_scaled, train_feature_imp_y)
enet_model.coef_


# ##### Mutual Info

# In[1078]:

mutual_info = mutual_info_regression(train_feature_imp_scaled, train_feature_imp_y.values, n_neighbors=100,
                                    random_state=333)


# In[1079]:

mutual_info


# ##### Plots
# <code>
# ['DA_LMP_actual_value', 'chg_in_da_demd', 'chg_in_da_lmp', 'prev_day_da_lmp', 'DA_LMP_daily_mean',
#  'Hour', 'DA_DEMD_actual_value', 'DryBulb_actual_value', 'DryBulb_daily_mean', 
#  'Crude_Oil_Price']
# </code>
# <br>
# <code>
# ['Previous Hour DA LMP', 'Previous Day Change in DA DEMAND', 'Previous Day Change in DA LMP', 'Previous Day DA LMP',
#  'Daily Average DA LMP', 'Hour Indicator', 'Previous Hour DA DEMAND', 'Hourly Dry Bulb Temp', 
#  'Daily Average Dry Bulb Temp', 'Crude Oil Prices', 'Weekday Indicator', 'Date Part Day Indicator', 'Month Indicator']
# </code>
# 

# In[1080]:

sanitised_colnames_set = tuple(['Previous Hour DA LMP', 'Previous Day Change in DA DEMAND', 
                                'Previous Day Change in DA LMP', 'Previous Day DA LMP',
 'Daily Average DA LMP', 'Hour Indicator', 'Previous Hour DA DEMAND', 'Hourly Dry Bulb Temp', 
 'Daily Average Dry Bulb Temp', 'Crude Oil Prices', 'Weekday Indicator', 
                                'Date Part Day Indicator', 'Month Indicator'])


# In[1109]:

#plt.plot(lasso_model.coef_, linewidth=2, linestyle='--', label='LASSO', color='blue')
plt.plot(mutual_info, linewidth=2, linestyle='-', label='Mutual Information Value', color='red')
plt.plot(enet_model.coef_, linewidth=2, linestyle='--', label='ElasticNet Coefficients', color='green')
plt.ylim(-3, 7)
plt.xticks(np.arange(len(train_feature_imp_x.columns.values)), 
           sanitised_colnames_set, rotation=90, fontsize=12)
plt.axhline(y=0, color='black')
plt.xlabel("Variable Names", {'fontsize':23})
plt.ylabel("Mutual Information Value / Coefficient Values ",{'fontsize':23})
plt.yticks(fontsize=20, rotation=0)
plt.title("Feature Importances", {'fontsize':30})
plt.legend(loc=1, fontsize=23)
plt.gcf()
plt.savefig("../NEWPLOTS_Feb19/FeatureImportancePlot.jpg", format='jpg', bbox_inches='tight', dpi=600)

## ARIMAX#### Train Test Split
Different method used because model is time series modelX.set_index('Date', drop=True, inplace=True)train = X['2012-01-02':'2012-12-31']test = X['2013-01-01':'2013-12-31']plt.plot(train['TARGET_DA_LMP'])#### Evaluation Metricsfrom sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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
    return (max_err, min_err, ci, sd, var)#### Stationarity testfrom statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMAdef test_stationarity(timeseries):
    
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
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)test_stationarity(train['TARGET_DA_LMP'])test_stationarity(test['TARGET_DA_LMP'])### MODELcond_to_use = ['DA_LMP_actual_value', 'chg_in_da_demd', 'chg_in_da_lmp', 'prev_day_da_lmp', 'DA_LMP_daily_mean',
              'Hour', 'DA_DEMD_actual_value', 'DryBulb_actual_value', 'DryBulb_daily_mean', 'Crude_Oil_Price']
train_x = train[cond_to_use]
test_x = test[cond_to_use]
train_y = train[['TARGET_DA_LMP']]
test_y = test[['TARGET_DA_LMP']]train_std_scaler2 = StandardScaler()
train_std_scaler2_y = StandardScaler()

test_std_scaler2 = StandardScaler()
test_std_scaler2_y = StandardScaler()

train_x_scaled = pd.DataFrame(train_std_scaler2.fit_transform(train_x)).set_index(train_x.index)
train_y_scaled = pd.DataFrame(train_std_scaler2_y.fit_transform(train_y)).set_index(train_y.index)

test_x_scaled = pd.DataFrame(test_std_scaler2.fit_transform(test_x)).set_index(test_x.index)
test_y_scaled = pd.DataFrame(test_std_scaler2_y.fit_transform(test_y)).set_index(test_y.index)### TRAINmodel = ARIMA(train_y_scaled, exog=train_x_scaled, order=(0,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()residuals.plot(kind='kde')
plt.show()
print(residuals.describe())TRAIN RESULTSprint('MAPE:', mean_absolute_percentage_error(train['TARGET_DA_LMP'].values, 
                                              train_std_scaler2_y.inverse_transform(model_fit.fittedvalues)))
print('MAE:', mean_absolute_error(train['TARGET_DA_LMP'].values, 
                                  train_std_scaler2_y.inverse_transform(model_fit.fittedvalues)))### Iterative Train - Test Process
Daily basis <br>
NEED to convert this to a monthly method -> This will allow trendline comparisonfrom collections import defaultdicthistory_x, history_y = train_x, train_y
prediction_results_history = []
day_wise_mape_history = []
day_wise_mae_history = []monthly_preds = defaultdict(list)
monthly_mape =  defaultdict(list)

weekly_preds = defaultdict(list)
weekly_mape =  defaultdict(list)day_count = 1

for i in range(0, len(test_x), 24):
    
    test_data_x = test_x[i:i+24]
    test_data_y = test_y[i:i+24]
    
    model = ARIMA(history_y, exog=history_x, order=(0,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(exog=test_data_x, steps=24)
    
    mape_day = mean_absolute_percentage_error(test_data_y.values, 
                                             output[0])
    mae_day = mean_absolute_error(test_data_y.values, 
                                  output[0])
    print('DAY '+str(day_count)+' MAPE: ', mape_day)
    print('DAY '+str(day_count)+' MAE: ', mae_day, end='\n\n\n')
    
    ### save results to list
    prediction_results_history.extend(output[0])
    history_x.append(test_data_x)
    history_y.append(test_data_y)
    day_wise_mape_history.append(mape_day)
    day_wise_mae_history.append(mae_day)
    
    ### save monthly and weekly results to dict
    month = test_x.index[i].month
    week = test_x.index[i].week
    
    monthly_mape[month].append(mape_day)
    weekly_mape[week].append(mape_day)
    
    monthly_preds[month].append(output[0])
    weekly_preds[week].append(output[0])
    
    
    day_count+=1
    if day_count==365:
        break ### TEST Performancesnp.mean(day_wise_mape_history)[np.mean(day_wise_mape_history[len(day_wise_mape_history)-31:-1])]np.mean(day_wise_mape_history[0:30])np.mean(day_wise_mae_history)### LASSOfrom sklearn.linear_model import Lasso, Ridge, ElasticNetscaler = StandardScaler()
train_scaled = scaler.fit_transform(train[train.columns.difference(['TARGET_DA_LMP'])])train_y = train['TARGET_DA_LMP']Lasso()lasso_model = Lasso(alpha=1)
lasso_model = lasso_model.fit(train_scaled, train_y)
lasso_model.coef_ridge_model = Ridge(alpha=0.7)
ridge_model = ridge_model.fit(train_scaled, train_y)
ridge_model.coef_enet_model = ElasticNet(alpha=0.2, l1_ratio=0.1)
enet_model = enet_model.fit(train_scaled, train_y)
enet_model.coef_plt.plot(lasso_model.coef_, linewidth=2, linestyle='--', label='LASSO', color='blue')
plt.plot(ridge_model.coef_, linewidth=2, linestyle='-', label='Ridge', color='green')
plt.plot(enet_model.coef_, linewidth=2, linestyle='-', label='ElasticNet', color='red')
plt.ylim(-14, 14)
plt.xticks(np.arange(len(train.columns.difference(['TARGET_DA_LMP']).values)), 
           tuple(train.columns.difference(['TARGET_DA_LMP']).values), rotation=90)
plt.legend()
plt.show()tuple(train.columns.difference(['TARGET_DA_LMP']).values)train.columns.difference(['TARGET_DA_LMP']).values