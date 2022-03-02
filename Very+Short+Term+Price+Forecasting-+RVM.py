
# coding: utf-8

# ## Very Short Term Price Forecasting - New England Dataset
# - Very Short Term == next hour price
# - For comparison to a research publication
# - Main objective is to optimize method for forecasting short term prices i.e. Next Day Prices
# - Error Metrics are Mean Absolute Error and Mean Absolute Percentage Error
# - Use RVM and Stack it with other Models to improve Generalization Error

# In[345]:

#import required libraries
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from scipy import stats

#import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10


# In[350]:

import warnings 
warnings.filterwarnings('ignore')


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


# ### ------------ Data Prep Ends Here ---------------

# ### Split Data to Train and Test set

# In[112]:

from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler


# In[113]:

X = tot_data


# In[114]:

sns.distplot(X['TARGET_DA_LMP'])


# In[115]:

for i in range(len(X['TARGET_DA_LMP'])):
    if(X['TARGET_DA_LMP'][i] < 1):
        X['TARGET_DA_LMP'][i] = (X['TARGET_DA_LMP'][i-1]+X['TARGET_DA_LMP'][i+1])/2


# In[116]:

plt.plot(X['DA_LMP_actual_value'])
plt.xlabel("Time(Hours)", {'fontsize':30})
plt.ylabel("Electricity Price Values",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Electricity Prices Raw Data", {'fontsize':30})
plt.savefig("plot_electricity_prices.jpg", format='jpg', bbox_inches='tight', dpi=600)


# - Split Method - **23 Days** as **Training Set** for each month and remaining as **Test Set**

# In[117]:

#train1 = X.query('Day_Date <24 & Date < datetime(2013,1,1)')#[(X.Day_Date <24) and (X.Date > )]
#train2 = X.query('Day_Date <24 & Date > datetime(2013,2,1)')#X[X['Date'] >= datetime(2013,1,1)]
#test1 = X.query('Day_Date >24 & Date < datetime(2013,1,1)')
#test2 = X.query('Day_Date >24 & Date > datetime(2013,2,1) & Date < datetime(2013,12,1)')
#[X.Day_Date>=24 and X.Date >= datetime(2013,1,1)]

train = X[X.Day_Date<24]
test = X[X.Day_Date>=24]

#train1 = train.query('Day_Date < 2')
#train2 = train.query('Day_Date > 9 & Day_Date < 12')
#train3 = train.query('Day_Date > 18 & Day_Date < 22')
#train = train1.append(train2)
#train = train.append(train3)
#test = test1.append(test2)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print(len(train))
print(len(test))


# ### Selecting Specific Columns and Feature Scaling

# In[593]:

#conditions = [x for x in train.columns if 'Date' 
#              not in x and 'TARGET_DA_LMP' not in x]
#test_conditions = [x for x in train.columns if 'Date' 
#             not in x and 'TARGET_DA_LMP' not in x and 'DewPnt_daily_mean' not in x 
#             and 'DryBulb_daily_mean' not in x and 'Prev_Week_DA_DEMD' not in x 
#                and 'DA_DEMD_daily_mean' not in x and 'DA_LMP_daily_mean' not in x 
#               ]

cond_to_use = ['DA_LMP_actual_value', 'chg_in_da_demd', 'chg_in_da_lmp', 'prev_day_da_lmp', 'DA_LMP_daily_mean',
              'Hour', 'DA_DEMD_actual_value', 'DryBulb_actual_value', 'DryBulb_daily_mean', 
               'Crude_Oil_Price']

train_x = train[cond_to_use]
train_y = pd.DataFrame(train['TARGET_DA_LMP'])
test_x = test[cond_to_use]
test_y = pd.DataFrame(test['TARGET_DA_LMP'])


# In[594]:

std_scaler = StandardScaler()
train_x_scaled = std_scaler.fit_transform(train_x)
test_x_scaled = std_scaler.fit_transform(test_x)
train_y_scaled = std_scaler.fit_transform(train_y)


# ### MAPE Function and Other Error Metrics

# In[667]:

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


# Make custom scoring function

# In[670]:

from sklearn.metrics.scorer import make_scorer


# In[698]:

def inverse_scaled_mae_scoring(true_y_scaled, pred_y_scaled):
    true = std_scaler.inverse_transform(true_y_scaled)
    pred = std_scaler.inverse_transform(pred_y_scaled)
    return mean_absolute_error(true, pred)
    
inversed_mae_score = make_scorer(inverse_scaled_mae_scoring, greater_is_better=False)


# ### PRICE PREDICTIONS

# In[596]:

#import all models required from scikit learn

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from skbayes.rvm_ard_models import RVR


from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


# In[122]:

import timeit


# ### Function to save model to disk

# In[123]:

import pickle

def save_model(model_obj, filename):
    try:
        with open(filename, 'wb') as fid:
            pickle.dump(model_obj, fid) 
        print("Done")
    except Exception as e: print(e)


# In[129]:

def load_saved_pickle_model(fileanme):
    return pickle.load(open(filename, 'rb'))


# ### Random Forest

# In[604]:

params = {
    'max_depth': [13, 10, 7], #np.linspace(50,100,5, dtype=int),
    'n_estimators': [1000, 700], #np.linspace(500,2000,5, dtype=int)
    'bootstrap': [True]
}

#rf_reg_price = GridSearchCV(RandomForestRegressor(), params, n_jobs=-1, cv=7, scoring='neg_mean_squared_error')
rf_reg_price = RandomForestRegressor(max_depth=8, n_estimators=700, bootstrap=True)

start = timeit.default_timer()

rf_reg_price = rf_reg_price.fit(train_x_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)


try:
    print(rf_reg_price.best_params_)
    print("\n")
    print(rf_reg_price.best_score_)
except:
    print("Did Not Use GridSearch")


# #### Training Error and Test Error- MAPE

# In[605]:

#train mape
rf_preds_price_train = rf_reg_price.predict(train_x_scaled)
rf_preds_price_train = std_scaler.inverse_transform(rf_preds_price_train)
err = mean_absolute_percentage_error(rf_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
rf_preds_price = rf_reg_price.predict(test_x_scaled)
rf_preds_price = std_scaler.inverse_transform(rf_preds_price)
err = mean_absolute_percentage_error(rf_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)

point_err_rf = point_absolute_percentage_error(rf_preds_price, test_y['TARGET_DA_LMP'])
max_err_rf, min_err_rf, ci_rf, sd_rf, var_rf = get_CI_VAR_SD(point_err_rf)
print(max_err_rf, min_err_rf, ci_rf, sd_rf, var_rf)

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list_rf =[]
mae_list_rf =[]
rmse_list_rf =[]
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(days_list[i], np.sum(days_list[:i+2])*24)
    temp_list_rf.append(mean_absolute_percentage_error(
        rf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list_rf.append(mean_absolute_error(
        rf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list_rf.append(np.sqrt(mean_squared_error(
        rf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list_rf


# In[606]:

rf_preds_price

#saved model doesnt work well for some reason for RF
save_model(rf_reg_price, 'model_rf.pkl')
# #### RF Feature Importance Plot
features = pd.DataFrame()
features['feature'] = train_x.columns
features['importance'] = rf_reg_price.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))
# # Level 1 Learning

# ### Adaboosted RVM
# Make RVM as weak learner and then train many weak learners and apply ada boost to give weights
# 
# ###### Algorithm worked well but not like what was expected
# Improvements of upto 4-5% in MAPE from base classifiers but peaked around 8% MAPE and no further improvements
from sklearn.ensemble import AdaBoostRegressor

idx_list = [0,3000,6000,7000,10000]

#train adaboost with rvm in batches
models_list = list()

for i in range(0,len(idx_list)):
    rvr_ada = RVR(n_iter=1, kernel='poly', degree=1, gamma=1, tol=1, coef0=1, verbose=1)
    
    ada = AdaBoostRegressor(base_estimator=rvr_ada, 
                        n_estimators=50, learning_rate=0.01, random_state=3, loss='linear')
    if i<len(idx_list):
        ada = ada.fit(train_x_scaled[idx_list[i]:idx_list[i+1]], train_y_scaled[idx_list[i]:idx_list[i+1]])
        models_list.append(ada)
    else:
        ada = ada.fit(train_x_scaled[idx_list[i]:], train_y_scaled[idx_list[i]:])
        models_list.append(ada)
        
    ada_preds_price = ada.predict(test_x_scaled)
    ada_preds_price = std_scaler.inverse_transform(ada_preds_price)
    err = mean_absolute_percentage_error(ada_preds_price, test_y['TARGET_DA_LMP'])
    print("Test: ", err) rvr_ada = RVR(n_iter=1, kernel='linear', degree=1, gamma=1, tol=1, coef0=1, verbose=1)
rvr_ada = rvr_ada.fit(train_x_scaled[9000:], train_y_scaled[9000:])### train mape
ada_preds_price_train = ada.predict(train_x_scaled)
ada_preds_price_train = std_scaler.inverse_transform(ada_preds_price_train)
err = mean_absolute_percentage_error(ada_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
ada_preds_price = ada.predict(test_x_scaled)
ada_preds_price = std_scaler.inverse_transform(ada_preds_price)
err = mean_absolute_percentage_error(ada_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list =[]
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list.append(mean_absolute_error(
        ada_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
temp_list
# ### eXtreme Gradient Boosting XGBoost
params = {
    'max_depth': [7],
    'min_child_weight':[14],
    'n_estimators': [800],
    'learning_rate':[0.01],
    'reg_lambda':[3],
    'gamma': [0.01],
    'reg_alpha':[1e-5]
}

#xgb_reg_price = GridSearchCV(XGBRegressor(silent=0), params, n_jobs=-1, cv=10, 
                       #scoring='neg_mean_squared_error') 
xgb_reg_price = XGBRegressor(silent=0, max_depth=7, min_child_weight=14, n_estimators=800, learning_rate=0.01,
                            reg_lambda=3, gamma=0.01, reg_alpha=1e-5)

start = timeit.default_timer()

xgb_reg_price.fit(train_x_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)

try:
    print(xgb_reg_price.best_params_)
    print("\n")
    print(xgb_reg_price.best_score_)
except:
    print("Did Not Use GridSearch")
# #### Sensitivity Analysis

# MaxDepth

# In[697]:

choices = [2,7,11,15,25]
cv_scores = []

for i in choices:
    sensitivity_analysis = XGBRegressor(silent=False, max_depth=i, n_estimators=100)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    cv_scores.append(np.mean(cross_val_scores) * -1)

pd.DataFrame({
    'Value': choices,
    '3 Fold CV MAE Score': cv_scores
})


# N-Estimators

# In[699]:

choices = [400,600,800,1000]
cv_scores = []

for i in choices:
    sensitivity_analysis = XGBRegressor(silent=False, n_estimators=i)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    cv_scores.append(np.mean(cross_val_scores) * -1)

pd.DataFrame({
    'Value': choices,
    '3 Fold CV MAE Score': cv_scores
})


# In[702]:

choices = [1,4,8,12,15,20]
cv_scores = []

for i in choices:
    sensitivity_analysis = XGBRegressor(silent=False, min_child_weight=i)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    cv_scores.append(np.mean(cross_val_scores) * -1)

pd.DataFrame({
    'Value': choices,
    '3 Fold CV MAE Score': cv_scores
})


# In[705]:

choices = [0.0005, 0.005, 0.01, 0.5, 1, 3]
cv_scores = []

for i in choices:
    sensitivity_analysis = XGBRegressor(silent=False, gamma=i)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    cv_scores.append(np.mean(cross_val_scores) * -1)

pd.DataFrame({
    'Value': choices,
    '3 Fold CV MAE Score': cv_scores
})


# Load Saved Pickle File

# In[132]:

filename = 'model_xgb.pkl'
xgb_reg_preds_price = load_saved_pickle_model(filename)


# In[165]:

#train mape
xgb_reg_preds_price_train = xgb_reg_price.predict(train_x_scaled)
xgb_reg_preds_price_train = std_scaler.inverse_transform(xgb_reg_preds_price_train)
err = mean_absolute_percentage_error(xgb_reg_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
xgb_reg_preds_price = xgb_reg_price.predict(test_x_scaled)
xgb_reg_preds_price = std_scaler.inverse_transform(xgb_reg_preds_price)
err = mean_absolute_percentage_error(xgb_reg_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)

point_err_xgb = point_absolute_percentage_error(xgb_reg_preds_price, test_y['TARGET_DA_LMP'])
max_err_xgb, min_err_xgb, ci_xgb, sd_xgb, var_xgb = get_CI_VAR_SD(point_err_xgb)
print(max_err_xgb, min_err_xgb, ci_xgb, sd_xgb, var_xgb)


# In[166]:

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list1 =[]
mae_list1 = []
rmse_list1 = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list1.append(mean_absolute_percentage_error(
        xgb_reg_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list1.append(mean_absolute_error(
        xgb_reg_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list1.append(np.sqrt(mean_squared_error(
        xgb_reg_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
np.mean(temp_list1)

save_model(xgb_reg_price, 'model_xgb.pkl')
# ### RVM - RBF Kernel
rvm_rbf_price = RVR(kernel='rbf', gamma=0.001, verbose=1, n_iter=600)

start = timeit.default_timer()

rvm_rbf_price = rvm_rbf_price.fit(train_x_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)
# #### Sensitivity 

# n_iter

# In[680]:

rbf_n_iter_range = [100, 400, 550, 600, 650, 800]
rbf_n_iter_sensitivity = []

for i in rbf_n_iter_range:
    sensitivity_analysis = RVR(kernel='rbf', gamma=0.001, verbose=10, n_iter=i)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    rbf_n_iter_sensitivity.append(np.mean(cross_val_scores) * -1)


# In[681]:

pd.DataFrame({
    'n_iter': rbf_n_iter_range,
    'cross_val_3fold_mae_scores': rbf_n_iter_sensitivity
})


# In[690]:

rbf_gamma_range = [0.00005, 0.0001, 0.001, 0.05, 0.1, 1, 3]
rbf_gamma_sensitivity = []

for i in rbf_gamma_range:
    sensitivity_analysis = RVR(kernel='rbf', gamma=i, verbose=10, n_iter=300)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    rbf_gamma_sensitivity.append(np.mean(cross_val_scores) * -1)


# In[691]:

pd.DataFrame({
    'n_iter': rbf_gamma_range,
    'cross_val_3fold_mae_scores': rbf_gamma_sensitivity
})


# Load Saved Model

# In[135]:

filename = 'model_rvm-rbf.pkl'
rvm_rbf_price = load_saved_pickle_model(filename)


# In[664]:

rvm_rbf_price.get_params


# In[167]:

#train mape
rvm_rbf_preds_price_train = rvm_rbf_price.predict(train_x_scaled)
rvm_rbf_preds_price_train = std_scaler.inverse_transform(rvm_rbf_preds_price_train)
err = mean_absolute_percentage_error(rvm_rbf_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
rvm_rbf_preds_price = rvm_rbf_price.predict(test_x_scaled)
rvm_rbf_preds_price = std_scaler.inverse_transform(rvm_rbf_preds_price)
err = mean_absolute_percentage_error(rvm_rbf_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)


point_err_rvm_rbf = point_absolute_percentage_error(rvm_rbf_preds_price, test_y['TARGET_DA_LMP'])
max_err_rvm_rbf, min_err_rvm_rbf, ci_rvm_rbf, sd_rvm_rbf, var_rvm_rbf = get_CI_VAR_SD(point_err_rvm_rbf)
print(max_err_rvm_rbf, min_err_rvm_rbf, ci_rvm_rbf, sd_rvm_rbf, var_rvm_rbf)

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list2 =[]
mae_list2 = []
rmse_list2 = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list2.append(mean_absolute_percentage_error(
        rvm_rbf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list2.append(mean_absolute_error(
        rvm_rbf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list2.append(np.sqrt(mean_squared_error(
        rvm_rbf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list2


# In[665]:

mae_list2

save_model(rvm_rbf_price, 'model_rvm-rbf.pkl')
# ### RVM - polynomial kernel
rvm_pol_price = RVR(kernel='poly', degree=3, coef0=0.08, gamma=0.05, verbose=1, n_iter=300)

start = timeit.default_timer()

rvm_pol_price = rvm_pol_price.fit(train_x_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)
# Sensitivity Analysis

# In[706]:

#Gamma
choices = [0.00005, 0.0001, 0.001, 0.05, 0.1, 1, 3]
cv_scores = []

for i in choices:
    sensitivity_analysis = RVR(kernel='poly', degree=3, gamma=i, verbose=1, n_iter=300)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    cv_scores.append(np.mean(cross_val_scores) * -1)

pd.DataFrame({
    'Value': choices,
    '3 Fold CV MAE Score': cv_scores
})


# In[708]:

#n_iter
choices = [2,3,5,8,12]
cv_scores = []

for i in choices:
    sensitivity_analysis = RVR(kernel='poly', degree=i, gamma=None, verbose=1, n_iter=300)
    cross_val_scores = cross_val_score(sensitivity_analysis, train_x_scaled, train_y_scaled, 
                                       scoring=inversed_mae_score, cv=3)
    cv_scores.append(np.mean(cross_val_scores) * -1)

pd.DataFrame({
    'Value': choices,
    '3 Fold CV MAE Score': cv_scores
})


# Load Saved Model

# In[169]:

filename = 'model_rvm-poly.pk'
rvm_pol_price = load_saved_pickle_model(filename)


# In[170]:

#train mape
rvm_pol_preds_price_train = rvm_pol_price.predict(train_x_scaled)
rvm_pol_preds_price_train = std_scaler.inverse_transform(rvm_pol_preds_price_train)
err = mean_absolute_percentage_error(rvm_pol_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
rvm_pol_preds_price = rvm_pol_price.predict(test_x_scaled)
rvm_pol_preds_price = std_scaler.inverse_transform(rvm_pol_preds_price)
err = mean_absolute_percentage_error(rvm_pol_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)


# In[171]:

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list3 =[]
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list3.append(mean_absolute_percentage_error(
        rvm_pol_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
temp_list3

save_model(rvm_pol_price, 'model_rvm-poly.pk')
# ### Kernel Ridge Regression
# - Noresult after running for more than half hour
params = {
    'alpha':[0.1,1,5],
    'gamma': [1],
    'kernel':['linear'],
    'degree':[3],
    'coef0': [1]
}

krr_price = GridSearchCV(KernelRidge(), params, n_jobs=-1, cv=7, scoring='neg_mean_squared_error')
krr_price = krr_price.fit(train_x_scaled, train_y_scaled)

try:
    print(krr_price.best_params_)
    print("\n")
    print(krr_price.best_score_)
except:
    print("Did Not Use GridSearch")
# ### ANN - MLP for comparison

# In[172]:

params = {
    'hidden_layer_sizes': [(20,10,5)],
    'activation': ['relu'],
    'solver': ['lbfgs'],
    'max_iter': [1000], #np.linspace(40,100,60, dtype=int)
    'alpha': [0.0001],
}

mlp_reg_price = GridSearchCV(MLPRegressor(learning_rate='adaptive', verbose=1), params, cv=7, n_jobs=-1)
#mlp_reg_price = MLPRegressor(hidden_layer_sizes=(20,10,5), activation='relu', verbose=1, max_iter=5000, solver='lbfgs', 
                      #alpha=0.0001)

start = timeit.default_timer()

mlp_reg_price.fit(train_x_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)

try:
    print(mlp_reg_price.best_params_)
    print("\n")
    print(mlp_reg_price.best_score_)
except:
    print("Did Not Use GridSearch")


# In[173]:

#train mape
mlp_reg_preds_price_train = mlp_reg_price.predict(train_x_scaled)
mlp_reg_preds_price_train = std_scaler.inverse_transform(mlp_reg_preds_price_train)
err = mean_absolute_percentage_error(mlp_reg_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
mlp_reg_preds_price = mlp_reg_price.predict(test_x_scaled)
mlp_reg_preds_price = std_scaler.inverse_transform(mlp_reg_preds_price)
err = mean_absolute_percentage_error(mlp_reg_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)


point_err_mlp = point_absolute_percentage_error(mlp_reg_preds_price, test_y['TARGET_DA_LMP'])
max_err_mlp, min_err_mlp, ci_mlp, sd_mlp, var_mlp = get_CI_VAR_SD(point_err_mlp)
print(max_err_mlp, min_err_mlp, ci_mlp, sd_mlp, var_mlp)

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list4 =[]
mae_list4 = []
rmse_list4 = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list4.append(mean_absolute_percentage_error(
        mlp_reg_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list4.append(mean_absolute_error(
        mlp_reg_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list4.append(np.sqrt(mean_squared_error(
        mlp_reg_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list4


# ### SVM
params = {
        'C': [3],
}

svm_price = GridSearchCV(SVR(), params, cv=7)

start = timeit.default_timer()

svm_price.fit(train_x_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)
# In[174]:

filename = 'model_svm-rbf.pk'
svm_price = load_saved_pickle_model(filename)


# In[175]:

#train mape
svm_rbf_preds_price_train = svm_price.predict(train_x_scaled)
svm_rbf_preds_price_train = std_scaler.inverse_transform(svm_rbf_preds_price_train)
err = mean_absolute_percentage_error(svm_rbf_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
svm_rbf_preds_price = svm_price.predict(test_x_scaled)
svm_rbf_preds_price = std_scaler.inverse_transform(svm_rbf_preds_price)
err = mean_absolute_percentage_error(svm_rbf_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)


point_err_svm = point_absolute_percentage_error(svm_rbf_preds_price, test_y['TARGET_DA_LMP'])
max_err_svm, min_err_svm, ci_svm, sd_svm, var_svm = get_CI_VAR_SD(point_err_svm)
print(max_err_svm, min_err_svm, ci_svm, sd_svm, var_svm)

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list5 =[]
mae_list5 = []
rmse_list5 = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list5.append(mean_absolute_percentage_error(
        svm_rbf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list5.append(mean_absolute_error(
        svm_rbf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list5.append(np.sqrt(mean_squared_error(
        svm_rbf_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list5

save_model(svm_price, 'model_svm-rbf.pk')
# ### Lasso
from sklearn.linear_model import Lasso

lasso_price = Lasso(alpha=0.001)
lasso_price = lasso_price.fit(train_x_scaled, train_y_scaled)
# In[176]:

filename = 'model_lasso.pk'
lasso_price = load_saved_pickle_model(filename)


# In[177]:

#train mape
lasso_preds_price_train = lasso_price.predict(train_x_scaled)
lasso_preds_price_train = std_scaler.inverse_transform(lasso_preds_price_train)
err = mean_absolute_percentage_error(lasso_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
lasso_preds_price = lasso_price.predict(test_x_scaled)
lasso_preds_price = std_scaler.inverse_transform(lasso_preds_price)
err = mean_absolute_percentage_error(lasso_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)

point_err_lasso = point_absolute_percentage_error(lasso_preds_price, test_y['TARGET_DA_LMP'])
max_err_lasso, min_err_lasso, ci_lasso, sd_lasso, var_lasso = get_CI_VAR_SD(point_err_lasso)
print(max_err_lasso, min_err_lasso, ci_lasso, sd_lasso, var_lasso)


days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list6 =[]
mae_list6 = []
rmse_list6 = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list6.append(mean_absolute_percentage_error(
        lasso_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list6.append(mean_absolute_error(
        lasso_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list6.append(np.sqrt(mean_squared_error(
        lasso_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list6

save_model(lasso_price, 'model_lasso.pk')
# ### RNN

# In[178]:

from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM


# In[179]:

train_x_scaled[:, None].shape

rnn_train_x = train_x_scaled[:, None]
rnn_test_x = test_x_scaled[:, None]

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)


# In[180]:

rnn_train_x.shape


# In[181]:

K.clear_session()
model = Sequential()

model.add(LSTM(15, input_shape=(1, 10), activation='relu', return_sequences=True))
model.add(LSTM(15, activation='relu', return_sequences=True))
model.add(LSTM(15, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam')

start = timeit.default_timer()

model.fit(rnn_train_x, train_y_scaled,
          epochs=30, batch_size=20, verbose=1,
         callbacks=[early_stop], validation_split=0.1)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)
#number of iters is approx 3000 - each epoch has number of (datapoints/batch_size)*2 iterations


# In[182]:

#train mape
rnn_preds_price_train = model.predict(rnn_train_x)
rnn_preds_price_train = rnn_preds_price_train.flatten()
rnn_preds_price_train = std_scaler.inverse_transform(rnn_preds_price_train)
err = mean_absolute_percentage_error(rnn_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
rnn_preds_price = model.predict(rnn_test_x)
rnn_preds_price = rnn_preds_price.flatten()
rnn_preds_price = std_scaler.inverse_transform(rnn_preds_price)
err = mean_absolute_percentage_error(rnn_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)

point_err_rnn = point_absolute_percentage_error(rnn_preds_price, test_y['TARGET_DA_LMP'])
max_err_rnn, min_err_rnn, ci_rnn, sd_rnn, var_rnn = get_CI_VAR_SD(point_err_rnn)
print(max_err_rnn, min_err_rnn, ci_rnn, sd_rnn, var_rnn)

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list7 =[]
mae_list7 = []
rmse_list7 = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list7.append(mean_absolute_percentage_error(
        rnn_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list7.append(mean_absolute_error(
        rnn_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list7.append(np.sqrt(mean_squared_error(
        rnn_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list7


# # Level 2 Learning

# In[183]:

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    print(np.sum(days_list[:i+1])*24)    


# #### Create stacked train and test datasets

# In[184]:

stacked_train = pd.DataFrame({
    'xgb': xgb_reg_preds_price_train,
    'rvm_rbf': rvm_rbf_preds_price_train,
    'rvm_pol': rvm_pol_preds_price_train,
    #'rnn': rnn_preds_price_train
})

stacked_test = pd.DataFrame({
    'xgb': xgb_reg_preds_price,
    'rvm_rbf': rvm_rbf_preds_price,
    'rvm_pol': rvm_pol_preds_price,
    #'rnn': rnn_preds_price
})


# In[185]:

std_scaler2 = StandardScaler()

stacked_train_scaled = std_scaler2.fit_transform(stacked_train)
stacked_test_scaled = std_scaler2.fit_transform(stacked_test)


# In[186]:

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

stk_br = BaggingRegressor(base_estimator=ElasticNet(alpha=0.07, l1_ratio=0.2, tol=0.0001, max_iter=500),
                          n_estimators=50,bootstrap=True,bootstrap_features=True,random_state=3)
#l1_ratio=0.4 or 0.3 dec overall mape but inc jan mae # test this and check
start = timeit.default_timer()

stk_br.fit(stacked_train_scaled, train_y_scaled)

stop = timeit.default_timer()
execution_time = stop - start
print("Time: ", execution_time)
# #### Sensitivity of Elastic Net

# Alpha

# In[659]:

alpha_range = [0, 0.01, 0.07, 0.14, 0.2, 0.5, 1, 5, 10]
alpha_sensitivity = []

for i in alpha_range:
    sensitivity_analysis = BaggingRegressor(
        base_estimator=ElasticNet(alpha=i, l1_ratio=0.5, tol=0.0001, max_iter=500),
                          n_estimators=50,bootstrap=True,bootstrap_features=True,random_state=3, verbose=10)
    cross_val_scores = cross_val_score(sensitivity_analysis, stacked_train, train_y, 
                                       scoring='neg_mean_absolute_error', cv=3)
    alpha_sensitivity.append(np.mean(cross_val_scores) * -1)


# In[660]:

pd.DataFrame({
    'alpha': alpha_range,
    'cross_val_3fold_mae_scores': alpha_sensitivity 
})


# L1 Ratio

# In[656]:

l1_ratio_range = np.linspace(0,1,11).tolist()
l1_sensitivity = []

for i in l1_ratio_range:
    sensitivity_analysis = BaggingRegressor(
        base_estimator=ElasticNet(alpha=1, l1_ratio=i, tol=0.0001, max_iter=500),
                          n_estimators=50,bootstrap=True,bootstrap_features=True,random_state=3)
    cross_val_scores = cross_val_score(sensitivity_analysis, stacked_train, train_y, 
                                       scoring='neg_mean_absolute_error', cv=3)
    l1_sensitivity.append(np.mean(cross_val_scores) * -1)


# In[657]:

pd.DataFrame({
    'l1_ratio': l1_ratio_range,
    'cross_val_3fold_mae_scores': l1_sensitivity 
})


# Number of Bagging Estimators

# In[661]:

bagg_estimators_range = [10,30,50,70,90]
bagging_sensitivity = []

for i in bagg_estimators_range:
    sensitivity_analysis = BaggingRegressor(
        base_estimator=ElasticNet(alpha=1, l1_ratio=0.5, tol=0.0001, max_iter=500),
                          n_estimators=i,bootstrap=True,bootstrap_features=True,random_state=3, verbose=10)
    cross_val_scores = cross_val_score(sensitivity_analysis, stacked_train, train_y, 
                                       scoring='neg_mean_absolute_error', cv=3)
    bagging_sensitivity.append(np.mean(cross_val_scores) * -1)


# In[662]:

pd.DataFrame({
    'bagg_estimators': bagg_estimators_range,
    'cross_val_3fold_mae_scores': bagging_sensitivity 
})


# Load Pickle Model

# In[187]:

filename = 'model_stacked.pk'
stk_br = load_saved_pickle_model(filename)


# In[188]:

#train mape
stacked_preds_price_train = stk_br.predict(stacked_train_scaled)
stacked_preds_price_train = std_scaler.inverse_transform(stacked_preds_price_train)
err = mean_absolute_percentage_error(stacked_preds_price_train, train_y['TARGET_DA_LMP'])
print("Train: ", err)

#test mape
stacked_preds_price = stk_br.predict(stacked_test_scaled)
stacked_preds_price = std_scaler.inverse_transform(stacked_preds_price)
err = mean_absolute_percentage_error(stacked_preds_price, test_y['TARGET_DA_LMP'])
print("Test: ", err)

point_err_stacked = point_absolute_percentage_error(stacked_preds_price, test_y['TARGET_DA_LMP'])
max_err_stacked, min_err_stacked, ci_stacked, sd_stacked, var_stacked = get_CI_VAR_SD(point_err_stacked)
print(max_err_stacked, min_err_stacked, ci_stacked, sd_stacked, var_stacked)

days_list = [0,8,5,8,7,8,7,8,8,7,8,7,8,8,5,8,7,8,7,8,8,7,8,7,8]
temp_list =[]
mae_list = []
rmse_list = []
for i in range(0,len(days_list)-1):
    #print(int(np.sum(days_list[:i+1])),int(np.sum(days_list[:i+2])))
    #print(np.sum(days_list[:i+1])*24)
    temp_list.append(mean_absolute_percentage_error(
        stacked_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    mae_list.append(mean_absolute_error(
        stacked_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24]))
    
    rmse_list.append(np.sqrt(mean_squared_error(
        stacked_preds_price[int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24], 
                            test_y['TARGET_DA_LMP'][int(np.sum(days_list[:i+1]))*24:int(np.sum(days_list[:i+2]))*24])))
    
temp_list

save_model(stk_br, "model_stacked.pk")
# ### Exogenous ARIMA for Comparison

# In[207]:

arimax_results = pd.read_csv('ARIMAX/prediction_results_exogenous_arima.csv')


# In[211]:

arimax_results['Month'] = arimax_results['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)


# In[212]:

arimax_results['Date'] = pd.to_datetime(arimax_results['Date'])


# In[213]:

arimax_results.set_index('Date', drop=True, inplace=True)


# In[214]:

arimax_results.head()


# In[215]:

arimax_for_comparison = arimax_results[arimax_results['Day_Date'] >= 24]


# In[291]:

def mean_absolute_percentage_error_arimax(df):
    y_pred = df['Predicted_Values']
    y_true = df['TARGET_DA_LMP']
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error_arimax(df):
    y_pred = df['Predicted_Values']
    y_true = df['TARGET_DA_LMP']
    return mean_absolute_error(y_true, y_pred)

def mean_squared_error_arimax(df):
    y_pred = df['Predicted_Values']
    y_true = df['TARGET_DA_LMP']
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[235]:

arimax_2013_monthly_mape = arimax_for_comparison.groupby(['Month'])['Predicted_Values', 
                                         'TARGET_DA_LMP'].apply(mean_absolute_percentage_error_arimax).reset_index()


# In[268]:

arimax_2013_monthly_mae = arimax_for_comparison.groupby(['Month'])['Predicted_Values', 
                                         'TARGET_DA_LMP'].apply(mean_absolute_error_arimax).reset_index()


# In[292]:

arimax_2013_monthly_rmse = arimax_for_comparison.groupby(['Month'])['Predicted_Values', 
                                         'TARGET_DA_LMP'].apply(mean_squared_error_arimax).reset_index()


# In[238]:

plt.bar(np.arange(12),arimax_2013_monthly_mape[0])
plt.xticks(np.arange(12), ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                          'November', 'December'))
plt.ylabel('MAPE')
plt.show()


#  

# # Error Metrics, Comparisons and Plots

# ### MAPE

# In[255]:

abcdef = pd.DataFrame({
    #'RF': temp_list_rf,
    'ARIMA': [0 for _ in range(12)] + arimax_2013_monthly_mape[0].values.tolist(),
    'RVM': temp_list2,
    'SVM': temp_list5,
    'MLP': temp_list4,
    'LASSO':temp_list6,
    'Proposed': temp_list,
    'RNN': temp_list7
})


# In[256]:

abcdef_2013_only = pd.DataFrame({
    #'RF': temp_list_rf,
    'ARIMAX': arimax_2013_monthly_mape[0].values.tolist(),
    'RVM': temp_list2[12:],
    'SVM': temp_list5[12:],
    'MLP': temp_list4[12:],
    'LASSO':temp_list6[12:],
    'Proposed': temp_list[12:],
    'RNN': temp_list7[12:]
})


# In[257]:

abcdef


# In[258]:

abcdef_2013_only


# In[259]:

print(abcdef_2013_only.mean())
my_plt = abcdef_2013_only.mean().plot(kind='bar')
plt.xlabel("Models", {'fontsize':30})
plt.ylabel("Mean Absolute Percentage Errors",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("MAPE Comparison of Each Model", {'fontsize':30})
plt.savefig("plot_mean_mape.jpg", format='jpg', bbox_inches='tight', dpi=600)


# In[271]:

print(abcdef_2013_only.iloc[[0,5,7,10]])
abc = abcdef_2013_only.iloc[[0,5,7,10]].plot(kind='bar')
plt.xlabel("Seasons", {'fontsize':30})
plt.ylabel("Mean Absolute Percentage Error",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
abc.set_xticklabels(['Winter','Spring','Summer','Fall'])
plt.title("Season Wise Mean Absolute Percentage Errors", {'fontsize':30})
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1, fontsize=18)
plt.show()
plt.savefig("plot_season_mape.jpg", orientation='landscape', format='jpg', bbox_inches='tight', dpi=600)


# ### MAE

# In[272]:

mae_df_final = pd.DataFrame({
    #'RF': mae_list_rf,
    'RVM': mae_list2,
    'SVM': mae_list5,
    'MLP': mae_list4,
    'LASSO':mae_list6,
    'Proposed': mae_list,
    'RNN': mae_list7
})


# In[273]:

mae_df_final_2013_only = pd.DataFrame({
    'ARIMAX': arimax_2013_monthly_mae[0].values.tolist(),
    'RVM': mae_list2[12:],
    'SVM': mae_list5[12:],
    'MLP': mae_list4[12:],
    'LASSO':mae_list6[12:],
    'Proposed': mae_list[12:],
    'RNN': mae_list7[12:]
})


# In[494]:

mae_df_final


# In[336]:

mae_df_final_2013_only


# In[283]:

print(mae_df_final.mean(), end='\n\n')
print(mae_df_final_2013_only.mean())
my_plt = mae_df_final_2013_only.mean().plot(kind='bar')
plt.xlabel("Models", {'fontsize':30})
plt.ylabel("Mean Absolute Errors",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("MAE Comparison of Each Model", {'fontsize':30})
plt.show()
#plt.savefig("plot_mean_mae.jpg")


# ### RMSE

# In[287]:

rmse_df_final = pd.DataFrame({
    #'RF': rmse_list_rf,
    'RVM': rmse_list2,
    'SVM': rmse_list5,
    'MLP': rmse_list4,
    'LASSO':rmse_list6,
    'Proposed': rmse_list,
    'RNN': rmse_list7
})


# In[293]:

rmse_df_final_2013_only = pd.DataFrame({
    #'RF': rmse_list_rf,
    'ARIMAX': arimax_2013_monthly_rmse[0].values.tolist(),
    'RVM': rmse_list2[12:],
    'SVM': rmse_list5[12:],
    'MLP': rmse_list4[12:],
    'LASSO':rmse_list6[12:],
    'Proposed': rmse_list[12:],
    'RNN': rmse_list7[12:]
})


# In[294]:

print(rmse_df_final.mean(), end='\n\n')
print(rmse_df_final_2013_only.mean())
my_plt = rmse_df_final.mean().plot(kind='bar')
plt.xlabel("Models", {'fontsize':30})
plt.ylabel("Root Mean Squared Errors",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("RMSE Comparison of Each Model", {'fontsize':30})
plt.savefig("plot_mean_rmse.jpg")


# # Trend Plots

# In[609]:

test_for_final_comparisons = test.copy()


# In[610]:

test_for_final_comparisons['Proposed'] = stacked_preds_price
test_for_final_comparisons['RVM'] = rvm_pol_preds_price
test_for_final_comparisons['SVM'] = svm_rbf_preds_price
test_for_final_comparisons['XGB'] = xgb_reg_preds_price
test_for_final_comparisons['LASSO'] = lasso_preds_price
test_for_final_comparisons['RNN'] = rnn_preds_price
test_for_final_comparisons['MLP'] = mlp_reg_preds_price
test_for_final_comparisons['RF'] = rf_preds_price


# In[611]:

test_for_final_comparisons['YEAR'] = test_for_final_comparisons['Date'].apply(lambda x: x.year)
test_for_final_comparisons['MONTH'] = test_for_final_comparisons['Date'].apply(lambda x: x.month)
test_for_final_comparisons['WEEK'] = test_for_final_comparisons['Date'].apply(lambda x: x.week)


# In[612]:

test_for_final_comparisons_2013_only = test_for_final_comparisons[test_for_final_comparisons['YEAR'] == 2013]


# In[613]:

test_for_final_comparisons_2013_only['ARIMAX'] = arimax_for_comparison['Predicted_Values'].values.tolist()


# In[615]:

test_for_final_comparisons_2013_only.head()


# ### Plot Predicted Vs. Actual

# ### Single Day

# In[628]:

#import matplotlib.pyplot as plt
start = 0+24+24+24+24+24
end = start + 24

temp = pd.DataFrame({
    'Actual':test_for_final_comparisons_2013_only['TARGET_DA_LMP'][start:end],
    'RNN':test_for_final_comparisons_2013_only['RNN'][start:end],
    'MLP':test_for_final_comparisons_2013_only['MLP'][start:end],
})
temp = temp.reset_index(drop=True)
plt.plot(temp['Actual'], 'r--', label="Actual")
plt.plot(temp['RNN'], 'b-', label="RNN")
plt.plot(temp['MLP'], 'y-', label="MLP")
plt.xlabel("Hour", {'fontsize':30})
plt.ylabel("Electricity Price",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Predicted vs Actual Prices (29 January, 2013)", {'fontsize':30})
plt.legend(loc=2, fontsize=23)
plt.gcf()
plt.savefig("NEWPLOTS_Feb19/plot_single_day1.jpg", format='jpg', bbox_inches='tight', dpi=600)


# In[629]:

start = 0+24+24+24+24+24
end = start + 24

temp = pd.DataFrame({
    'Actual':test_for_final_comparisons_2013_only['TARGET_DA_LMP'][start:end],
    'SVM':test_for_final_comparisons_2013_only['SVM'][start:end],
    'Proposed':test_for_final_comparisons_2013_only['Proposed'][start:end],
})
temp = temp.reset_index(drop=True)
plt.plot(temp['Actual'], 'r--', label="Actual")
plt.plot(temp['SVM'], 'b-', label="SVM")
plt.plot(temp['Proposed'], 'g--', label="Proposed")
plt.xlabel("Hour", {'fontsize':30})
plt.ylabel("Electricity Price",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Predicted vs Actual Prices (29 January, 2013)", {'fontsize':30})
plt.legend(loc=2, fontsize=23)
plt.gcf()
plt.savefig("NEWPLOTS_Feb19/plot_single_day2.jpg", format='jpg', bbox_inches='tight', dpi=600)


# In[631]:

#import matplotlib.pyplot as plt
start = 0+24+24+24+24+24
end = start + 24

temp = pd.DataFrame({
    'Actual':test_for_final_comparisons_2013_only['TARGET_DA_LMP'][start:end],
    'LASSO':test_for_final_comparisons_2013_only['LASSO'][start:end],
    'RF':test_for_final_comparisons_2013_only['RF'][start:end],
})
temp = temp.reset_index(drop=True)
plt.plot(temp['Actual'], 'r--', label="Actual")
plt.plot(temp['LASSO'], 'b-', label="LASSO")
plt.plot(temp['RF'], 'y-', label="RF")
plt.xlabel("Hour", {'fontsize':30})
plt.ylabel("Electricity Price",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Predicted vs Actual Prices (29 January, 2013)", {'fontsize':30})
plt.legend(loc=2, fontsize=23)
plt.gcf()
plt.savefig("NEWPLOTS_Feb19/plot_single_day3.jpg", format='jpg', bbox_inches='tight', dpi=600)


# ### WEEK
temp = pd.DataFrame({
    'Proposed':stacked_preds_price[2136:2136+168], 
    'Actual':test_y['TARGET_DA_LMP'][2136:2136+168],
    'RVM':rvm_pol_preds_price[2136:2136+168],
    'SVM':svm_rbf_preds_price[2136:2136+168],
})
# In[632]:

week_trend_plots = test_for_final_comparisons_2013_only[test_for_final_comparisons_2013_only['MONTH'] == 1]


# In[633]:

week_trend_plots.shape


# In[634]:

#import matplotlib.pyplot as plt

temp = pd.DataFrame({
    'Proposed':week_trend_plots['Proposed'], 
    'Actual':week_trend_plots['TARGET_DA_LMP'],
    'SVM':week_trend_plots['SVM'],
})
temp = temp.reset_index(drop=True)
plt.plot(temp['Proposed'], 'g--', label="Proposed")
plt.plot(temp['Actual'], 'r--', label="Actual")
plt.plot(temp['SVM'], 'b-', label="SVM")
#temp.plot(kind='line')
plt.xlabel("Hour", {'fontsize':30})
plt.ylabel("Electricity Price",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Predicted vs Actual Prices (Winter Week, January 2013)", {'fontsize':30})
plt.legend(loc=1, fontsize=23)
plt.gcf()
plt.savefig("NEWPLOTS_Feb19/plot_winter_week1.jpg", format='jpg', bbox_inches='tight', dpi=600)


# In[635]:

#import matplotlib.pyplot as plt
temp = pd.DataFrame({
    'RNN':week_trend_plots['RNN'], 
    'Actual':week_trend_plots['TARGET_DA_LMP'],
    'MLP':week_trend_plots['MLP'],
})
temp = temp.reset_index(drop=True)
plt.plot(temp['RNN'], 'b-', label="RNN")
plt.plot(temp['Actual'], 'r--', label="Actual")
plt.plot(temp['MLP'], 'y-', label="MLP")
#temp.plot(kind='line')
plt.xlabel("Hour", {'fontsize':30})
plt.ylabel("Electricity Price",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Predicted vs Actual Prices (Winter Week, January 2013)", {'fontsize':30})
plt.legend(loc=1, fontsize=23)
plt.gcf()
plt.savefig("NEWPLOTS_Feb19/plot_winter_week2.jpg", format='jpg', bbox_inches='tight', dpi=600)


# In[636]:

temp = pd.DataFrame({
    'LASSO':week_trend_plots['LASSO'], 
    'Actual':week_trend_plots['TARGET_DA_LMP'],
    'RF':week_trend_plots['RF']
})
temp = temp.reset_index(drop=True)
plt.plot(temp['LASSO'], 'b-', label="LASSO")
plt.plot(temp['Actual'], 'r--', label="Actual")
plt.plot(temp['RF'], 'y-', label="RF")
#temp.plot(kind='line')
plt.xlabel("Hour", {'fontsize':30})
plt.ylabel("Electricity Price",{'fontsize':30})
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title("Predicted vs Actual Prices (Winter Week, January 2013)", {'fontsize':30})
plt.legend(loc=1, fontsize=23)
plt.gcf()
plt.savefig("NEWPLOTS_Feb19/plot_winter_week3.jpg", format='jpg', bbox_inches='tight', dpi=600)

