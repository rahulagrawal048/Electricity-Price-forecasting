# README #

This repository contains the python code for an an ensemble model of Bayesian RVM Model and Extreme Gradient Boosting which is used for predicting 
hour ahead Electricity Prices. It also contains the code for other models the performances of which were tested against this ensemble model for 
the same task. There is a .py and .ipynb version of all the code files. The models train have been stored as a pickle file to avoid retraining
for testing purposes. 

1) Very Short Term Price Forecasting- RVM.ipynb is the main code file.
2) loaddata1.xls contains ISO New England electricity dataset of 2012.
3) loaddata2.xls contains ISO New England electricity dataset of 2013.
4) loaddata3.xls contains ISO New England electricity dataset of 2014.
5) CrudeOil.csv contains crude oil prices dataset from Quandl.
6) naturalgas.csv contains natural gas prices dataset from US Henry Hub.