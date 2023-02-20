import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import mean_squared_error,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
def history_train_test_timewise_split(data,column,history_start_date='2010-01-01',history_end_date='2050-12-31',train_start_date='2010-01-01',train_end_date='2050-12-31',test_start_date='2010-01-01',test_end_date='2050-12-31'): 
    import pandas as pd
    try:
        history_start_date=pd.to_datetime(history_start_date)
        history_end_date=pd.to_datetime(history_end_date)
        train_start_date=pd.to_datetime(train_start_date)
        train_end_date=pd.to_datetime(train_end_date)
        test_start_date=pd.to_datetime(test_start_date)
        test_end_date=pd.to_datetime(test_end_date)
    except:
        raise ValueError('Either format is incorrect or date is in correct.Dates are expected as string and in "yyyy-mm-dd" format')
        return
    if data[column].isnull().sum()>0:
        raise ValueError('Split column contains null values')
        return
    else:
        data[column]=pd.to_datetime(data[column])
        history=data.loc[(data[column]>=history_start_date)&(data[column]<=history_end_date)]
        train=data.loc[(data[column]>=train_start_date)&(data[column]<=train_end_date)]
        test=data.loc[(data[column]>=test_start_date)&(data[column]<=test_end_date)]
        left_history=data.loc[(data[column]>=history_start_date)&(data[column]<train_start_date)]
        history.reset_index(drop=True,inplace=True)
        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)
        return history,train,left_history,test
       
 dataset = pd.read_csv("covid_19_india.csv",error_bad_lines=False)
 dataset.isna().any()
       
 no_use = ['Sno','Time']
       
dataset.drop(columns = no_use, inplace = True)
print("The shape of remaining data: ",dataset.shape)

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
dataset['State/UnionTerritory']= label_encoder.fit_transform(dataset['State/UnionTerritory'])
  
dataset['State/UnionTerritory'].unique()

dispute_history,train,left_history,test=history_train_test_timewise_split(dataset.copy(),
                                                                 'Date',
                                                                 history_start_date='2020-01-30',
                                                                 history_end_date='2021-06-08',
                                                                 train_start_date='2020-01-30',
                                                                 train_end_date='2021-03-15',
                                                                 test_start_date='2021-03-16',
                                                                  #test_end_date='2019-05-12'
                                                                 )
print("Percentage of data in train: ",((train.shape[0]/dispute_history.shape[0])*100))
print("Percentage of data in test: ",((test.shape[0]/dispute_history.shape[0])*100))

# date features
def date_features(train, dispute_history, test, date_col):
    train['weekday'] = train[date_col].dt.weekday
    
    dispute_history['weekday'] = dispute_history[date_col].dt.weekday
    
    test['weekday'] = test[date_col].dt.weekday
    
    return train, dispute_history, test

# forming the date features
train, dispute_history, test = date_features(train, dispute_history, test, 'Date')

train.head()
res = train['Confirmed']
res_test = test['Confirmed']

train.drop(columns = ['Date','ConfirmedIndianNational','ConfirmedForeignNational','Confirmed'], inplace = True)
test.drop(columns = ['Date','ConfirmedIndianNational','ConfirmedForeignNational','Confirmed'], inplace = True)
print("The shape of remaining data: ",train.shape)

from sklearn import linear_model
reg_linear = linear_model.LinearRegression()

# fit the model to start training.
reg_linear.fit(train, res)
# on test set
pred_test = reg_linear.predict(test)

from sklearn.metrics import mean_squared_error

rms = mean_squared_error(res_test, pred_test, squared=False)
rms

rms/test.shape[0]

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators= 500 ,max_depth=3, random_state=0)
regr.fit(train, res)

pred_test = regr.predict(test)
rms = mean_squared_error(res_test, pred_test, squared=False)
rms

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train,res)

pred_test = regressor.predict(test)
rms = mean_squared_error(res_test, pred_test, squared=False)
rms

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

regr.fit(train, res)
pred_test = regr.predict(test)

rms = mean_squared_error(res_test, pred_test, squared=False)
rms

from sklearn import linear_model
clf = linear_model.Lasso(alpha=1)

clf.fit(train,res)
pred_test = clf.predict(test)

rms = mean_squared_error(res_test, pred_test, squared=False)
rms
rms/test.shape[0]

r2_score = clf.score(test,res_test)
print(r2_score*100,'%')

















       
       
