#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from tqdm import tqdm
import time
import FinanceDataReader as fdr
from multiprocessing import Pool
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def make_code(x):
    x = str(x)
    return '0'*(6-len(x)) + x

code_data = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]
code_data['종목코드'] = code_data['종목코드'].apply(make_code)
code_list = code_data[code_data['상장일'] < '2020-01-01']['종목코드']

def merging_stock_data(code):
    merge_stock_list = []
    stock_list = fdr.DataReader(code, '2020').reset_index().values.tolist()
    for row in stock_list:
        row.append(code)
        merge_stock_list.append(row)
    return merge_stock_list

start_time = time.time()

with Pool(32) as p:  
    result = p.map(merging_stock_data, code_list)

end_time = time.time()
print(f"Time taken for multiprocessing: {end_time - start_time} seconds")

result_flattened = [item for sublist in result for item in sublist]
columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Code']
stock_df = pd.DataFrame(result_flattened, columns=columns)

stock_df['Timestamp'] = pd.to_datetime(stock_df['Timestamp'])

print(f"Shape of the DataFrame: {stock_df.shape}")

stock_df['up'] = (stock_df['Change'] > 0).astype(int)

train_input, test_input, train_target, test_target = train_test_split(
    stock_df[['Open', 'High', 'Low', 'Close', 'Volume']], 
    stock_df['up'], 
    test_size=0.2, 
    random_state=42
)

param_fixed = {
    'learning_rate': 0.1,
    'max_depth': 10,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.2
}

start_time_xg = time.time()

xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, **param_fixed)
xgb_model.fit(train_input, train_target)

print(f"Train Score of XGBoost: {xgb_model.score(train_input, train_target)}")
print(f"Test Score of XGBoost: {xgb_model.score(test_input, test_target)}")

probabilities = xgb_model.predict_proba(test_input)
log_loss_value = log_loss(test_target, probabilities)
print(f"Log loss of XGBoost: {log_loss_value:.4f}")

end_time_xg = time.time()
print(f"Time taken for XGBoost training and evaluation: {end_time_xg - start_time_xg} seconds")

