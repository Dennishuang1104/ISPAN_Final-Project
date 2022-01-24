# -*- coding: utf-8 -*-
"""將最後特徵工程結果輸入模型"""
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score #評估指標
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

'''輕度違規'''
'''進行輕度違規的分類模型''' 
#Step1 Load Data 
V1 = pd.read_csv("V1.csv")

#Step2 prepare X　Y
X =  V1.iloc[:,0:45].values
Y =  V1.iloc[:,45:46].values

# 測試集進行 切成 70% 為訓練集 30% 為測試集
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=31) #random_state 種子值

scaler = preprocessing.StandardScaler().fit(X) # 進行正規化
X_ = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Step3 使用訓練資料訓練模型
model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                       importance_type='split', learning_rate=0.1, max_depth=-1,
                       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                       n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                       random_state=123, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

model.fit(X_train , y_train) 

preY2 = model.predict(X)   #預測Y值

# Step4 Evaluate Model
ACC = accuracy_score(Y,preY2)
print(f'準確率為:{ACC}')
V1['Pre*_V'] = model.predict(X)
from sklearn.metrics import plot_confusion_matrix
'''畫出輕度違規的混淆矩陣'''
cm = plot_confusion_matrix( model, X , Y, cmap=plt.cm.Oranges)
plt.show()

'''進行輕度違規的回歸模型'''

# Step1 Load Data 
V1R = pd.read_csv("RV1.1.csv")


# Step2 定義X , Y
X = V1R.iloc[:,0:47].values
Y = V1R.iloc[:,47:48].values

# 將資料切成訓練集&測試集
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.2,random_state=1) 

# 進行標準化
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

# Step3 Build Model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',\
                          init=None, learning_rate=0.1, loss='ls', max_depth=3,\
                          max_features=None, max_leaf_nodes=None,\
                          min_impurity_decrease=0.0, min_impurity_split=None,\
                          min_samples_leaf=2, min_samples_split=2,\
                          min_weight_fraction_leaf=0.0, n_estimators=100,\
                          n_iter_no_change=None,\
                          random_state=123, subsample=1.0, tol=0.0001,\
                          validation_fraction=0.1, verbose=0, warm_start=False)
model.fit(X_train,y_train)

# Step4 Evaluate Model 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2: {}'.format(r2))
print('mse: {}'.format(mse))


#將預測結果丟回原本的Dateframe

V1R["Predict_*(times)"] = pd.DataFrame(model.predict(X))

#
V1R['Final_Result'] = abs(V1R['*']- V1R['Predict_*(times)'])
Len = V1R.index.stop
FR = list(V1R['Final_Result'])
One_Star_halfstd = 1  # 差距使用1.5顆星

#跑出結果輸出至原本DF
Result = []
for i in range(0,Len):
    if FR[i] < One_Star_halfstd :   #代表正確
        Result.append(0)
    else:
        Result.append(1)
        
V1R['Result'] = pd.DataFrame(Result)
Correct = V1R['Result'].value_counts()[0]
Wrong  =  V1R['Result'].value_counts()[1]
ACC2 = Correct / (Correct+Wrong) 
print(F"Offset_準確率為{ACC2}")

'''輕度違規最終結果'''
Total = 43689 + 21066   #輕度違規混淆矩陣的TP FP 數據 
ACC3 =(43689/Total)*ACC +(21066/Total)*ACC2*ACC
print(f"最終的準確率為{ACC3}")

'''中度違規'''
'''進行中度違規的分類模型'''

from lightgbm import LGBMClassifier

#Step1 Load Data 
V2 = pd.read_csv("V2.csv")
#Step2 prepare X　Y
X =  V2.iloc[:,0:45].values
Y =  V2.iloc[:,45:46].values

# 切割成訓練集以及測試集
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=1) #random_state 種子值

# 進行標準化
scaler = preprocessing.StandardScaler().fit(X)
X_ = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                       importance_type='split', learning_rate=0.1, max_depth=-1,
                       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                       n_estimators=120, n_jobs=-1, num_leaves=31, objective=None,
                       random_state=123, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

# 使用訓練資料訓練模型
model.fit(X_train , y_train)
preY2 = model.predict(X)
ACC = accuracy_score(Y,preY2)
print(f'準確率為:{ACC}')
print('##########################')
print('訓練集: ',model.score(X_train,y_train))
print('測試集: ',model.score(X_test,y_test))
V2['Pre*_V'] = model.predict(X)

'''畫出分類的混淆矩陣'''
cm = plot_confusion_matrix( model, X , Y, cmap=plt.cm.Oranges)
plt.show()

'''進行中度違規的回歸模型'''
# Step 1 Load Data 
V2R = pd.read_csv("RV2.1.csv")

# Step 2 進行切割
X = V2R.iloc[:,2:48].values
Y = V2R.iloc[:,52:53].values
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.2,random_state=1) #random_state 種子值

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

# Step3 Build Model
from catboost import CatBoostRegressor
model = CatBoostRegressor(random_state=42,
                         loss_function='RMSE',
                         eval_metric='RMSE',
                         use_best_model=True)


#模型評估指標
model.fit(X_train,y_train, eval_set=(X_test, y_test), verbose=0, plot=True)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2: {}'.format(r2))
print('mse: {}'.format(mse))


#將預測結果丟回原本的Dateframe
Two_Star_halfstd = 1
V2R["Predict_*(times)"] = pd.DataFrame(model.predict(X))
V2R['Final_Result'] = abs(V2R['**']- V2R['Predict_*(times)'])
Len = V2R.index.stop
FR = list(V2R['Final_Result'])

# 將結果帶回原本的DF
Result = []
for i in range(0,Len):
    if FR[i] < Two_Star_halfstd :   #代表正確
        Result.append(0)
    else:
        Result.append(1)
        
V2R['Result'] = pd.DataFrame(Result)
Correct = V2R['Result'].value_counts()[0]
Wrong  =  V2R['Result'].value_counts()[1]
ACC2 = Correct / (Correct+Wrong) 
print(F"Offset_準確率為{ACC2}")

'''最終 中度違規準確率'''
Total = 54971 + 13436
ACC3 =(54971/Total)*ACC +(13436/Total)*ACC2*ACC
print(f"最終的準確率為{ACC3}")


'''重度違規'''
'''進行重度違規的分類模型'''

#Step1 Load Data 
V3 = pd.read_csv("V3.csv")

#Step2 prepare X　Y
X =  V3.iloc[:,0:45].values
Y =  V3.iloc[:,45:46].values

# 切割訓練集與訓練集
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=1) #random_state 種子值

# 進行標準化
scaler = preprocessing.StandardScaler().fit(X)
X_ = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=1) #random_state 種子值

scaler = preprocessing.StandardScaler().fit(X)
X_ = scaler.transform(X)

model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                       importance_type='split', learning_rate=0.1, max_depth=-1,
                       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                       n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                       random_state=123, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

# Step3 Build model 
model.fit(X , Y)


# Step 4 Evaluate Model 
preY2 = model.predict(X)
ACC = accuracy_score(Y , preY2)
print(f'準確率為:{ACC}')
V3['Pre*_V'] = model.predict(X)

'''重度違規的分類 混淆矩陣'''
cm = plot_confusion_matrix( model, X , Y, cmap=plt.cm.Oranges)


'''進行重度違規的分類模型'''
# Step1 Load Data 
V3R = pd.read_csv("RV3.1.csv")


# Step2 Prepare X , y 
X = V3R.iloc[:,0:50].values
Y = V3R.iloc[:,50:51].values
# 測試集以及訓練集
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=1) #random_state 種子值
# 進行標準化
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

# Step3 Build Model
Model = CatBoostRegressor(random_state=42,
                         loss_function='RMSE',
                         eval_metric='RMSE',
                         use_best_model=True)

Model.fit(X_train,y_train, eval_set=(X_test, y_test), verbose=0, plot=True)

# Step4 Evaluate Model 模型評估指標
y_pred = Model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2: {}'.format(r2))
print('mse: {}'.format(mse))

# 評估adjusting R 
adj_r_squared = r2 - (1 - r2) * (X.shape[1] / (X.shape[0] - X.shape[1] - 1))
print(adj_r_squared)

V3R["Predict_*(times)"] = pd.DataFrame(Model.predict(X))
Three_Star_halfstd = 1
V3R['Final_Result'] = abs(V3R['***']- V3R['Predict_*(times)'])
Len = V3R.index.stop
FR = list(V3R['Final_Result'])

# 將結果帶回原本的DF
Result = []
for i in range(0,Len):
    if FR[i] < Three_Star_halfstd :   #代表正確
        Result.append(0)
    else:
        Result.append(1)
        
V3R['Result'] = pd.DataFrame(Result)
Correct = V3R['Result'].value_counts()[0]
Wrong  =  V3R['Result'].value_counts()[1]
ACC2 = Correct / (Correct+Wrong) 
print(F"Offset_準確率為{ACC2}")

'''最終重度違規的結果'''
Total = 43456 + 20963  #混淆矩陣的結果
ACC3 =(43456/Total)*ACC +(20963/Total)*ACC2*ACC


print(f"最終的準確率為{ACC3}")