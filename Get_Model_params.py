""" 運用爬山演算法找到局部最佳參數 """
import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Load Data
V1 = pd.read_csv("V1.csv")
X =  V1.iloc[:,0:45].values
Y =  V1.iloc[:,45:46].values

#訓練集測試集切割
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=1) #random_state 種子值

#標準化
scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#目前參數
params = {"n_estimators":500 , "n_jobs" : 6, "gamma" : 0.1 ,  "scale_pos_weight" : 0.2,
          "max_depth": 6 , "learning_rate":0.3}
         
#超參數的範圍(在此範圍找到最佳解) 
params_range = {"n_estimators": (200,1000) ,"max_depth":(5,7), "n_jobs" :(3,10), "gamma":[0.1 , 0.3] , "scale_pos_weight":[0.1,0.25,0.5],"learning_rate":[0.3,0.4,0.45,0.5,0.55]}

#定義隨機的參數 只有一個變數跳動

def get_new_params(old_params, params_range=params_range):
    params = old_params.copy()
    rand = np.zeros(len(params))
    rand[np.random.randint(len(params))] = 1
    for i, item in enumerate(params_range.items()):
        k, v = item[0], item[1]
        if rand[i]==1:
#             print(item[0])
#             print(item[1])
            if type(v) == tuple:
                var_max = ((v[1] - v[0]) //10) +1
                while(params[k]==old_params[k]):
#                     print(params[k])
#                     print(old_params[k])
                    if (params[k] < np.random.randint(v[0], v[1])):
                        params[k] = params[k] + (np.random.randint(1, var_max) if 1!=var_max else 1)
                    else:
                        params[k] = params[k] - (np.random.randint(1, var_max) if 1!=var_max else 1)
                    if params[k] > v[1]:
                        params[k] = v[1]
                    if params[k] < v[0]:
                        params[k] = v[0]
            elif type(v) == list:
                while(params[k]==old_params[k]):
                    params[k] = v[np.random.randint(len(v))]
    return params


print(params)
NEW =print(get_new_params(params))

# 輸入需要做幾次迭代
iters = int(input("enter the number of iterations:"))

model = XGBClassifier(**params)     # 選用模型
error = model.score(X_test, y_test) # 將模型分數assign成 Model.score
error_history = []
for i in range(iters):
    if i%20==0:                     # 每20次列印出結果
      print("current iteration=",i)
    new_params = get_new_params(params_range)  #用先定義的函式隨機丟入參數
    print(new_params)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    new_error = model.score(X_test,y_test)
    if new_error > error:   #如果新的準確率大於舊的準確率則用目前的參數
        error = new_error
        params = new_params
        error_history.append(new_error)
        print("################################")
        print("new error occurs at number of iterations=",i)
        print(new_error)
        print("new parameters",params)
 
#秀出最後的error 以及參數
print("final error",error)
print("final parameters",params)