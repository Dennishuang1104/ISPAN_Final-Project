'''針對Violation_level 進行特徵工程並把探索性的資料儲存起來'''


import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#每次檢查的稽核狀態
Violation = pd.read_csv("./Dataset_TMP_violation_level(F2).csv")
#TMP 檢查表
TMP_FULL = pd.read_csv("./Dataset_TMP_Full(F1).csv")

#每家餐廳狀況
Business = pd.read_csv(r"C:\Users\User\BDSE_Final_project\Step2 Data_preprocessing\Finish\Dataset_Business(F1) .csv")


def Get_LabelEncoder(DataSeries):
    labelencoder = LabelEncoder()
    DataSeries = labelencoder.fit_transform(DataSeries)
    DataFrame = pd.DataFrame(DataSeries)
    return DataFrame


def Get_Mapping_dict(df1, df2):
    L1 = list(df1)
    L2 = list(df2)
    Dict = dict(zip(L1,L2))
    return Dict


##########1 加入上一次違規紀錄到 目標Y 表格
Predict_one=[]
Predict_two=[]
Predict_three=[]

for i in range(0,81439):
    if i ==0:
        Predict_one.append(0)
        Predict_two.append(0)
        Predict_three.append(0)
        
    elif Violation['Business_number'][i] ==Violation['Business_number'][i-1]:
        Predict_one.append(Violation['*'][i-1])
        Predict_two.append(Violation['**'][i-1])
        Predict_three.append(Violation['***'][i-1])
        
    else:
        Predict_one.append(0)
        Predict_two.append(0)
        Predict_three.append(0)

#帶回Violation 這張表
Violation['Pred_*'] = pd.DataFrame(Predict_one)
Violation['Pred_**'] = pd.DataFrame(Predict_two)
Violation['Pred_***'] = pd.DataFrame(Predict_three)
Violation.to_csv("./Dataset_TMP_violation_level(F2).csv")

##########2 針對City

TMP_Evaluate = TMP_FULL 
# 字典對應Business_IP 對應 City_code
BUS_City_code = Get_Mapping_dict(TMP_Evaluate['Business_ID'] , TMP_Evaluate['new_city'])

#城市對應號碼Code
City_dict = Get_Mapping_dict(TMP_Evaluate['City_code'] , TMP_FULL['new_city'])

# Mapping 看各City的違規次數
Violation['City'] = Violation['Business_ID'].map(BUS_City_code)
Violation_City = Violation.groupby('City')['Violation_times'].sum() # Assign 各City的違規次數
Violation.to_csv("./Dataset_TMP_violation_level(F2).csv")

##########3 增加Violation Accmulate

## 製造出每次違規累加
Vio_times = list(Violation['Violation_tims'])
Bus_number = list(Violation['Business_number'])

Vio_T=[]
for i in range(0,81439):
    if i ==0:
        Vio_T.append(0)
    elif Bus_number[i] == Bus_number[i-1]:
        C = Vio_T[i-1]+ Vio_times[i]    #前一項+迴圈當次 得到當天累積次數
        Vio_T.append(C)
        Vio_times[i] = C
    
    elif Bus_number[i-1] != Bus_number[i]:
        Vio_T.append(Vio_times[i])
Violation['Violation_Accumulate'] = pd.DataFrame(Vio_T)
Violation.to_csv("./Dataset_TMP_violation_level(F2).csv")



##########4 製造出Violation_change

# 清理Business_number & violdttn  進行排序 並且群組到Business_number之中
Violation_Clean = Violation.sort_values(['Business_number','violdttm'],ascending=True).groupby('Business_number').head(89155)
Violation_Clean.reset_index(drop=True, inplace=True)

# 為了製造出新的餐廳編碼 並把沒有檢查到的餐廳drop掉
V_C = Violation_Clean.drop_duplicates(subset='Business_number')

ID = Get_Mapping_dict(V_C['Business_ID'], V_C['Business_number'])
Business['Business_Number2'] = Business.business_id.map(ID)
Business = Business.dropna(subset=['Business_Number2'])  #從3486項 到 3478項
#重新編排
Business = Business.sort_values(by='Business_Number2')

#為了增加"總違規次數"所map的字典
V_T = dict(Violation.groupby('Business_ID')["Violation_times"].sum()) 
Business['Violation_times'] = Business.business_id.map(V_T)

#為了增加"總違規檢查天數"所map的字典
Bus_time = dict(Violation_Clean['Business_number'].value_counts())
Business["Violation_counts"] = Business.Business_Number2.map(Bus_time)

#重新編排號碼
Business.reset_index(drop=True, inplace=True)


Bus_n = list(Violation_Clean.Business_number)  # Business_number 號碼
Vio_t = list(Violation_Clean.Violation_times)  # Vio_times 違規次數
##此迴圈是製造出已經排列好順序的Violation_Clean稽核順序
First = 0
L = []
for i in range (0,81439):    
    if i ==0:
        L.append("First_time")    
    else: 
        N1 = Bus_n[i]
        N2 = Bus_n[i-1]
        V2 = Vio_t[i-1]
        V1 = Vio_t[i]
        if N1 == N2:          #確認兩者號碼相同
            if V1 - V2 > 0:  #代表這次檢查次數沒有比上次好
                L.append(1)  
            else:
                L.append(0)  #代表這次檢查違規次數比上次好
        else:
            L.append("First_time")
Violation_Clean['Change_Violation'] = pd.DataFrame(L)
Violation_Clean.to_csv("./Dataset_TMP_violation_level(F2).csv")

##########5 製造出Better_rate 

# 此迴圈是用來尋找單一餐廳的違規變化 是否變好或是變差
Worse=[]
Better=[]
a=0
b=0
for i in range(0,81439):
    
    if i ==0:
        continue
    if i ==81438:           #最後一項直接填入
        Better.append(a)
        Worse.append(b)
    elif L[i] == 0 :
        a=a+1
    elif L[i] == 1 :
        b=b+1 
    elif L[i] =='First_time':  #運用上一個list 如果是第一次稽核則讓次數歸零 
        Better.append(a)
        Worse.append(b)
        a=0
        b=0   
Business['Become_better'] = pd.DataFrame(Better)
Business['Become_worse']  = pd.DataFrame(Worse)
Business["Better_rate"] = Business["Become_better"] / (Business["Become_better"] + Business["Become_worse"])

Business.to_csv("Dataset_Business(F1) .csv")

##########6 Zip 郵遞區號編碼 探索

ZIP_code = Get_LabelEncoder(TMP_FULL.zip)
TMP_FULL['Zip_code'] = ZIP_code

#先做成字典以便mapping
TMP_Bus_zipcode = Get_Mapping_dict(TMP_FULL.Zip_code , TMP_FULL.zip)
Zip_code = Get_Mapping_dict(TMP_FULL.Zip_code , TMP_FULL.zip)

Violation['Zip_code'] = Violation.Business_ID.map(TMP_Bus_zipcode)

#進行One-Hot Encoding
One_hot_zip =pd.get_dummies(Violation.Zip_code)

#郵遞區號 與違規次數輸出
Zip_Zone = pd.concat([One_hot_zip,Violation['*'],Violation['**'],Violation['***']],axis=1)
Zip_Zone.to_csv("Zip.csv")


##########7 對年份進行分析
Year = []
for i in range(0 , 81439):
    Y = pd.to_datetime(Violation['violdttm'][i]).year
    Year.append(Y)

Violation['Year'] = pd.DataFrame(Year,columns=['Year'])

#每年違規情形
Year_Violation = Violation.groupby(Year)['Violation_times'].sum()
plt.figure(figsize=(8,6))
plt.bar(Year_Violation.index , Year_Violation.values ,alpha=1, width=0.8)
plt.xticks(Year_Violation.index, rotation=45 ,fontsize=9.87)
plt.xlabel('Year', fontsize=13 )
plt.ylabel('Violation_times', fontsize=13 )
plt.title('Yearly_Violations',fontsize=16)
plt.show()


#每年輕度中規違規情形
two_star = Violation.groupby(Year)['**'].sum()
plt.figure(figsize=(8,6))
plt.bar(Year_Violation.index , Year_Violation.values ,alpha=1, width=0.8)
plt.xticks(Year_Violation.index, rotation=45 ,fontsize=9.87)
plt.xlabel('Year', fontsize=13 )
plt.ylabel('Violation_times', fontsize=13 )
plt.title('*_Violations',fontsize=16)
plt.show()

#將中度違規轉換成DF
Two_star = pd.DataFrame(two_star)

#將中度違規進行非監督式分群
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

X =Two_star.iloc[:,0:1].values
scaler = preprocessing.StandardScaler().fit(X)
X_Processing = scaler.transform(X)

db = DBSCAN(eps=0.8, min_samples=2)\
            .fit(X_Processing)
labels = db.labels_

print('cluster on X {}'.format(labels))
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  #set可顯示不重複
print('number of clusters: {}'.format(n_clusters))

two_star_L=[]
for i in range(0,81439):
    if Violation.Year[i] < 2019:
        two_star_L.append(0)
    else:
        two_star_L.append(1)

Two = pd.DataFrame(two_star_L)
Violation["**_features"]=Two

Violation.to_csv('Dataset_TMP_violation_level(F2).csv')

##########8 Category 餐廳種類進行探索
Cat = pd.read_csv('category_significant2.csv')

# 所有城市Mappping 創建字典
A_dict = Get_Mapping_dict(Cat.business_id  , Cat.american)
B_dict = Get_Mapping_dict(Cat.business_id  , Cat.art)
C_dict = Get_Mapping_dict(Cat.business_id  , Cat.bar)
D_dict = Get_Mapping_dict(Cat.business_id  , Cat.beauty)
E_dict = Get_Mapping_dict(Cat.business_id  , Cat['breakfast brunch'])
F_dict = Get_Mapping_dict(Cat.business_id  , Cat.cafe)
G_dict = Get_Mapping_dict(Cat.business_id  , Cat.chinese)
H_dict = Get_Mapping_dict(Cat.business_id  , Cat.fastfood)
I_dict = Get_Mapping_dict(Cat.business_id  , Cat.fitness)
J_dict = Get_Mapping_dict(Cat.business_id  , Cat.gluten)
K_dict = Get_Mapping_dict(Cat.business_id  , Cat['health medical'])
L_dict = Get_Mapping_dict(Cat.business_id  , Cat['icecream frozenyogurt'])
M_dict = Get_Mapping_dict(Cat.business_id  , Cat['juicebars smoothy'])
O_dict = Get_Mapping_dict(Cat.business_id  , Cat['new restaurant'])
P_dict = Get_Mapping_dict(Cat.business_id  , Cat.nightlife)
Q_dict = Get_Mapping_dict(Cat.business_id  , Cat.spirit)
R_dict = Get_Mapping_dict(Cat.business_id  , Cat.venue)
S_dict = Get_Mapping_dict(Cat.business_id  , Cat.wine)
E1_dict = Get_Mapping_dict(Cat.business_id  , Cat.caribbean)
H1_dict = Get_Mapping_dict(Cat.business_id  , Cat.fooddeliveryservices)
H2_dict = Get_Mapping_dict(Cat.business_id  , Cat.foodtrucks)
H3_dict = Get_Mapping_dict(Cat.business_id  , Cat.french)
M1_dict = Get_Mapping_dict(Cat.business_id  , Cat.korean)
M2_dict = Get_Mapping_dict(Cat.business_id  , Cat.latinamerican)
M3_dict = Get_Mapping_dict(Cat.business_id  , Cat.middleeastern)
L1_dict = Get_Mapping_dict(Cat.business_id  , Cat.indian)
P1_dict = Get_Mapping_dict(Cat.business_id  , Cat.pizza)
P2_dict = Get_Mapping_dict(Cat.business_id  , Cat['restaurant seafood'])
Q1_dict = Get_Mapping_dict(Cat.business_id  , Cat.steakhouse)
Q2_dict = Get_Mapping_dict(Cat.business_id  , Cat.sushibars)
Q3_dict = Get_Mapping_dict(Cat.business_id  , Cat.thai)
Q4_dict = Get_Mapping_dict(Cat.business_id  , Cat.travel)
Q5_dict = Get_Mapping_dict(Cat.business_id  , Cat.vietnamese)

# 進行Mapping 

Violation['american'] = Violation['Business_ID'].map(A_dict)
Violation['art'] = Violation['Business_ID'].map(B_dict)
Violation['bar'] = Violation['Business_ID'].map(C_dict)
Violation['beauty'] = Violation['Business_ID'].map(D_dict)
Violation['breakfast brunch'] = Violation['Business_ID'].map(E_dict)
Violation['cafe'] = Violation['Business_ID'].map(F_dict)
Violation['chinese'] = Violation['Business_ID'].map(G_dict)
Violation['fastfood'] = Violation['Business_ID'].map(H_dict)
Violation['fitness'] = Violation['Business_ID'].map(I_dict)
Violation['gluten'] = Violation['Business_ID'].map(J_dict)
Violation['health medical'] = Violation['Business_ID'].map(K_dict)
Violation['icecream frozenyogurt'] = Violation['Business_ID'].map(L_dict)
Violation['juicebars smoothy'] = Violation['Business_ID'].map(M_dict)
Violation['new restaurant'] = Violation['Business_ID'].map(O_dict)
Violation['nightlife'] = Violation['Business_ID'].map(P_dict)
Violation['spirit'] = Violation['Business_ID'].map(Q_dict)
Violation['venue'] = Violation['Business_ID'].map(R_dict)
Violation['wine'] = Violation['Business_ID'].map(S_dict)

##############################################################
Violation['caribbean'] = Violation['Business_ID'].map(E1_dict)
Violation['fooddeliveryservices'] = Violation['Business_ID'].map(H1_dict)
Violation['foodtrucks'] = Violation['Business_ID'].map(H2_dict)
Violation['french'] = Violation['Business_ID'].map(H3_dict)
Violation['korean'] = Violation['Business_ID'].map(M1_dict)
Violation['latinamerican'] = Violation['Business_ID'].map(M2_dict)
Violation['middleeastern'] = Violation['Business_ID'].map(M3_dict)
Violation['indian'] = Violation['Business_ID'].map(L1_dict)
Violation['pizza'] = Violation['Business_ID'].map(P1_dict)
Violation['restaurant seafood'] = Violation['Business_ID'].map(P2_dict)
Violation['steakhouse'] = Violation['Business_ID'].map(Q1_dict)
Violation['sushibars'] = Violation['Business_ID'].map(Q2_dict)
Violation['thai'] = Violation['Business_ID'].map(Q3_dict)
Violation['travel'] = Violation['Business_ID'].map(Q4_dict)
Violation['vietnamese'] = Violation['Business_ID'].map(Q5_dict)

#查看相關係數判斷相關程度
Violation.corr()

