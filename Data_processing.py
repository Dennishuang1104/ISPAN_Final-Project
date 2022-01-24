'''因為TMP_violation這張表都是逐項紀錄，無法看出單日的總共次數，所以需整理成每次稽核輕度違規，中度違規，重度違規各有幾次'''

import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

#讀取TMP_Full.csv
TMP = pd.read_csv(r'Dataset_TMP_Full(F1).csv')
TMP_Count = 417360

#刪除不需要的欄位
TMP.drop(['Unnamed: 0.1','Unnamed: 0','dbname','legalowner'],axis=1,inplace=True)

#將違規時間切割
violation_date = list(TMP['violdttm'])
Result_date = list(TMP['resultdttm'])
violation_date_revise=[]

#因為Result_date & Viodate 部分沒有填寫，將兩欄合併。
for i in range (0,TMP_Count):
    if violation_date[i] != ' ':
        violation_date_revise.append(violation_date[i])
    elif Result_date[i] != ' ':
        violation_date_revise.append(Result_date[i])
    else :
        violation_date_revise.append('without_date')
TMP['violdttm'] = pd.DataFrame(violation_date_revise)


#創建一個簡單要輸出的版本欄位
TMP_violation = pd.concat([TMP['businessname'],TMP['Business_ID'],TMP['violdttm'],TMP['viollevel'],TMP['TMP_ID'],TMP['comments'],TMP['violdesc']],axis=1)

#將沒有匹配到Business_id的內容刪掉
TMP_violation = TMP_violation.drop(TMP_violation[TMP_violation.Business_ID=='0'].index)

#將violevel 無違規填上零
TMP_violation['viollevel'] = TMP_violation['viollevel'].fillna('0')
TMP_violation['viollevel'] = TMP_violation['viollevel'].str.replace('-', '0')
TMP_violation['viollevel'] = TMP_violation['viollevel'].str.replace(' ', '0')

# 檢查次數是否有遺缺
TMP_violation['viollevel'].value_counts()

# 檢查日期 如有缺失則丟棄
TMP_violation = TMP_violation.drop(TMP_violation[TMP_violation.violdttm== 'without_date'].index)


# 將打亂的Index 號碼重新排列
TMP_violation.reset_index(inplace=True)

# 將violation 長度確認下來
violation=list(TMP_violation.violdttm)
Violation_Length = len(violation)

# 將時間整理成年/月/日 為了讓同一日檢查當作同一件事 去進行LabelEncoder
Date_list=[]
for i in range(0, Violation_Length):
    Time = pd.to_datetime(violation[i])
    Date = (Time + timedelta(days=0)).strftime("%Y/%m/%d")
    Date_list.append(Date)
TMP_violation['violdttm'] = pd.DataFrame(Date_list)


# 創造一欄在違規時間表欄加上編碼 以便判別違規時間
labelencoder = LabelEncoder()
TMP_violation['Time_Number'] = labelencoder.fit_transform(TMP_violation["violdttm"])

# 將違規打進行one hot-encoding
level = pd.get_dummies(TMP_violation.viollevel)
# 刪除 0  保留(* / ** / ***)
del level['0']

#結合Dummies 結果 與原本表格進行合併
TMP_violation = pd.concat([TMP_violation , level],axis=1)


# Grouby 因為順序有問題 如果遇到不同間餐廳 同一天會遇到編碼相同問題，所以必須使用迴圈
# TMP_violation.groupby('Time_Number')['*'].sum()   

##### 進行一顆星 兩顆星 三顆星的三次迴圈

level_1 = list(TMP_violation['*'])
level_2 = list(TMP_violation['**'])
level_3 = list(TMP_violation['***'])

Search_number = list(TMP_violation['Time_Number'])# 因為日期獨立編號進行

#一顆星
Leve1_1_Final=[]

S1= 0  # 代表一顆星的違規次數
for i in range(0,417556):
    if  i == 0:
        Leve1_1_Final.append(level_1[i]) #因為要前後檢查，先將第一項先append    
    if  i > 0:                          
        B_1 = Search_number[i-1]  # 確認前一項與後一項是否相同，如果不同號碼代表不同家  
        N1 = Search_number[i]
        C1 = level_1[i]          # 第N項的輕度違規的數量
        
        if B_1 == N1 :         #如果前一項與後一項相同，則append 
            if C1 ==0:
                Leve1_1_Final.append(S1)                 
            else:            #如果號碼不相同則相加 
                S1 = C1 + S1  
                Leve1_1_Final.append(S1)       
        else:
            Leve1_1_Final.append(C1)
            S1=C1
#兩顆星            
Leve1_2_Final=[]

S2= 0  #代表兩顆星的違規次數
for i in range(0,417556):
    if  i == 0:
        Leve1_2_Final.append(level_2[i]) #第一項先append入     
    if  i > 0:                          
        B_2 = Search_number[i-1]    # 確認前一項與後一項是否相同，如果不同號碼代表不同家 
        N1 = Search_number[i]
        C2 = level_2[i]            # 第N項的輕度違規的數量
        
        if B_2 == N1 :         #如果前一項與後一項相同，則append 
            if C2 ==0:
                Leve1_2_Final.append(S2)                 
            else:            #如果號碼不相同，代表還是同一間餐廳，則相加
                S = C2 + S2  
                Leve1_2_Final.append(S2)       
        else:
            Leve1_2_Final.append(C2)
            S2=C2

#三顆星
Leve1_3_Final=[]
TMP_NumberID = []

S3= 1  #代表三顆星的違規次數，因為第一項是1所以帶入1
for i in range(0,417556):
    if  i == 0:
        Leve1_3_Final.append(level_3[i]) #第一項先append入
        TMP_NumberID.append(i)
    if  i > 0:                          
        B_3 = Search_number[i-1]     # 確認前一項與後一項是否相同，如果不同號碼代表不同家 
        N1 = Search_number[i]
        C3 = level_3[i]              # 第N項的輕度違規的數量
        
        if i ==417555:
            TMP_NumberID.append("稽核檢查完畢") #最後一項填入
        
        elif B_3 == N1 :     #如果號碼相同則直接append S
            TMP_NumberID.append(i)
            if C3 ==0:
                Leve1_3_Final.append(S3)                 
            else:             #如果號碼不相同，代表還是同一間餐廳，則相加
                S3 = C3 + S3  
                Leve1_3_Final.append(S3)
            
        else:
            Leve1_3_Final.append(C3)
            S3=C3                      
            TMP_NumberID[i-1] = "稽核檢查完畢"  #為了之後清理方便，如果餐廳號碼不相同，則在前一項(還是相同的情況)填入稽核完畢
            TMP_NumberID.append(i)

# 合計完修改原本的Dataframe
TMP_violation['*']   = pd.DataFrame(Leve1_1_Final)
TMP_violation['**']  = pd.DataFrame(Leve1_2_Final)
TMP_violation['***'] = pd.DataFrame(Leve1_3_Final)
TMP_violation['***'] = TMP_violation['***'].fillna(0)

# 再新增一欄 為了抓出總共稽核幾次 將剛剛有稽查檢查完畢加入DF
TMP_violation['Time_Number2'] = pd.DataFrame(TMP_NumberID)

#保留最後一項(稽核檢查完畢)
TMP_Final = TMP_violation.drop(TMP_violation[TMP_violation.Time_Number2!='稽核檢查完畢'].index)

#最後成品
TMP_Final.to_csv('TMP_violation_level(F2).csv')
