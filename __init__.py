import numpy as np 
import pandas as pd 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import xgboost as xgb
import operator
import warnings

warnings.filterwarnings('ignore')

# n_splites=5
n_splites=3
seed=42
params={
    'objective':'multi:softmax',
    'num_class':11,
    'gamma':0.001,
#     'eval_metric':'merror',
    'max_depth':10,
    'lambda':0.3,
    'subsample':0.9,
#     'gpu_id':0,
#     'updater':'grow_gpu',
#     'tree_method':'gpu_hist',
    'colsample_bytree':0.9,
    'scale_pos_weight':1.5,
    'min_child_weight': 0.7,
    'silient':1,
    'eta':0.1,
    'seed':1000,
    'n_estimators':150,
    }

train_data=pd.read_csv('train_all.csv',delimiter=',',low_memory=False)
test_data=pd.read_csv('republish_test.csv',delimiter=',',low_memory=False)
train_data=train_data.drop('user_id', axis=1)
label2current_service=dict(zip(range(0,len(set(train_data['current_service']))),sorted(list(set(train_data['current_service'])))))
current_service2label=dict(zip(sorted(list(set(train_data['current_service']))),range(0,len(set(train_data['current_service'])))))

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


train_col=train_data.columns

for i in train_col:
    if i !='current_service':
        train_data[i]=train_data[i].replace('\\N',-1)
        test_data[i]=test_data[i].replace('\\N',-1)
        
print(train_data['gender'].value_counts())
print(train_data['age'].unique())
print(train_data['contract_type'].value_counts())
print(train_data['net_service'].value_counts())
print(train_data['service_type'].value_counts())

train_data['gender']=train_data['gender'].astype('int64')
train_data['2_total_fee']=train_data['2_total_fee'].astype('float64')
train_data['3_total_fee']=train_data['3_total_fee'].astype('float64')
train_data['age']=train_data['age'].astype('int64')

test_data['gender']=test_data['gender'].astype('int64')
test_data['2_total_fee']=test_data['2_total_fee'].astype('float64')
test_data['3_total_fee']=test_data['3_total_fee'].astype('float64')
test_data['age']=test_data['age'].astype('int64')

train_data=train_data.drop(train_data[train_data['contract_type']==8].axes[0],axis=0)
train_data=train_data.drop(train_data[train_data['gender']==-1].axes[0],axis=0)
train_data=train_data.drop(train_data[train_data['age']==-1].axes[0],axis=0)
train_data=train_data.reset_index(drop=True)


Y=train_data['current_service'].map(current_service2label)
train_data=train_data.drop('current_service',axis=1)
# print(Y.value_counts())

train_data.loc[train_data.service_type==3,'last_month_traffic']=train_data.loc[train_data.service_type==3].last_month_traffic/1024.0
# train_data['last_month_traffic']=train_data['last_month_traffic']/1024.0
train_data['month_traffic']=train_data['month_traffic']/1024.0
train_data['contract_time']=train_data['contract_time']/12.0
train_data['local_trafffic_month']=train_data['local_trafffic_month']/1024.0
train_data['former_complaint_fee']=train_data['former_complaint_fee']/100.0
 
test_data.loc[test_data.service_type==3,'last_month_traffic']=test_data.loc[test_data.service_type==3].last_month_traffic/1024.0
# test_data['last_month_traffic']=test_data['last_month_traffic']/1024.0
test_data['month_traffic']=test_data['month_traffic']/1024.0
test_data['contract_time']=test_data['contract_time']/12.0
test_data['local_trafffic_month']=test_data['local_trafffic_month']/1024.0
test_data['former_complaint_fee']=test_data['former_complaint_fee']/100.0

train_data['total_fee_weight']=0.8*train_data['1_total_fee']+0.5*train_data['2_total_fee']+0.2*(train_data['3_total_fee']+train_data['4_total_fee'])
# train_data['total_free_mean']=train_data['1_total_fee']+train_data['2_total_fee']+train_data['3_total_fee']+train_data['4_total_fee']
# train_data['total_free_mean']=train_data['total_free_mean']/4.0
# mean=train_data['total_free_mean'].mean()
# var=train_data['total_free_mean'].var()
# train_data['total_free_mean']=train_data['total_free_mean']-mean
# print(train_data['total_free_mean'])
# train_data['total_free_mean']=train_data['total_free_mean']/np.sqrt(var)
# print(train_data['total_free_mean'])
# train_data['former_complaint_importance']=train_data['former_complaint_num'].mul(train_data['complaint_level'],axis=0)
#  
# train_data['each_pay']=train_data['pay_num'].div(train_data['pay_times'],axis=0)
# 
# 
test_data['total_fee_weight']=0.8*test_data['1_total_fee']+0.5*test_data['2_total_fee']+0.2*(test_data['3_total_fee']+test_data['4_total_fee'])
# test_data['total_free_mean']=test_data['1_total_fee']+train_data['2_total_fee']+train_data['3_total_fee']+train_data['4_total_fee']
# test_data['total_free_mean']=test_data['total_free_mean']/4.0
# test_data['total_free_mean']=test_data['total_free_mean']-mean
# test_data['total_free_mean']=test_data['total_free_mean']/np.sqrt(var)
# test_data['former_complaint_importance']=test_data['former_complaint_num'].mul(test_data['complaint_level'],axis=0)
# test_data['each_pay']=test_data['pay_num'].div(test_data['pay_times'],axis=0)

# train_data=train_data.drop('1_total_fee',axis=1)
# train_data=train_data.drop('2_total_fee',axis=1)
# train_data=train_data.drop('3_total_fee',axis=1)
# train_data=train_data.drop('4_total_fee',axis=1)
#  
# test_data=test_data.drop('1_total_fee',axis=1)
# test_data=test_data.drop('2_total_fee',axis=1)
# test_data=test_data.drop('3_total_fee',axis=1)
# test_data=test_data.drop('4_total_fee',axis=1)

cols=['service_type','net_service','gender','contract_type']
 
for i in cols:
    ont_hot=pd.get_dummies(train_data[i],prefix=i)
    one_hot_DP=pd.DataFrame(ont_hot)
    train_data=pd.concat([train_data,one_hot_DP], axis=1)
    train_data=train_data.drop(i, axis=1)
     
for i in cols:
    ont_hot=pd.get_dummies(test_data[i],prefix=i)
    one_hot_DP=pd.DataFrame(ont_hot)
    test_data=pd.concat([test_data,one_hot_DP], axis=1)
    test_data=test_data.drop(i, axis=1)


# print(train_data.info())
train_cols_new=train_data.columns
X=train_data
X_test=test_data[train_cols_new]
test_id=test_data['user_id']

X,Y,X_test=X.values,Y,X_test.values


def f1_score_vali(preds,data_vali):
    labels=data_vali.get_label()
    score_vali=f1_score(y_true=labels, y_pred=preds,average='weighted')
    return 'f1_score',-score_vali

xx_score=[]
# cv_score=[]

skf=StratifiedKFold(n_splits=n_splites,random_state=seed,shuffle=True)
for index,(train_index,test_index) in enumerate(skf.split(X,Y)):
    X_train,X_valid,Y_train,Y_valid=X[train_index],X[test_index],Y[train_index],Y[test_index]
    train=xgb.DMatrix(X_train,Y_train)
    validation=xgb.DMatrix(X_valid,Y_valid)
    X_val=xgb.DMatrix(X_valid)
    test=xgb.DMatrix(X_test)
    wachlist=[(validation,'val')]
    bst=xgb.train(params,train,10000,wachlist,early_stopping_rounds=15,feval=f1_score_vali,verbose_eval=1)
    
    features = [x for x in train_data.columns if x not in ['user_id','current_service']]
    ceate_feature_map(features)
    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("../feat_importance.csv", index=False)
    
#     plt.figure()
#     df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 15))
#     plt.title('XGBoost Feature Importance')
#     plt.xlabel('relative importance')
#     plt.show()


    xx_pred=bst.predict(X_val)
    xx_score.append(f1_score(Y_valid,xx_pred,average='weighted'))
    Y_test=bst.predict(test)
    Y_test=Y_test.astype('int64')
    print(Y_test.shape)
    if index==0:
        cv_pred=np.array(Y_test).reshape(-1,1)
    else:
        cv_pred=np.hstack((cv_pred,np.array(Y_test).reshape(-1,1)))
    print(cv_pred)
      
 
submit=[]
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
    print(np.argmax(np.bincount(line)))
 
 
df_test=pd.DataFrame()
df_test['user_id']=list(test_id.unique())
df_test['current_service']=submit
df_test['current_service']=df_test['current_service'].map(label2current_service)
df_test.to_csv('sub2.csv',index=False)
print(xx_score,np.mean(xx_score))

# X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3, random_state=42)
# train=xgb.DMatrix(X_train,y_train)
# validation=xgb.DMatrix(X_valid,y_valid)
# X_val=xgb.DMatrix(X_valid)
# test=xgb.DMatrix(X_test)
# wachlist=[(validation,'val')]
# bst=xgb.train(params,train,10000,wachlist,early_stopping_rounds=15,feval=f1_score_vali,verbose_eval=1)
# xx_pred=bst.predict(X_val)  
# xx_score.append(f1_score(y_valid,xx_pred,average='weighted'))  
# Y_test=bst.predict(test) 
# Y_test=Y_test.astype('int64')    
#  
# df_test=pd.DataFrame()
# df_test['user_id']=list(test_id.unique())
# df_test['current_service']=Y_test
# df_test['current_service']=df_test['current_service'].map(label2current_service)
# df_test.to_csv('sub2.csv',index=False)    
     
    
    
    
    
