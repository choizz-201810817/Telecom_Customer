#%%
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from dataprep.eda import create_report
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost

import pymysql
import sqlalchemy
from sqlalchemy import create_engine

import pandas as pd

from md_sql import pushToDB, pullFromDB
from md import histPlot, distPlot, playML, getLC, getRoc
# %%
password = '20130511'
dbName = 'telecomDB'
tableName = 'customer'
df = pullFromDB(password, dbName, tableName)
print(f"df shape: {df.shape}")
df.head()
# %%
print(df.info())
# %%
objDf = df.select_dtypes('object')
numDf = df.select_dtypes('number')
objCols = list(objDf)
numCols = list(numDf)

print(f"length of objCols: {len(objCols)}")
print(f"lengthof  numCols: {len(numCols)}")

# %%
#범주형 변수 시각화
figsize = (16,30)
cols = objCols
nrow = 6
ncol = 3
df = df
histPlot(df, figsize, cols, nrow, ncol)

# %%
#연속형 변수 시각화
figsize = (15,6)
cols = numCols
nrow = 1
ncol = 3
df = df
distPlot(df, figsize, cols, nrow, ncol)
# %%
#TotalCharges 컬럼: object > float
#SeniorCitizen 컬럼: int64 > object
df = df.astype({'SeniorCitizen':'object'})
# df = df.astype({'TotalCharges':'float'}) #문제 발생
# %%
#TotalCharges 컬럼의 데이터 타입을 변경하려고 했지만 error 발생
#Error: could not convert string to float: '' > 제거 필요
df['TotalCharges'].value_counts().index
df[df['TotalCharges']==' '] = None #' ' > None
df['TotalCharges'].value_counts(dropna=False)
# %%
df['TotalCharges'].value_counts(dropna=False)
df.dropna(axis=0, inplace=True) #None drop
# %%
#TotalCharges 컬럼 데이터 타입 변경: object > float
df = df.astype({'TotalCharges':'float'})
df.info()
# %%
#범주형, 연속형 컬럼명
objDf = df.select_dtypes('object')
numDf = df.select_dtypes('number')
objCols = list(objDf)
numCols = list(numDf)
#%%
#범주형 변수 시각화 재시도
figsize = (16,30)
cols = objCols
nrow = 6
ncol = 3
df = df
histPlot(df, figsize, cols, nrow, ncol)

#%%
#연속형 변수 시각화 재시도
figsize = (15,6)
cols = numCols
nrow = 1
ncol = 3
df = df
distPlot(df, figsize, cols, nrow, ncol)

#%%
#범주형, 연속형 컬럼 구분
print(objCols)
print('='*70)
print(numCols)

#%%
df = df
figsize = (16, 30)
nrow = 6
ncol = 3
cols = objCols
histPlot(df, figsize, nrow, ncol, cols, True)

#%%
#연속형 컬럼과 Churn(target) 컬럼 간 상관관계 확인
df = df
figsize = (15,6)
nrow = 1
ncol = 3
cols = numCols
histPlot(df, figsize, nrow, ncol, cols, y=True)
# %%
#범주형 컬럼 중 id drop 후 레이블인코딩 진행

objDf = objDf.iloc[:,1:]
objDf = objDf.apply(LabelEncoder().fit_transform)
objDf.head()
# %%
#레이블인코딩 된 범주형 컬럼들과 연속형 컬럼 concat
df = pd.concat([objDf, numDf], axis=1)
print(f"df shape: {df.shape}")
df.head()
# %%
#범주형 > 레이블인코딩 완료
#상관계수 및 pairplot 확인
df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', cbar_kws={'shrink':.5})
# %%
sns.pairplot(data=df)
# %%
#####머신러닝
# X, y정의 및 데이터 분리
X = df.drop(['Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
#%%
#알고리즘
dt = DecisionTreeClassifier(random_state=0)
lr = LogisticRegression(random_state=0)
vt = VotingClassifier([('decision', dt), ('logistic', lr)], voting='soft')
rf = RandomForestClassifier(random_state=0)
xgb = xgboost.XGBClassifier()

models = [dt, lr, vt, rf, xgb]

#%%
train_size=np.linspace(.1, 1.0, 5)
cv = 3
getLC(dt, X_train, X_test, y_train, y_test, train_size, cv)
# %%
train_size=np.linspace(.1, 1.0, 5)
cv = 3
getLC(lr, X_train, X_test, y_train, y_test, train_size, cv)
# %%
train_size=np.linspace(.1, 1.0, 5)
cv = 3
getLC(vt, X_train, X_test, y_train, y_test, train_size, cv)
# %%
train_size=np.linspace(.1, 1.0, 5)
cv = 3
getLC(rf, X_train, X_test, y_train, y_test, train_size, cv)
# %%
train_size=np.linspace(.1, 1.0, 5)
cv = 3
getLC(xgb, X_train, X_test, y_train, y_test, train_size, cv)
# %%
