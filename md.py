#%%
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


#%%
#연속형 변수 시각화 함수
def distPlot(df, figsize, cols, nrow, ncol):
    plt.figure(figsize=figsize)
    for idx, col in enumerate(cols):
        plt.subplot(nrow, ncol, idx+1)
        sns.distplot(df[col])
        plt.tight_layout()
    plt.show()


#범주형 변수 시각화 함수(y case 추가)
def histPlot(df, figsize, nrow, ncol, cols=None, y=False):
    if y == True:
        plt.figure(figsize=figsize)
        for idx, col in enumerate(cols):
            plt.subplot(nrow, ncol, idx+1)
            sns.histplot(data=df, x=col, y='Churn')
            plt.title(f"{col}")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        for idx, col in enumerate(cols):
            plt.subplot(nrow, ncol, idx+1)
            sns.histplot(data=df, x=col)
            plt.title(f"{col}")
        plt.tight_layout()
        plt.show()


def playML(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:,1]
    train_pred = [1 if p>0.5 else 0 for p in train_prob]

    test_prob = model.predict_proba(X_test)[:,1]
    test_pred = [1 if p>0.5 else 0 for p in test_prob]

    trainReport = classification_report(y_train, train_pred)
    testReport = classification_report(y_test, test_pred)

    trainRoc = roc_auc_score(y_train, train_prob)
    testRoc = roc_auc_score(y_test, test_prob)
    print(trainReport)
    print(testReport)
    print(f"trainRoc: {trainRoc}")
    print(f"testRoc: {testRoc}")
    return test_pred


def getRoc(model, X_train, X_test, y_train, y_test):
    test_pred = playML(model, X_train, X_test, y_train, y_test)
    fig = plt.figure(figsize=(10, 10))
    tpr, fpr, thr = roc_curve(y_test, test_pred)
    plt.plot(tpr, fpr)
    plt.title(f"ROC Curve {model.__class__.__name__}", fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)


def getLC(model, X_train, X_test, y_train, y_test, train_size, cv):
    getRoc(model, X_train, X_test, y_train, y_test)

    trainSizes, trainScore, testScore = learning_curve(model, X_train, y_train, train_sizes=train_size, cv=cv)
    fig = plt.figure(figsize=(10, 10))
    trainMean = np.mean(trainScore, axis=1)
    testMean = np.mean(testScore, axis=1)

    plt.plot(trainSizes, trainMean, "-o", label="train")
    plt.plot(trainSizes, testMean, "-o", label="cross val")
    plt.title(f"{model.__class__.__name__} Learning Curve", size=20)
    plt.xlabel("Train Sizes", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.legend()