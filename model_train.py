# !/usr/bin/python
#-*-coding:utf-8-*-


import numpy as np
import pandas as pd
import urllib
import pickle
import sklearn
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV  
from sklearn.svm import SVC  
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def getTokens(input):
	tokensBySlash = str(input.encode('utf-8')).split('/')	#get tokens after splitting by slash
	allTokens = []
	for i in tokensBySlash:
		tokens = str(i).split('-')	#get tokens after splitting by dash
		tokensByDot = []
		for j in range(0,len(tokens)):
			tempTokens = str(tokens[j]).split('.')	#get tokens after splitting by dot
			tokensByDot = tokensByDot + tempTokens
		allTokens = allTokens + tokens + tokensByDot
	allTokens = list(set(allTokens))	#remove redundant tokens
	if 'com' in allTokens:
		allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
	return allTokens


def data_prepared():
    ‘’‘
    Feature Extraction
    ’‘’
    allurls = './data1.csv'	#path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe

    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling

    y = [d[1] for d in allurlsdata]	#all labels 
    corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)	#split into training and testing set 80/20 ratio
    return X_train, y_train, X_test, y_test



def SVM_model_test(X_train, y_train, X_test, y_test, model_file='model/svm.pkl'):
    '''
    SVM
    '''
    model = SVC()
    model.fit(X_train, y_train)
    print 'SVM Model：'
    print(model)
    with open(model_file, 'w') as f:
        joblib.dump(model, model_file)
    svm_model_file=file(model_file)
    svm_model=joblib.load(svm_model_file)
    expected = y_test
    predicted = svm_model.predict(X_test)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print (np.linalg.matrix_rank)





def KNN_model_test(X_train, y_train, X_test, y_test, model_file='model/knn.pkl'):
    '''
    KNN
    '''  
    model = KNeighborsClassifier()  
    model.fit(X_train, y_train) 
    print model 
    with open(model_file, 'w') as f:
        joblib.dump(model, model_file)
    knn_model_file=file(model_file)
    knn_model=joblib.load(knn_model_file)
    expected = y_test
    predicted = knn_model.predict(X_test)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted)) 

      




def LR_model_test(X_train, y_train, X_test, y_test, model_file='model/lr.pkl'):
    '''
    LR
    '''
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model)
    with open(model_file, 'w') as f:
        joblib.dump(model, model_file)
    lr_model_file=file(model_file)
    lr_model=joblib.load(lr_model_file)
    expected = y_test
    predicted = lr_model.predict(X_test)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print model.get_params([3])
    print model.coef_
    print model.intercept_


if __name__=='__main__':
    X_train, y_train, X_test, y_test=data_prepared()
    SVM_model_test(X_train, y_train, X_test, y_test, model_file='model/svm.pkl')
    # NB_model_test(X_train, y_train, X_test, y_test, model_file='model/knn.pkl')
    KNN_model_test(X_train, y_train, X_test, y_test)
    LR_model_test(X_train, y_train, X_test, y_test, model_file='model/lr.pkl')
