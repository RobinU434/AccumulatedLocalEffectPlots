#Importing all essential liabraries
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px

#Importing Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier 


from sklearn.metrics import accuracy_score
import time


def train(model, X_train, y_train, X_test, y_test):
    Name = []
    Accuracy = []
    Time_Taken = []
    Model = []

    Name.append(type(model).__name__)
    begin = time.time()
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    end = time.time()
    accuracyScore = accuracy_score(prediction,y_test)
    Accuracy.append(accuracyScore)
    Time_Taken.append(end-begin)

    Dict = {'Name':Name,'Accuracy':Accuracy,'Time Taken':Time_Taken}
    model_df = pd.DataFrame(Dict)
    print(model_df)
    
    return model


# available models are:
# models = [
#     KNeighborsClassifier(),
#     SGDClassifier(),
#     LogisticRegression(),
#     RandomForestClassifier(),
#     GradientBoostingClassifier(),
#     AdaBoostClassifier(),
#     BaggingClassifier(),
#     SVC(),
#     GaussianNB(),
#     DummyClassifier(),
#     ExtraTreeClassifier()
#     ]