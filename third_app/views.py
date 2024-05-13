from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views import View
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from django.conf import settings

# Create your views here.
def hola_mundo(request):
    return HttpResponse('Hola Api3!')

# Construcción de una función que realice el particionado completo
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

df = settings.DF_GLOBAL

# Dividimos el conjunto de datos
train_set, val_set, test_set = train_val_test_split(df)
X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')
X_test, y_test = remove_labels(test_set, 'calss')
# Modelo entrenado con el conjunto de datos sin escalar
clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train, y_train)
# Predecimos con el conjunto de datos de validación
y_pred = clf_rnd.predict(X_val)

# Uso de Grid Search para selección del modelo
param_grid = [
    # try 9 (3×3) combinations of hyperparameters
    {'n_estimators': [2, 2, 2], 'max_leaf_nodes': [2, 2, 2]},
    #{'n_estimators': [100, 500, 1000], 'max_leaf_nodes': [16, 24, 36]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [2, 2], 'max_features': [2, 2, 2]},
    #{'bootstrap': [False], 'n_estimators': [100, 500], 'max_features': [2, 3, 4]},
  ]
rnd_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
# train across 5 folds, that's a total of (9+6)*5=75 rounds of training 
grid_search = GridSearchCV(rnd_clf, param_grid, cv=2,scoring='f1_weighted', return_train_score=True)
grid_search.fit(X_train, y_train)

grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_

param_distribs = {
        'n_estimators': randint(low=1, high=2),
        'max_depth': randint(low=1, high=2),
    }
rnd_clf = RandomForestClassifier(n_jobs=-1)
# train across 2 folds, that's a total of 5*2=10 rounds of training
rnd_search = RandomizedSearchCV(rnd_clf, param_distributions=param_distribs,
                                n_iter=1, cv=2, scoring='f1_weighted')
rnd_search.fit(X_train, y_train)

rnd_search.best_params_
rnd_search.best_estimator_

cvres = rnd_search.cv_results_

# Seleccionamos el mejor modelo
clf_rnd = rnd_search.best_estimator_
# Predecimos con el conjunto de datos de entrenamiento
y_train_pred = clf_rnd.predict(X_train)

# Predecimos con el conjunto de datos de entrenamiento
y_val_pred = clf_rnd.predict(X_val)

class GetLength(View):
    def get(self, request):
        result = len(df)
        return JsonResponse({'length': result})
class GetNCaracteristics(View):
    def get(self, request):
        result = len(df.columns)
        return JsonResponse({'n_caracteristics': result})
class GetF1Score(View):
    def get(self, request):
        result = f1_score(y_pred, y_val, average='weighted')
        return JsonResponse({'F1_score': result})
class GetBestParams(View):
    def get(self, request):
        return JsonResponse({'best_params': grid_search.best_params_})
class GetParams(View):
    def get(self, request):
        result=[]
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            p="F1 score:", mean_score, "-", "Parámetros:", params
            result.append(p)
        return JsonResponse(result,safe=False)
class GetBestParamsRSCV(View):
    def get(self, request):
        return JsonResponse({'best_params': rnd_search.best_params_})
class GetParamsRSCV(View):
    def get(self, request):
        result=[]
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            p="F1 score:", mean_score, "-", "Parámetros:", params
            result.append(p)
        return JsonResponse(result,safe=False)
class GetBestEstimator(View):
    def get(self, request):
        result=rnd_search.best_estimator_.get_params()
        return JsonResponse(result,safe=False)
class GetTestF1Score(View):
    def get(self,request):
        result="F1 score Train Set:", f1_score(y_train_pred, y_train, average='weighted')
        return JsonResponse(result,safe=False)
class GetValidationF1Score(View):
    def get(self,request):
        result="F1 score Validation Set:", f1_score(y_val_pred, y_val, average='weighted')
        return JsonResponse(result,safe=False)