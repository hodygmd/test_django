from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views import View
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
from django.conf import settings

# Create your views here.
def hola_mundo(request):
    return HttpResponse('Hola Api1!')

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


clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train, y_train)
# Predecimos con el conjunto de datos de validación
#y_pred = clf_rnd.predict(X_val)
y_pred1=clf_rnd.predict(X_val)

feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)

columns = list(feature_importances_sorted.head(10).index)

X_train_reduced = X_train[columns].copy()
X_val_reduced = X_val[columns].copy()

clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train_reduced, y_train)

# Predecimos con el conjunto de datos de validación
#y_pred = clf_rnd.predict(X_val_reduced)
y_pred2=clf_rnd.predict(X_val_reduced)
########print("F1 score:", f1_score(y_pred, y_val, average='weighted'))#########
class GetHead(View):
    def get(self, request):
        # Obtener los primeros 10 registros del DataFrame
        result = df.head(10).to_dict(orient='records')
        
        # Devolver la respuesta en formato JSON
        return JsonResponse(result, safe=False)
class GetDescribe(View):
    def get(self,request):
        result=df.describe().to_dict(orient='records')
        return JsonResponse(result,safe=False)
class GetInfo(View):
    def get(self,request):
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        buffer.close()
        # Dividir la información en líneas para una mejor legibilidad en JSON
        info_lines = info_str.split('\n')
        return JsonResponse(info_lines,safe=False)
class GetTestF1Score(View):
    def get(self,request):
        result="F1 score:", f1_score(y_pred1, y_val, average='weighted')
        return JsonResponse(result,safe=False)
class GetClf(View):
    def get(self,request):
        result=clf_rnd.feature_importances_.tolist()
        return JsonResponse(result,safe=False)
class GetFeature(View):
    def get(self,request):
        result=feature_importances_sorted.head(20).to_dict()
        return JsonResponse(result,safe=False)
class GetFeatureSorted(View):
    def get(self,request):
        result=columns
        return JsonResponse(result,safe=False)
class GetXTrainReduced(View):
    def get(self,request):
        result=X_train_reduced.head(10).to_dict(orient='records')
        return JsonResponse(result,safe=False)
class GetValidationF1Score(View):
    def get(self,request):
        result="F1 score:", f1_score(y_pred2, y_val, average='weighted')
        return JsonResponse(result,safe=False)