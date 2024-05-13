from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views import View
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from django.conf import settings
# Create your views here.
def hola_mundo(request):
    return HttpResponse('Hola Api18!')

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
def plot_dbscan(dbscan, X, size):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker=".", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

df = settings.DF_GLOBAL_2


X = df[["V10", "V14"]].copy()
y = df["Class"].copy()
dbscan1 = DBSCAN(eps=0.05, min_samples=5)
dbscan1.fit(X)
df.drop(["Time", "Amount"], axis=1)
plt.figure(figsize=(12, 6))
plot_dbscan(dbscan1, X.values, size=100)
plt.xlabel("V10", fontsize=14)
plt.ylabel("V14", fontsize=14)
response = HttpResponse(content_type='image/png')
plt.savefig(response, format='png')
class GetDBSCAN(View):
    def get(self, request):
        
        return response
    
counter = Counter(dbscan1.labels_.tolist())
bad_counter = Counter(dbscan1.labels_[y == 1].tolist())

#for key in sorted(counter.keys()):
#    print("Label {0} has {1} samples - {2} are malicious samples".format(
#        key, counter[key], bad_counter[key]))
X = df.drop("Class", axis=1)
y = df["Class"].copy()
clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X, y)
feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X_reduced)

counter = Counter(dbscan.labels_.tolist())
bad_counter = Counter(dbscan.labels_[y == 1].tolist())

#for key in sorted(counter.keys()):
#    print("Label {0} has {1} samples - {2} are malicious samples".format(
#        key, counter[key], bad_counter[key]))

clusters = dbscan.labels_

class GetPurity(View):
    def get(self,request):
        return JsonResponse({"Purity Score": purity_score(y, clusters)})
class GetSilhouette(View):
    def get(self,request):
        return JsonResponse({"Silhouette Score": metrics.silhouette_score(X_reduced, clusters, sample_size=10000)})
class GetCalinski(View):
    def get(self,request):
        return JsonResponse({"Calinski Harabasz": metrics.calinski_harabasz_score(X_reduced, clusters)})

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
class GetMoons(View):
    def get(self,request):
        X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
        plt.figure(figsize=(12, 6))
        plt.scatter(X[:,0][y == 0], X[:,1][y == 0], c="g", marker=".")
        plt.scatter(X[:,0][y == 1], X[:,1][y == 1], c="r", marker=".")
        response=HttpResponse(content_type='image/png')
        plt.savefig(response, format='png')
        return response


class GetDBSCAN2(View):
    def get(self,request):
        dbscan2 = DBSCAN(eps=0.05, min_samples=6)
        dbscan2.fit(X)
        plt.figure(figsize=(12, 6))
        plot_dbscan(dbscan2, X, size=100)
        response=HttpResponse(content_type='image/png')
        plt.savefig(response, format='png')
        return response

#counter = Counter(dbscan2.labels_.tolist())
#bad_counter = Counter(dbscan2.labels_[y == 1].tolist())

#for key in sorted(counter.keys()):
#    print("Label {0} has {1} samples - {2} are malicious samples".format(
#        key, counter[key], bad_counter[key]))