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
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from django.conf import settings
# Create your views here.
def hola_mundo(request):
    return HttpResponse('Hola Api1!')

def plot_data(X, y):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'k.', markersize=2)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, y, resolution=1000, show_centroids=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X, y)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

df = settings.DF_GLOBAL_2

class GetHead(View):
    def get(self, request):
        # Obtener los primeros 10 registros del DataFrame
        result = df.head(10).to_dict(orient='records')
        
        # Devolver la respuesta en formato JSON
        return JsonResponse(result, safe=False)
    
class GetCC(View):
    def get(self, request):
        return JsonResponse({"Número de características": len(df.columns), "Longitud del conjunto de datos": len(df)})

class GetNull(View):
    def get(self,request):
        # Comprobamos si alguna columna tiene valores nulos
        result=df.isna().any().to_dict()
        return JsonResponse(result,safe=False)

class GetDescribe(View):
    def get(self,request):
        result=df.describe().to_dict(orient='records')
        return JsonResponse(result,safe=False)

features = df.drop("Class", axis=1)
plt.figure(figsize=(12,32))
gs = gridspec.GridSpec(8, 4)
gs.update(hspace=0.8)
for i, f in enumerate(features):
    ax = plt.subplot(gs[i])
    sns.distplot(df[f][df["Class"] == 1])
    sns.distplot(df[f][df["Class"] == 0])
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(f))
response = HttpResponse(content_type='image/png')
plt.savefig(response, format='png')

class GetCar(View):
    def get(self,request):
        '''# Representamos gráficamente las características
        features = df.drop("Class", axis=1)
        plt.figure(figsize=(12,32))
        gs = gridspec.GridSpec(8, 4)
        gs.update(hspace=0.8)
        for i, f in enumerate(features):
            ax = plt.subplot(gs[i])
            sns.distplot(df[f][df["Class"] == 1])
            sns.distplot(df[f][df["Class"] == 0])
            ax.set_xlabel('')
            ax.set_title('feature: ' + str(f))
        response = HttpResponse(content_type='image/png')
        plt.savefig(response, format='png')
        return response'''
        return response

class GetTCar(View):
    def get(self,request):
        # Representación gráfica de dos características
        plt.figure(figsize=(12, 6))
        plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".")
        plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".")
        plt.xlabel("V10", fontsize=14)
        plt.ylabel("V14", fontsize=14)
        #img_buffer = BytesIO()
        #plt.savefig(img_buffer, format='png')
        #img_buffer.seek(0)
        response = HttpResponse(content_type='image/png')
        plt.savefig(response, format='png')
        return response


class GetDataset(View):
    def get(self,request):
        df.drop(["Time", "Amount"], axis=1)
        X = df[["V10", "V14"]].copy()
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X)
        plt.figure(figsize=(12, 6))
        plot_decision_boundaries(kmeans, X.values, df["Class"].values)
        plt.xlabel("V10", fontsize=14)
        plt.ylabel("V14", fontsize=14)
        response = HttpResponse(content_type="image/png")
        plt.savefig(response, format="png")
        return response

'''counter = Counter(clustersR.tolist())
bad_counter = Counter(clustersR[df['Class'] == 1].tolist())'''

X = df.drop("Class", axis=1)
y = df["Class"].copy()

# Utilizamos Random Forest para realizar selección de característica

clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X, y)
# Seleccionamos las características más importantes
feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
# Reducimos el conjunto de datos a las 7 características más importantes
X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

# Evaluamos los clusters y el contenido que se han formado
counter = Counter(clusters.tolist())
bad_counter = Counter(clusters[y == 1].tolist())

class GetPurity(View):
    def get(self,request):
        return JsonResponse({"Purity Score": purity_score(y, clusters)})
# Calculamos el purity score, es importante darse cuenta de que recibe las etiquetas
#print("Purity Score:", purity_score(y, clusters))

class GetSiloutte(View):
    def get(self,request):
        return JsonResponse({"Shiloutte": metrics.silhouette_score(X_reduced, clusters, sample_size=10000)})
# Calculamos el coeficiente de Shiloutte, es importante darse cuenta de que no le pasamos las etiquetas
#print("Shiloutte: ", metrics.silhouette_score(X_reduced, clusters, sample_size=10000))

class GetCalinski(View):
    def get(self,request):
        return JsonResponse({"Calinski harabasz": metrics.calinski_harabasz_score(X_reduced, clusters)})
# Calculamos el Calinski harabasz score, es importante darse cuenta de que no le pasamos las etiquetas
#print("Calinski harabasz: ", metrics.calinski_harabasz_score(X_reduced, clusters))