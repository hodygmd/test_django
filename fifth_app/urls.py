from django.urls import path

from fifth_app import views


urlpatterns = [
    path('', views.hola_mundo),
    path('dbscan/', views.GetDBSCAN.as_view(), name='dbscan'),
    path('purity/', views.GetPurity.as_view(), name='purity'),
    path('shiloutte/', views.GetSilhouette.as_view(), name='shiloutte'),
    path('calinski/', views.GetCalinski.as_view(), name='calinski'),
    path('moons/', views.GetMoons.as_view(), name='moons'),
    path('dbscan2/', views.GetDBSCAN2.as_view(), name='dbscan2'),
    #path('describe/', views.GetDescribe.as_view(), name='describe'),
    #path('info/', views.GetInfo.as_view(), name='info'),
]