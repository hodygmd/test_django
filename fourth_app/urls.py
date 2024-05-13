from django.urls import path

from fourth_app import views


urlpatterns = [
    path('', views.hola_mundo),
    path('head/', views.GetHead.as_view(), name='head'),
    path('cc/', views.GetCC.as_view(), name='cc'),
    path('null/', views.GetNull.as_view(), name='null'),
    path('describe/', views.GetDescribe.as_view(), name='describe'),
    path('car/', views.GetCar.as_view(), name='car'),
    path('tcar/', views.GetTCar.as_view(), name='tcar'),
    path('dataset/', views.GetDataset.as_view(), name='dataset'),
    path('purity/', views.GetPurity.as_view(), name='purity'),
    path('shiloutte/', views.GetSiloutte.as_view(), name='shiloutte'),
    path('calinski/', views.GetCalinski.as_view(), name='calinski'),
    
]