# example/urls.py
from django.urls import path

from first_app import views


urlpatterns = [
    path('', views.hola_mundo),
    path('head/', views.GetHead.as_view(), name='head'),
    path('describe/', views.GetHead.as_view(), name='describe'),
    path('info/', views.GetInfo.as_view(), name='info'),
    path('test_f1_score/',views.GetTestF1Score.as_view(),name='test_f1_score'),
    path('clf/',views.GetClf.as_view(),name='clf'),
    path('feature/',views.GetFeature.as_view(),name='feature'),
    path('feature_sorted/',views.GetFeatureSorted.as_view(),name='feature_sorted'),
    path('x_train_reduced/',views.GetXTrainReduced.as_view(),name='x_train_reduced'),
    path('validation_f1_score/',views.GetValidationF1Score.as_view(),name='validation_f1_score'),
]