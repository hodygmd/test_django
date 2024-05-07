# example/urls.py
from django.urls import path

from third_app import views


urlpatterns = [
    path('', views.hola_mundo),
    path('length/', views.GetLength.as_view(), name='length'),
    path('ncaracteristics/', views.GetNCaracteristics.as_view(), name='ncaracteristics'),
    path('f1_score/', views.GetF1Score.as_view(), name='f1_score'),
    path('best_params/', views.GetBestParams.as_view(), name='best_params'),
    path('params/', views.GetParams.as_view(), name='params'),
    path('best_params_rscv/', views.GetBestParamsRSCV.as_view(), name='best_params_rscv'),
    path('params_rscv/', views.GetParamsRSCV.as_view(), name='params_rscv'),
    path('best_estimator/', views.GetBestEstimator.as_view(), name='best_estimator'),
    path('test_f1_score/', views.GetTestF1Score.as_view(), name='test_f1_score'),
    path('validation_f1_score/', views.GetValidationF1Score.as_view(), name='validation_f1_score'),
]