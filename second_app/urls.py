# example/urls.py
from django.urls import path

from second_app import views


urlpatterns = [
    path('', views.hola_mundo),
    path('value_counts/', views.GetValueCounts.as_view(), name='value_counts'),
    path('df_reduced/', views.GetDfReduced.as_view(), name='df_reduced'),
    path('dataset/', views.GetDataSet.as_view(), name='dataset'),
    path('variance_ratio/', views.GetVarianceRatio.as_view(), name='variance_ratio'),
    path('decision_limit/', views.GetDecisionLimit.as_view(), name='decision_limit'),
    path('ncomponents/', views.GetNComponents.as_view(), name='ncomponents'),
    path('variance_ratio2/', views.GetVarianceRatio2.as_view(), name='variance_ratio2'),
    path('dataframe/', views.GetDataFrame.as_view(), name='dataframe'),
    path('validation_f1_score/', views.GetValidationF1Score.as_view(), name='validation_f1_score'),
    path('test_f1_score/', views.GetTestF1Score.as_view(), name='test_f1_score'),
]