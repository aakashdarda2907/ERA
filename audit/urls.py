from django.urls import path
from . import views

app_name = 'audit'

urlpatterns = [
    path('', views.report, name='report'),
    path('risk-assessment/', views.risk_assessment, name='risk_assessment'),
]
