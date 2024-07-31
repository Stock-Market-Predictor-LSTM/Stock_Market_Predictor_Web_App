from django.urls import path, include
from dashboard import views

urlpatterns = [
    path('',views.dashboard, name = 'dashboard'),
    path('load_data/',views.load_data, name = 'load_data'),
    path('abort/',views.abort, name = 'abort'),
]