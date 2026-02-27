from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'), # The home route is now the login page
    path('logout/', views.logout_view, name='logout'),
    
    path('upload/', views.upload_csv_view, name='upload_csv'),
    path('portfolio/', views.portfolio_view, name='portfolio'), 
    path('dashboard/<str:business_name>/', views.dashboard_view, name='dashboard'),
]