from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about', views.about, name='about'),
    path('analyze/', views.analyze_news, name='analyze_news'),
    path('generate_query/', views.generate_query, name='generate_query'),
    path('search_web/', views.search_web, name='search_web')
]
