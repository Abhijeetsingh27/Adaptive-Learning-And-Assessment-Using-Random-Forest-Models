from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    
    # Student
    path('dashboard/', views.student_dashboard, name='student_dashboard'),
    path('topics/', views.topic_list, name='topic_list'),
    path('read/<int:material_id>/', views.read_material, name='read_material'),
    path('quiz/<int:topic_id>/', views.take_quiz, name='take_quiz'),
    path('profile/', views.student_profile, name='student_profile'),
    path('analytics/', views.student_analytics, name='student_analytics'),
    
    # Admin
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('manage/topics/', views.manage_topics, name='manage_topics'),
    path('manage/topics/add/', views.add_topic, name='add_topic'),
    path('manage/materials/add/', views.add_material, name='add_material'),
    path('manage/quiz/add/', views.add_question, name='add_question'),
    path('ml/retrain/', views.trigger_retrain, name='trigger_retrain'),
    
    # API for charts
    path('api/analytics-data/', views.analytics_data, name='analytics_data'),
]
