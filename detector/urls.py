from django.urls import path
from . import views
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.upload_page, name='upload_page'),
    path('process/', views.process_uploads, name='process_uploads'),
    path('results/', views.results_page, name='results_page'),
    path('clear/', views.clear_results, name='clear_results'),
    path('mark_correct/<int:log_id>/', views.mark_correct, name='mark_correct'),
    path('mark_incorrect/<int:log_id>/', views.mark_incorrect, name='mark_incorrect'),
    path('camera/', views.camera_page, name='camera_page'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('report/', views.download_report, name='download_report'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('mark/<int:log_id>/<str:status>/', views.mark_result, name='mark_result'),
    path('about/', views.about_page, name='about_page'), # 🌟 المسار الجديد 🌟
    






]
