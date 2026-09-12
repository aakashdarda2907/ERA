from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('report/', include('audit.urls')),
    path('', include('chat.urls')),
]