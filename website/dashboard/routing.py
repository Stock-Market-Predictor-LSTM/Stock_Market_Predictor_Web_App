# routing.py
from django.urls import re_path
from dashboard import consumers # Import your consumers

websocket_urlpatterns = [
    re_path(r'ws/queue_position/(?P<task_id>[\w-]+)/$', consumers.QueuePositionConsumer.as_asgi()),
]

