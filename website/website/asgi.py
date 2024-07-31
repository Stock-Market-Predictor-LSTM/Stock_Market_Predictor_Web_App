# asgi.py
import os
import django
from django.core.asgi import get_asgi_application
django_asgi_app = get_asgi_application()
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import dashboard.routing  # Replace with your app's routing module

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')
django.setup()

print("ASGI configuration loaded.")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            dashboard.routing.websocket_urlpatterns  # Replace with your app's WebSocket URL patterns
        )
    ),
})
