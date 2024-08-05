call env\Scripts\activate
cd website
daphne -p 8001 website.asgi:application