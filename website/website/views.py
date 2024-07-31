from django.http import HttpResponseRedirect

def redirect_to_external(request):
    return HttpResponseRedirect("https://www.ganels.com")