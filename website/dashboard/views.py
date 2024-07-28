from django.shortcuts import render,redirect
from django.urls import reverse
from datetime import datetime
from dashboard.utilies_helpers.utilies import request_to_dict,validate_data
from dashboard.utilies_helpers.fetch_stock_info import get_close_price

# Create your views here.

def dashboard(request):
    form_data = request.session.get('form_data', None)
    data_valid = request.session.get('data_valid', None)
    error_msg = request.session.get('error_msg', None)
    price_data = request.session.get('price_data', None)

    if price_data == True:
        price_data = False

    if form_data:
        del request.session['form_data']
    if data_valid is not None:
        del request.session['data_valid']
    if error_msg is not None:
        del request.session['error_msg']
    if price_data is not None:
        del request.session['price_data']

    return render(request, "dashboard/basic.html", {'form_data': form_data, 'data_valid': data_valid,'error_msg': error_msg, 'price_data': price_data})


def load_data(request):
    data = request_to_dict(request)
    
    data_valid, error_msg = validate_data(data)
    if data_valid:
        price_dict = get_close_price(data)
        request.session['price_data'] = price_dict 

    request.session['form_data'] = data
    request.session['data_valid'] = data_valid
    request.session['error_msg'] = error_msg
    
    return redirect(reverse('dashboard'))
