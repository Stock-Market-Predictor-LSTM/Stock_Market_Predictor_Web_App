from django.shortcuts import render,redirect
from django.urls import reverse
from datetime import datetime
from dashboard.utilities_helpers.utilities import request_to_dict,validate_data,redis_client
from dashboard.utilities_helpers.fetch_stock_info import get_close_price
from celery.result import AsyncResult
from celery import Celery
from website.celery import app

# Create your views here.

def dashboard(request):
    form_data = request.session.get('form_data', None)
    data_valid = request.session.get('data_valid', None)
    error_msg = request.session.get('error_msg', None)
    price_data = request.session.get('price_data', None)

    if request.session.get('task_id', None):
        task = get_close_price.AsyncResult(request.session.get('task_id', None))
        task.revoke(terminate=True)
        task.abort()
        redis_client.zrem("celery.insertion_times", request.session['task_id'])
        request.session['task_id'] = None
        

    task_id = None

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
    if task_id is not None:
        del request.session['task_id']

    return render(request, "dashboard/basic.html", {'form_data': form_data, 'data_valid': data_valid,'error_msg': error_msg, 'price_data': price_data, 'task_id':task_id})


def load_data(request):
    if request.session.get('task_id', None):
        task = get_close_price.AsyncResult(request.session.get('task_id', None))
        task.revoke(terminate=True)
        task.abort()
        redis_client.zrem("celery.insertion_times", request.session['task_id'])
        request.session['task_id'] = None
    data = request_to_dict(request)
    print(data)
    data_valid, error_msg = validate_data(data)
    task_id = None
    if data_valid:
        # Start the Celery task and get the task ID
        task = get_close_price.delay(data)
        task_id = task.id
        request.session['task_id'] = task.id
        

    request.session['form_data'] = data
    request.session['data_valid'] = data_valid
    request.session['error_msg'] = error_msg
    
    return render(request, "dashboard/basic.html", {'form_data': data, 'data_valid': data_valid,'error_msg': error_msg, 'price_data': None, 'task_id':task_id})
    #return redirect(reverse('dashboard'))

def abort(request):
    return redirect(reverse('dashboard'))
