from datetime import datetime

from django_celery_results.models import TaskResult
from celery.result import AsyncResult
from datetime import datetime
from celery.signals import after_task_publish, task_postrun
import redis

redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)


@after_task_publish.connect
def record_insertion_time(sender=None, headers=None, body=None, **kwargs):
    task_id = headers['id']
    current_time = datetime.utcnow()
    insertion_time = current_time.timestamp() 
    redis_client.zadd("celery.insertion_times", {task_id: insertion_time})


@task_postrun.connect
def remove_task_after_completion(sender=None, task_id=None, **kwargs):
    # Remove the completed task from the Redis sorted set
    redis_client.zrem("celery.insertion_times", task_id)

def get_task_position(task_id):
    position = redis_client.zrank("celery.insertion_times", task_id)
    if position is not None:
        return position + 1  # Redis ranks are zero-based
    return None



def request_to_dict(request):
    # Initialize an empty dictionary
    data = {}

    # Get the GET parameters
    for key in request.GET.keys():
        data[key] = request.GET.get(key)

    # Get the POST parameters
    for key in request.POST.keys():
        data[key] = request.POST.get(key)

    data['method'] = request.method
    data['path'] = request.path
    data['user_agent'] = request.META.get('HTTP_USER_AGENT')
    data['ip_address'] = request.META.get('REMOTE_ADDR')
    return data

def validate_data(data):
    if not data.get('ticker'):
        return (False, 'Please choose a ticker.')
    if not data.get('start_date'):
        return (False, 'Please choose a start date.')
    if not data.get('end_date'):
        return (False, 'Please choose a end date.')
    
    
    try:
        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d').date()
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d').date()
        today = datetime.today().date()
        # Check that start date is before end date
        if start_date >= end_date:
            return (False, 'Please make sure the start date is before the end date.')
        # Check that end date is at least today's date
        if end_date > today:
            return (False, 'Please make sure the end date is today or earlier.')
    except ValueError:
        return (False, 'Something werid has happened, please email me with your inputs :)')
    
    return (True,None)