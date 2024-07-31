from datetime import datetime

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