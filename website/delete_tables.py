# delete_tables.py
import os
import django
from django.utils import timezone

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')
django.setup()

from django_celery_results.models import TaskResult


expiry_time = timezone.now() - timezone.timedelta(hours=0)
TaskResult.objects.filter(date_done__lt=expiry_time).delete()