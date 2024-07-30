from celery import shared_task
from django.utils import timezone
from django_celery_results.models import TaskResult

@shared_task
def cleanup_expired_task_results():
    expiration_time = timezone.now() - timezone.timedelta(seconds=10)
    TaskResult.objects.filter(date_done__lt=expiration_time).delete()