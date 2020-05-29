from __future__ import absolute_import, unicode_literals
import os, sys
from celery import Celery

from django.conf import settings
# docu from http://docs.celeryproject.org/en/latest/django/first-steps-with-django.html
# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

# config doesnt take effect
# CELERY_ACCEPT_CONTENT = ['json']
# CELERY_TASK_SERIALIZER = 'json'
# CELERY_RESULT_SERIALIZER = 'json'
# ACCEPT_CONTENT = ['json']
# TASK_SERIALIZER = 'json'
# RESULT_SERIALIZER = 'json'

app = Celery('breath',
             broker='amqp://breath:development@localhost:5672/breath_host', # using rabbitmq as broker and backend
             backend='amqp://breath:development@localhost:5672/breath_host', # using rabbitmq as broker and backend
             result='djcelery.backends.database:DatabaseBackend',
             include='breath.tasks',
             task_serializer='json',
             )

app.conf.update(
    CELERY_TASK_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],  # Ignore other content
    CELERY_RESULT_SERIALIZER='json',
    CELERY_TIMEZONE='Europe/Berlin',
    CELERY_ENABLE_UTC=True,)

#
# # Optional configuration, see the application user guide.
# app.conf.update(
#     result_expires=3600,
# )

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
# app.config_from_object('django.conf:settings', namespace='CELERY')
# for celery < 4 we cannot give namespace kw
# app.config_from_object('django.conf:settings')#, namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

if __name__ == '__main__':
    app.start()