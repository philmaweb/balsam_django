"""
Django settings for mysite project.

Generated by 'django-admin startproject' using Django 2.2.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os
import sys

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PROJECT_DIR = "/".join(BASE_DIR.split("/")[:-1])

dir_to_add = BASE_DIR + "/breath/external/breathpy"
if dir_to_add not in sys.path:
    sys.path.append(dir_to_add)
    print("Adding {0} to pythonpath".format(dir_to_add))
if "PEAX_BINARY_PATH" not in os.environ:
    PEAX_BINARY_PATH = dir_to_add + "/bin/peax1.0-LinuxX64/peax"
    os.environ.setdefault("PEAX_BINARY_PATH", PEAX_BINARY_PATH)

# set peax binary path
PEAX_BINARY_PATH = os.environ['PEAX_BINARY_PATH']
try:
    fh = open(PEAX_BINARY_PATH, "r")
    fh.close()
except FileNotFoundError as fe:
    raise FileNotFoundError(fe.args,
        "Could not find PEAX binary. Please make sure to specify the binary.")


# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

FORCE_SCRIPT_NAME = "/balsam/"

# SECURITY WARNING: keep the secret key used in production secret! - will + needs overwrite on deployment
SECRET_KEY = 'verylongstringverylongstringverylongstringverylong'

# SECURITY WARNING: don't run with debug turned on in production!
# DEBUG = True
DEBUG = False

# let nginx/apache upstream handle the https
#SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
# SESSION_COOKIE_SECURE = True
# CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = "DENY"
# SECURE_SSL_REDIRECT = True
SECURE_CONTENT_TYPE_NOSNIFF = True


ALLOWED_HOSTS = ['0.0.0.0', '127.0.0.1', 'localhost', 'exbio.wzw.tum.de', 'www.exbio.wzw.tum.de']


# Application definition

INSTALLED_APPS = [
    'breath.apps.BreathConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 'django_extensions',
    'cookie_consent',
    'crispy_forms',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mysite.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['./templates',],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

WSGI_APPLICATION = 'mysite.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases
# this is changed during project build in docker container
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'breath',
        'USER': 'breathuser',
        'PASSWORD': 'development',
        'HOST': 'localhost',
        'PORT': '',
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'CET'

USE_I18N = True

USE_L10N = True

USE_TZ = True



# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(PROJECT_DIR, 'static')

# MEDIA and File config from https://simpleisbetterthancomplex.com/tutorial/2016/08/01/how-to-upload-files-with-django.html
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = 'media/'

INTERNAL_IPS = ['127.0.0.1']


# Redirect to home URL after login (Default redirects to /accounts/profile/)
LOGIN_URL = '/breath/accounts/login'
LOGIN_REDIRECT_URL = '/'
# TODO enable email backend?
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# chose crispy form template style
CRISPY_TEMPLATE_PACK = 'bootstrap4'
# make crispy forms fail loud
CRISPY_FAIL_SILENTLY = not DEBUG

if DEBUG:
    DEBUG_TOOLBAR_CONFIG = {
        'SHOW_TEMPLATE_CONTEXT': True,
    }


COOKIE_CONSENT_NAME = "cookie_consent"
COOKIE_CONSENT_MAX_AGE = 60 * 60 * 24 * 365 * 1  # 1 year
COOKIE_CONSENT_ENABLED = lambda r: DEBUG or (r.user.is_authenticated() and r.user.is_staff)