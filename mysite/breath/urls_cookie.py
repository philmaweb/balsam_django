# -*- coding: utf-8 -*-
from django.urls import path
from django.conf.urls import include, url
from django.conf.urls.static import static
from django.conf import settings

# from djcelery.views import task_status
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

from . import views, views_gcms, views_cookie
from cookie_consent.views import CookieGroupListView

# path('cookies/', include('cookie_consent.urls')),

urlpatterns = [
    # ex: /breath/
    # path('', views.index, name='index'),

    # cookie consent
    path('cookie_consent', views_cookie.TestPageView.as_view(), name='cookie_consent'),

    url(r'^accept/$',
        csrf_exempt(views_cookie.CookieGroupAcceptView2.as_view()),
        name='cookie_consent_accept_all'),
    url(r'^accept/(?P<varname>.*)/$',
        csrf_exempt(views_cookie.CookieGroupAcceptView2.as_view()),
        name='cookie_consent_accept'),
    url(r'^decline/(?P<varname>.*)/$',
        csrf_exempt(views_cookie.CookieGroupDeclineView2.as_view()),
        name='cookie_consent_decline'),
    url(r'^decline/$',
        csrf_exempt(views_cookie.CookieGroupDeclineView2.as_view()),
        name='cookie_consent_decline_all'),
    path('cookies/',
        CookieGroupListView.as_view(),
        name='cookie_consent_cookie_group_list'),
]