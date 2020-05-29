# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.utils.translation import gettext as _
from django.views.generic import (
    TemplateView,
)
from cookie_consent.util import (get_cookie_value_from_request,
                                 accept_cookies, decline_cookies)
from cookie_consent.views import CookieGroupBaseProcessView

class TestPageView(TemplateView):
    template_name = "cookies/cookie_policy.html"

    def get(self, request, *args, **kwargs):
        response = super(TestPageView, self).get(request, *args, **kwargs)
        # if get_cookie_value_from_request(request, "optional") is True:
        #     val = "optional cookie set from django"
        #     response.set_cookie("optional_test_cookie", val)
        return response

# classes on top of cookie_consent - that modify session for initial accept / decline to show hide cookie bar

class CookieGroupAcceptView2(CookieGroupBaseProcessView):
    """
    View to accept CookieGroup.
    """

    def process(self, request, response, varname):
        # log into session
        request.session['not_first_visit'] = True
        accept_cookies(request, response, varname)


class CookieGroupDeclineView2(CookieGroupBaseProcessView):
    """
    View to decline CookieGroup.
    """

    def process(self, request, response, varname):
        request.session['not_first_visit'] = True
        decline_cookies(request, response, varname)

    def delete(self, request, *args, **kwargs):
        request.session['not_first_visit'] = False
        return self.post(request, *args, **kwargs)
