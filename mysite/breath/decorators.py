from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate
from django.shortcuts import reverse, redirect
# from django.http import HttpResponseRedirect

from .models import TempUserManager, User

def temp_or_login_required(function):
    def wrapper(request, *args, **kwargs):
        # wrap for cookie management - hide bar once pressed a button - and prevent template errors
        if request.session.get('not_first_visit') is None:
            request.session['not_first_visit'] = False  # if not set - is first visit

        decorated_view_func = login_required(request)
        if decorated_view_func.user.is_authenticated:
            # return decorated_view_func(request)#(request, *args, **kwargs)  # return redirect to signin
            request.session['user_id'] = request.user.pk
            return function(request, *args, user=request.user, **kwargs)

        user_id = request.session.get('user_id', '')
        if user_id:
            user = User.objects.get(pk=user_id)
            return function(request, *args, user=user, **kwargs)

        # actually never happens, as we always have a user associated
        # tmp_user_id = request.session.get('tmp_user_id', '')
        # if tmp_user_id:
        #     temp_user = TempUser.objects.get(pk=tmp_user_id)
        #     return function(request, *args, user=temp_user.user, **kwargs)

        # create tmpuser and redirect to requested page
        tmpUserManager = TempUserManager()
        # print(request, args, kwargs)
        # import pdb; pdb.set_trace()
        temp_user, tmp_username = tmpUserManager.create_temp_user()
        request.session['user_id'] = temp_user.user.pk
        request.session['tmp_user_id'] = temp_user.pk
        return function(request, *args, user=temp_user.user, **kwargs)
        # return redirect(reverse('index', args=(), kwargs={}))


    wrapper.__doc__ = function.__doc__
    wrapper.__name__ = function.__name__
    return wrapper