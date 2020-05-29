from django import template

register = template.Library()

@register.filter
def keyvalue(dic, key):
    try:
        return dic[key]
    except KeyError:
        return ''

@register.filter
def notfirstvisit(session):
    """
    if not first visit - then session should have key from tmp_ot_login_required
    :param session: `request.session`
    :return: `bool` - whether first visit or not
    """
    return session.get('not_first_visit', False)


@register.filter
def isfirstvisit(session):
    """
    if first visit - then session should not have key from tmp_ot_login_required
    :param session: `request.session`
    :return: `bool` - whether first visit or not
    """
    return not session.get('not_first_visit', False)