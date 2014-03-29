from __future__ import absolute_import

import inspect


def in_stack(function):
    """Check whether the given function is in the call stack"""
    codes = [record[0].f_code for record in inspect.stack()]
    return function.__code__ in codes
