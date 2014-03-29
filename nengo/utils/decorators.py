"""Functions for making better decorators.

This contains functions for avoiding some common problems with decorators:
  - __name__ and __doc__ are wrong
  - function argspec is wrong
  - function source code cannot be retrieved
  - cannot be applied on top of decorators implemented as descriptors

Most of these come from Graham Dumpleton's blog; see
  http://blog.dscpl.com.au/search/label/decorators
"""

import functools
import inspect


class object_proxy(object):

    def __init__(self, wrapped):
        self.wrapped = wrapped
        try:
            self.__name__ = wrapped.__name__
        except AttributeError:
            pass

    @property
    def __class__(self):
        return self.wrapped.__class__

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class bound_function_wrapper(object_proxy):

    def __init__(self, wrapped, instance, wrapper, binding, parent):
        super(bound_function_wrapper, self).__init__(wrapped)
        self.instance = instance
        self.wrapper = wrapper
        self.binding = binding
        self.parent = parent

    def __call__(self, *args, **kwargs):
        if self.binding == 'function':
            if self.instance is None:
                instance, args = args[0], args[1:]
                wrapped = functools.partial(self.wrapped, instance)
                return self.wrapper(wrapped, instance, args, kwargs)
            else:
                return self.wrapper(self.wrapped, self.instance, args, kwargs)
        else:
            instance = getattr(self.wrapped, '__self__', None)
            return self.wrapper(self.wrapped, instance, args, kwargs)

    def __get__(self, instance, owner):
        if self.instance is None and self.binding == 'function':
            descriptor = self.parent.wrapped.__get__(instance, owner)
            return bound_function_wrapper(descriptor, instance, self.wrapper,
                                          self.binding, self.parent)
        return self


class function_wrapper(object_proxy):

    def __init__(self, wrapped, wrapper):
        super(function_wrapper, self).__init__(wrapped)
        self.wrapper = wrapper
        if isinstance(wrapped, classmethod):
            self.binding = 'classmethod'
        elif isinstance(wrapped, staticmethod):
            self.binding = 'staticmethod'
        else:
            self.binding = 'function'

    def __get__(self, instance, owner):
        wrapped = self.wrapped.__get__(instance, owner)
        return bound_function_wrapper(wrapped, instance, self.wrapper,
                                      self.binding, self)

    def __call__(self, *args, **kwargs):
        return self.wrapper(self.wrapped, None, args, kwargs)


def decorator(wrapper):
    """decorates decorators.

    Example
    -------

    @decorator
    def my_function_wrapper(wrapped, instance, args, kwargs):
        return wrapped(*args, **kwargs)
    """
    def _wrapper(wrapped, instance, args, kwargs):
        def _execute(wrapped):
            if instance is None:
                return function_wrapper(wrapped, wrapper)
            elif inspect.isclass(instance):
                return function_wrapper(
                    wrapped, wrapper.__get__(None, instance))
            else:
                return function_wrapper(
                    wrapped, wrapper.__get__(instance, type(instance)))
        return _execute(*args, **kwargs)
    return function_wrapper(wrapper, _wrapper)
