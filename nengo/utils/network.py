from nengo.utils.decorators import decorator


@decorator
def with_self(wrapped, instance, args, kwargs):
    with instance:
        return wrapped(*args, **kwargs)
