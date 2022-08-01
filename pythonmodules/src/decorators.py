import warnings
import functools


def deprecated(func):
    """
    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.
    """

    @functools.wraps(func)
    def inner_function(*args, **kwargs):
        warnings.simplefilter('always', PendingDeprecationWarning)  # turn off filter
        warnings.warn("Call to pending deprecated function {}.".format(func.__name__),
                      category=PendingDeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', PendingDeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return inner_function

