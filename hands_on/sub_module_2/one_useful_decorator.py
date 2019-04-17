import functools
    
def deco(func):
    @functools.wraps(func)
    def wrapper_deco(*args, **kwargs):
        print('this is wrapper first line')
        func(*args, **kwargs)
        print('this is wrapper second line')
    return wrapper_deco