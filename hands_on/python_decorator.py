'''
this file is zabbour barcha jarrbou to7sol
'''
import functools

def my_deco(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        func(*args, **kwargs)
        print("Something is happening after the function is called.")
    return wrapper

def my_decorator(func):
    '''
    el zebda mta3 el zebda mawjoud lihna
    '''
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        ret = func(*args, **kwargs)
        print("use the : {} to create a bigger one".format(ret))
    return wrapper

def say_whee(name):
    print("Whee {} !".format(name))
    return 'boom'

say_whee = my_decorator(say_whee)

say_whee('hamdoooun')

from datetime import datetime

def not_during_the_night(func):
    def wrapper(*args, **kwargs):
        if 7 <= datetime.now().hour < 22:
            func()
        else:
            pass  # Hush, the neighbors are asleep
    return wrapper

def say_whee():
    print("Whee!")

say_whee = not_during_the_night(say_whee)

say_whee()

@my_decorator
def say_whee_decorated(name):
    print("Whee {} !".format(name))
    return 'boom'

say_whee_decorated('hamadou')


import re
regex = re.compile('[^a-zA-Z]')
def string_cleaner(func):
    @functools.wraps(func)
    def wrapper_string(*args, **kwargs):
        string = func(*args, **kwargs)
        print("this is the input sentence that you intered: \n {}".format(string))
        new = regex.sub('', string)
        print("this is the input sentence after surgery: \n {}".format(new))
    return wrapper_string

@my_deco
@string_cleaner
def message_printer():
    sen = input("enter your sentence please")
    return sen


message_printer()
help(my_decorator)
print(message_printer.__name__)
print(message_printer.__name__)


import functools

def count_calls(func):
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print("Call {} of {}".format(wrapper_count_calls.num_calls,func.__name__))
        return func(*args, **kwargs)
    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls

@count_calls
def say_whee():
    print("Whee!")


for i in range(5):
    say_whee()