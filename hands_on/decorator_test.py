from sub_module_1.one_useful_module import hello
from sub_module_2.one_useful_decorator import deco


# test the class first 
a = hello()
a.a_very_useful_method(3)

# test the decorator
@deco
def deco_test():
    print('THIS IS THE CORE FUNCTION')

deco_test()

# overall test
@deco
a.a_very_useful_method

s(3)