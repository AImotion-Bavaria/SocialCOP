def a_new_decorator(a_func):

    def wrapTheFunction(x : int):
        print("I am doing some boring work before executing a_func()")

        a_func(x)

        print("I am doing some boring work after executing a_func()")

    return wrapTheFunction

def a_function_requiring_decoration(x:int):
    print(f"I am the function which needs some decoration to remove my foul smell {x}")

a_function_requiring_decoration(6)
#outputs: "I am the function which needs some decoration to remove my foul smell"

a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
#now a_function_requiring_decoration is wrapped by wrapTheFunction()

a_function_requiring_decoration(7)
#outputs:I am doing some boring work before executing a_func()
#        I am the function which needs some decoration to remove my foul smell
#        I am doing some boring work after executing a_func()