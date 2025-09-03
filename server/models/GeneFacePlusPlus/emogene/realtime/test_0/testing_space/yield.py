import random

def get_num():
    for i in range(5):
        yield i
        
gen = get_num()

print(next(gen))  # Get the first random number
print(next(gen))  # Get the second random number

print(gen.__next__())  # Get the third random number
print(list(gen)) # get the remaining random numbers