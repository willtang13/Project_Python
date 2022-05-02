#Ch04 Q2
from math import *

def split_n(li, n):
    out = []
    a = int(len(li) / n)
    r = len(li) % n
    if r != 0:
        a += 1
    
    for i in range(a):
        out.append(li[n*i:n*i+n])
    return out

li = list(range(10))

print('Method 1: using for loop')
out = split_n(li, 3)
print(out)

print('Method 2: using list comprehension')
n = 3
out2 = [li[n*i:n*i+n] for i in range(ceil(len(li)/n))]
print(out2)