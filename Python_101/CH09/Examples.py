# %% 9.1
def is_odd(n):
    if n == 0:
        return False
    else:
        return(is_even(n - 1))

def is_even(n):
    if n == 0:
        return True
    else:
        return (is_odd(n - 1)) 
    
print(is_odd(11))
print(is_even(23))

# %% 遞回實作
def my_sum(li):
    if li == []:
        return 0
    else:
        return li[0] + my_sum(li[1:])
    
def my_len(li):
    if li == []:
        return 0
    else:
        return 1 + my_len(li[1:])
    
def gcd_r(a, b):
    if a % b == 0:
        return b
    else:
        return (b, a%b)
    
def ctof(li):
    if li == []:
        return []
    else:
        return [li[0]*9.0/5.0+32] + ctof(li[1:])
    


x = [1, 2, 3, 4]    
print(my_sum(x), my_len(x))

# %%
