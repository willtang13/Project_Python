# Ch04 Q12

def polynomial(pars, x):
    y = 0
    fun = []
    for i, a in enumerate(pars):
        if i == 0:
            fun.append(str(a))
        else:
            fun.append(str(a)+'*x^'+str(i) )
        y += a*(x**i)
    fun_str = '+'.join(fun)
    print('f(x)='+fun_str)
    print(f'x = {x}')
    print(f'y = {y}')

pars = [1, 2, 3, 4]
polynomial(pars, 2)
