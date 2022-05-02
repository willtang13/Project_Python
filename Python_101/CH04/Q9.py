# Ch04 Q9

def rangeStr(a, b=None, step=None):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    if b is None and step is None:
        s_i = 0
        e_i = alphabet.index(a)
    elif step is None:
        s_i = alphabet.index(a)
        e_i = alphabet.index(b)
        step = 1
    else:
        s_i = alphabet.index(a)
        e_i = alphabet.index(b)

    return [x for x in alphabet[s_i:e_i:step]]

print(rangeStr('f'))
print(rangeStr('i','l'))
print(rangeStr('a','f',2))