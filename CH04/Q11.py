# Ch04 Q11

def array_2d(row, col, ini_value=None):
    result = [[ini_value for j in range(col)] for i in range(row) ]
    return result

print(array_2d(2,3,None))