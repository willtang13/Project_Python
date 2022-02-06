#Ch04 Q3

def add_index(li):
    return [x+i for i,x in enumerate(li)]

if __name__ == '__main__':
    li = [8, 4, 1, 7]
    print(add_index(li))