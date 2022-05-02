#Ch04 Q5

def k(x):
    return x[1]


data = [('John', 40, 174, 65), ('Amy', 28, 165, 44), ('Jessie', 32, 158, 45)]  # Name, Age, Height, Weight

data_sort_by_age = sorted(data, key=k)
print(data)
print('After sorting by age')
print(data_sort_by_age)