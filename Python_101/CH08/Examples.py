#%% named tuple
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y', 'z'])
pa = Point(1, 2, 3)
print(type(pa))
print(pa)
print(pa[0], pa[1], pa[2])
print(pa.x, pa.y, pa.z)

pb=Point(z=30, y=20, x=10)
for i in pb:
    print(i, end=' ')
# %% 
from collections.abc import Sequence
print(issubclass(Point, tuple))
print(issubclass(Point, Sequence))

# %%
Color = namedtuple('Color', 'red green blue')
c_red = Color(red=1.0, green=0, blue=0)
print(c_red)

Person = namedtuple('Rec', 'name, age, titles')
bob = Person('Bob', age=49, titles=['teacher', 'manager'])
print(bob)

Xxx = namedtuple('Xxx', ['def', '12ab', '_hi'], rename=True)
x = Xxx(1, 2, 3)
print(x)
# %% Ex.8.1 ch08_namedtuple_csv.py
from io import open
from collections import namedtuple
import csv

with open('ex.8.1.csv', 'r', encoding='ascii') as fin:
    csvreader = csv.reader(fin, delimiter=',')
    header = next(csvreader) # 根據表頭建立具名元祖
    Data = namedtuple('Header', ','.join(header))
    for row in csvreader:
        d = Data._make(row) # 將資料放入具名元祖的物件
        print(d.name, d.eng, d.history, d.math)
# %%
Point = namedtuple('Point', 'x, y, z')
pa = Point._make([1, 2, 3])
print(pa._asdict())

pa._replace(x = 10)
print(pa)

print(pa._fields)
# %% 雙向佇列 (Deque)
from collections import deque
d = deque()
d.append(1)
d.extend([3, 5, 7, 9])
print(d)
print(d.pop())
print(d.popleft())
d.reverse()
print(d)
d.rotate(2)
print(d)
# %%
ts = deque(maxlen=5)
ts.append(30.5)
ts.append(29.4)
ts.append(30.3)
ts.append(31.5)
ts.append(23.4)
ts.append(33.3)
print(ts)
print(sum(ts)/len(ts))
# %% 計數器 (Counter)

