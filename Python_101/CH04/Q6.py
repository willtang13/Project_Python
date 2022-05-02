#Ch04 Q6

def cumulative_product(li):
    p = 1
    out = []
    for i in li:
        p *= i
        out.append(p)

    return out

li = [1, 2, 3, 4, 5]
out = cumulative_product(li)

print(out)