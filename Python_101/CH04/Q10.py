# Ch04 Q10

def array_2d_mul(m1, m2):
    if len(m1[0]) != len(m2):
        print(f'Error!!!! array size is wrong: {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}')
    else:
        m = len(m1)
        n = len(m2)
        p = len(m2[0])
        m_new = []
        for i in range(m):
            s1 = m1[i]
            # print(s1)
            s_n = []
            for k in range(p):
                s2 = []
                for j in range(n):
                    s2.append(m2[j][k])
                # print(s2)

                r = 0
                for e1, e2 in zip(s1, s2):
                    r += e1*e2
                # print(f'after multiply = {r}')
                s_n.append(r)
            m_new.append(s_n)
            
                

        # s = [[m1[i][j]*m2[j][k] for j in range[n] for k in range(p)] for i in range(m)]
    return m_new


m1 = [[9, 13, 5], [1, 11, 7], [3, 9, 2], [6, 0, 7]]
m2 = [[1, 2], [3, 4], [5, 6]]

result = array_2d_mul(m1, m2)
print(result)