# Ch04 Q8

def under_60(li):
    out = []
    for i in li:
        if i < 60:
            out.append(i)
    return out

score = [35, 78, 99, 58, 60, 78, 95, 25]
score_under_60 = under_60(score)
print(score_under_60) 

score_under_60_2 = [i for i in score if i < 60]
print(score_under_60_2)