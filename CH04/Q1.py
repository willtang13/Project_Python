#Ch04 Q1

def unique(li):
    ui_li = []
    for m in li:
        if m in ui_li:
            continue
        else:
            ui_li.append(m)
    return ui_li

def main():
    li = [9,5,5,-4,7,6,4,1,-2,0,10,9,7]

    print(li)
    print('[Method 1]only show unique numbers by using set')
    ui_li = list(set(li))
    print(ui_li)

    print('[Method 2]only show unique numbers by using function')
    ui_li2 = unique(li)
    print(ui_li2)

if __name__ == '__main__':
    main()
