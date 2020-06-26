# a = 0
# for i in range(5):
    

x,y,z = 0,0,0
chk=False
for x in range(1,7):
    for y in range(1,7):
        for z in range(1,7):
            if x !=y and y!=z and z!=x:
                if x*y*z == x+y+z+3:
                    print(x,y,z)
                    chk=True
                    break
        if chk:
            break
    if chk:
        break