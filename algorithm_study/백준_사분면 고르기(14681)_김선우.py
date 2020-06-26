x = int(input())
y = int(input())

if (x > 0) & (y > 0):
    print(1)
elif (x < 0) & (y > 0):
    print(2)
elif (x < 0) & (y < 0):
    print(3)
else:
    print(4)