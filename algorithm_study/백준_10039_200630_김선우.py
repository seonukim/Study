import sys
a = int(sys.stdin.readline())
b = int(sys.stdin.readline())
c = int(sys.stdin.readline())
d = int(sys.stdin.readline())
e = int(sys.stdin.readline())
x = [a, b, c, d, e]
tmp_x = []
for i in x:
    if i >= 40:
        i = i
    elif i < 40:
        i = 40
    tmp_x.append(i)
    avg = sum(tmp_x) // len(tmp_x)
print(avg)