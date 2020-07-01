import sys
a = []
for i in range(9):
    a.append(int(sys.stdin.readline()))
print(max(a))
print(int(a.index(max(a))) + 1)