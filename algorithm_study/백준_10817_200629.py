import sys
a, b, c = map(int, sys.stdin.readline().split())
x = sorted([a, b, c])
x = x[1]
print(x)