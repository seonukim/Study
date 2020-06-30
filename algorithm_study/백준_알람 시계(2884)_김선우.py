import sys
input = sys.stdin.readline

h, m = map(int, input().split())
if  m - 45 < 0 & h > 0:
    h = h - 1
    m = m + 15
elif m - 45 < 0 & h == 0:
    h = h + 23
    m = m + 15
elif m - 45 >= 0:
    h = h
    m = m - 45

print(h, m)