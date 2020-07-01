import sys
n = int(sys.stdin.readline())
tmp = list(map(int, sys.stdin.readline().split()))
print(min(tmp), max(tmp))