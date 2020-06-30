import sys
input = sys.stdin.readline

tmp = [int(input()) for i in range(10)]

for i in range(10):
    tmp[i] = tmp[i] % 42
    