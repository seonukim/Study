h, m = map(int, input().split())
time = 0
if h == 0:
    h = 24
elif m <= 45:
    m = 0
time = h * 60 + m
alarm = time - 45
a_hour = int(alarm // 60)
a_minute = alarm % 60
print(a_hour, a_minute)