h, m = map(int, input().split())
time = 0
if h == 0 and m - 45 < 0:
    h = h + 23
    m = m + 15
    print(h,m)
else:
    time = h*60+m
    alarm = time - 45
    a_hour = alarm // 60
    a_minute = alarm % 60
    print(a_hour, a_minute)