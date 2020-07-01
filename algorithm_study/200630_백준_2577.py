import sys
a = int(sys.stdin.readline())
b = int(sys.stdin.readline())
c = int(sys.stdin.readline())
x = str(a * b * c)
tmp = []

tmp.append(x.count("0"))
tmp.append(x.count("1"))
tmp.append(x.count("2"))
tmp.append(x.count("3"))
tmp.append(x.count("4"))
tmp.append(x.count("5"))
tmp.append(x.count("6"))
tmp.append(x.count("7"))
tmp.append(x.count("8"))
tmp.append(x.count("9"))

for i in range(len(tmp)):
    print(tmp[i])