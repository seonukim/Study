# range 함수
a = range(10)
print(a)

b = range(1, 11)
print(b)

for i in a:
    print(i)

for i in b:
    print(i)

print(type(a))

sum = 0
for i in range(1, 11):
    sum = sum + i
print(sum)