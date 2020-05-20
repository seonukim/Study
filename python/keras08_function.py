'''
파이썬 - 함수
'''
# 함수를 정의하는 방법
# def 함수명(자유롭게, 이해하기 쉽게)(인자):
#     return 반환 값
def sum1(a, b):
    return a + b

print(sum1(3, 4))


a = 1
b = 2
c = sum1(a, b)

print(c)

## 곱셈, 나눗셈, 뺄셈 함수를 만드시오
# mult1, div1, sub1

# 1. 곱셈 함수
def mult1(a, b):
    return a * b

# 2. 나눗셈 함수
def div1(a, b):
    return a // b

# 3. 뺄셈 함수
def sub1(a, b):
    return a - b

def sayYeh():
    return 'Hi'

aaa = sayYeh()
print(aaa)