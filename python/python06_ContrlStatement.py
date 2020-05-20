'''
제어문:
    코드의 실행을 제어하는 문법
'''

# 1. for문
a = {'name': 'yun', 'phone': '010', 'birth': '0511'}

for seonu in a.keys():
    print(seonu)

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in a:
    i = i * i
    print(i)
    # print('melong')

for i in a:
    print(i)

## for문과 if문 조합
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 :", number, "명")


# 2. while문
'''
while 조건문 :      # 조건이 '참'일 동안 계속 도는 형태
    수행할 문장
'''


# 3. 조건문
'''
if 조건문 :
    조건이 '참'일 경우 실행할 코드
else :
    조건이 '거짓'일 경우 실행할 코드
'''
if 1:
    print('True')
else:
    print('False')

if 3:
    print('True')
else:
    print('False')

if 0:
    print('True')
else:
    print('False')

if -1:
    print('True')
else:
    print('False')
'''위 코드를 통해 알 수 있는 것; 머신은 0을 제외한 나머지는 True로 인식함'''

'''
비교연산자
 <,     >,   ==,    !=,        >=,         <=
크다, 작다, 같다, 다르다, 크거나 같다, 작거나 같다
'''
a = 1           # 대입연산자(변수에 할당)
if a == 1:      # 비교연산자(같다)
    print('출력 잘 돼')

money = 10000
if money >= 30000:
    print('한우 먹자')
else:
    print('라면 먹자')

'''
조건연산자
 and,    or,   not
그리고, 또는, 아니다
'''
money = 20000
card = 1
if money >= 30000 or card == 1:
    print('한우 먹자')
else:
    print('라면 먹자')

##############################################
# break, continue
print("=" * 20 , "break", "=" * 20)
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1
print("합격인원 :", number, "명")

print("=" * 20 , "continue", "=" * 20)
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 60:
        continue
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1
print("합격인원 :", number, "명")