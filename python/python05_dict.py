# 3. 딕셔너리 ; 중복 x
# {키 : 밸류}
# {key : value}

a = {1: 'hi', 2: 'hello'}
print(a)
print(a[1])

b = {'hi': 1, 'hello': 2}
print(b['hello'])


# 딕셔너리 요소 삭제
del a[1]    # del은 딕셔너리의 요소 삭제
print(a)

del a[2]
print(a)

a = {1: 'a', 1: 'b', 1: 'c'}    # 키 값이 중복되었을 경우, 가장 마지막 원소가 나옴
print(a)

b = {1: 'a', 2: 'a', 3: 'a'}
print(b)        # 밸류가 중복되었을 경우, 상관 없이 모두 정상 출력됨
'''여기서 알 수 있는 점, 키는 중복이 허용되지 않지만, 밸류 값은 중복이 허용된다.'''


a = {'name': 'yun', 'phone': '010', 'birth': '0511'}
print(a)
print(a.keys())
print(a.values())
print(type(a))
print(a.get('name'))
print(a.get('phone'))
print(a['phone'])