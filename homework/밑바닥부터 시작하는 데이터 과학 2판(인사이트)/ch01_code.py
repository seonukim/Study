# Chapter 01 _ 1.3.1
users = [{"id": 0, "name": "Hero"},
         {"id": 1, "name": "Dunn"},
         {"id": 2, "name": "Sue"},
         {"id": 3, "name": "Chi"},
         {"id": 4, "name": "Thor"},
         {"id": 5, "name": "Clive"},
         {"id": 6, "name": "Hicks"},
         {"id": 7, "name": "Devin"},
         {"id": 8, "name": "Kate"},
         {"id": 9, "name": "Klein"}]

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# 사용자별로 비어 있는 친구 목록 리스트를 지정하여 딕셔너리를 초기화
friendships = {user["id"]: [] for user in users}

# friendship_pairs 내 쌍을 차례대로 살펴보면서 딕셔너리 안에 추가
for i, j in friendship_pairs:
    friendships[i].append(j)     # j를 사용자 i의 친구로 추가
    friendships[j].append(i)     # i를 사용자 j의 친구로 추가

# friendship 내의 모든 리스트의 길이를 더해서 총 연결 수 구하기
def number_of_friends(user):
    '''user의 친구는 몇 명일까?'''
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user) for user in users)
print(total_connections)        # 24

num_users = len(users)
avg_connections = total_connections / num_users

print(avg_connections)      # 24 / 10 == 2.4


# 연결 수가 가장 많은 사람, 즉 친구가 가장 많은 사람 알아보기
# (user_id, number_of_friends)로 구성된 리스트 생성
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]

num_friends_by_id.sort(                             # sort() 는 정렬 함수
    key = lambda id_and_friends: id_and_friends[1], # num_friends 기준으로
    reverse = True)                                 # 내림차순 정렬


# Chapter 01 _ 1.3.2
# 데이터 과학자 추천하기
# 친구의 친구 소개하기 - 각 사용자의 친구에 대해 그 친구의 친구들을 살펴보고 결과 저장
def foaf_ids_bad(user):
    # "foaf"는 친구의 친구("friend of a friend")를 의미함
    return [foaf_id
            for friend_id in friendships[user["id"]]
            for foaf_id in friendships[friend_id]]

# foaf_ids_bad() 함수를 users[0]으로 인자를 넣어 실행해보기
print(foaf_ids_bad(users[0]))       # [0, 2, 3, 0, 1, 3]
'''
Hero도 자신의 친구의 '친구'이므로 사용자 0(자기 자신)이 두 번 포함됨
'''

print(friendships[0])   # [1, 2]
print(friendships[1])   # [0, 2, 3]
print(friendships[2])   # [0, 1, 3]

# 서로가 함께 아는 친구가 몇 명인지 세어보는 함수,
# 동시에 사용자가 이미 아는 사람을 제외하는 함수
from collections import Counter

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]       # 사용자의 친구 개개인에 대해
        for foaf_id in friendships[friend_id]       # 그들의 친구들을 세어 보고
        if foaf_id != user_id                       # 사용자 자신과
        and foaf_id not in friendships[user_id])    # 사용자의 친구는 제외

print(friends_of_friends(users[3]))     # Counter({0: 2, 5: 1})

'''p.8 ~ 9는 제외함'''


# Chapter 01 _ 1.3.3  생략
