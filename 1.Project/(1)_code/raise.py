## raise : 예외 발생시키기


list = []
try:
    while True:
        print(f'아이템 갯수 : {len(list)}')
        print(f'인벤토리 : {list}')
        # 인벤토리가 꽉 찼을 때, 예외를 일부러 발생시킴
        if len(list) >= 4:
            raise Exception('인벤토리 부족')
        item = 'item' + str(len(list))
        list.append(item)
except Exception as e:
    print('인벤토리가 꽉 찼습니다.')
    print(e)