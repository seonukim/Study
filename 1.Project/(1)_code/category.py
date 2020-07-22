import os
import re
import numpy as np
'''
# category
def categories(top_folder_path):
    path = top_folder_path

    categories = []

    regex = re.compile(r'([A-Za-z]+)')

    for root, dir, f in os.walk(path):
        # print(root)
        # 종 이름만 가져오기
        category = regex.findall(root)
        length = len(category)

        if length == 7:
            category = category[-1]
        elif length == 8:
            category = ' '.join(category[-2:])
        else:
            category = ' '.join(category[-3:])

        categories.append(category)
        # print(category)

    return np.array(categories)

cate = categories(path)
print(cate)

#----------------------------------------------------------------------------------

def categories(top_folder_path):                
    path = top_folder_path

    categories = []
    regex = re.compile('[$-]+([A-Za-z\-_]+)') # '-'을 기준으로 뒷 글자 가져오기

    for root, dir, f in os.walk(path):
        # print(root)
        # 종 이름만 가져오기
        category = regex.findall(root)
        categories.extend(category)
        # print(category)
    return np.array(categories)
'''
def categories(top_folder_path):                # folder이름으로 카테고리 생성
    category = os.listdir(top_folder_path)
    return category