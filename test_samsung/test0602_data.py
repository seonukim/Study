import numpy as np
import pandas as pd 

samsung = pd.read_csv('./data/csv/samsung.csv',
                      index_col = 0,    #None
                      header = 0, 
                      sep = ',',
                      encoding = 'CP949')
                       
hite = pd. read_csv('./data/csv/hite.csv',
                    index_col = 0,       #None
                    header = 0, 
                    sep = ',',
                    encoding = 'CP949')

# print(samsung)
# print(hite.head())
# print(samsung.shape)     #(700 , 1)
# print(hite.shape)        #(700 , 5)

#Nan 제거1

samsung = samsung.dropna(axis = 0)      # Non이 들어가 있는 행을 삭제하겠다
# samsung= samsung.dropna(how='all')

print(samsung)
# print(samsung.shape)     #(  ,   1)


hite = hite.fillna(method='bfill')  # bfill= backfill 전날 값으로 채우겠다.
# hite = hite.dropna(how='all')
# hite.iloc[0:1:] = hite.iloc[0,1:].fillna(value = str(0))
# hite1= hite.iloc[1:, :].copy()


hite = hite.dropna(axis = 0)           # Nan이 들어간 행을 지우겠다.
 

print(hite)


#Nan 제거2

# hite = hite[0:509]

# hite.iloc[0,1:5] = ['38750','36000','38750','1407345']   #iloc은 index location 

# hite.loc["2020-06-02", '고가':'거래량'] = ['10','20','30','40']
print(hite.head)



#삼성과 하이트의 정렬을 오름차순으로 변경
samsung = samsung.sort_values(['일자'], ascending = ['True']) #ascending 오름차순
hite = hite.sort_values(['일자'], ascending = ['True'])       #descending 내림차순

# print(samsung)
# print(hite)


#콤마제거, 문자를 정수로 변환
for i in range(len(samsung.index)): 
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))

print(samsung)
  

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',','') )

print(hite)
print(type(hite.iloc[1,1]))

print(samsung.shape)      #(509,1)
print(hite.shape)         #(509,5)

samsung = samsung.values
hite = hite.values

print(type(hite))         #class 'numpy.ndarray'

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)