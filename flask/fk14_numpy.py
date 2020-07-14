# DB에서 테이블 가져와서 넘파이로 저장하기

import pymssql as ms
import numpy as np

conn = ms.connect(server = '127.0.0.1', user = 'bit2',
                  password = '1234', database = 'bitdb')
cursor = conn.cursor()
cursor.execute('SELECT * FROM iris2;')
row = cursor.fetchall()
conn.close()
aaa = np.asarray(row)
np.save('./data/test_flask_iris2.npy', arr = aaa)


print(row)
print("=" * 40)
print(aaa)
print(aaa.shape)
print(type(aaa))