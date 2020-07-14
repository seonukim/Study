## MSSQL에서 파이썬으로 데이터 가져오기

import pymssql as ms
print(f'Version of pymssql : {ms.__version__}')               # Version of pymssql : 2.1.4

conn = ms.connect(server = '127.0.0.1', user = 'bit2',
                  password = '1234', database = 'bitdb')

cursor = conn.cursor()

# cursor.execute("SELECT * FROM iris2;")
# cursor.execute("SELECT * FROM wine;")
# cursor.execute("SELECT * FROM sonar;")
cursor.execute("SELECT * FROM Test01;")

rows = cursor.fetchone()

while rows :
    print(f'\n컬럼1: {rows[0]} \n컬럼2: {rows[1]} \n컬럼3: {rows[2]} \n컬럼4: {rows[3]}')
    row = cursor.fetchone()

conn.close()