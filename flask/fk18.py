from flask import Flask, render_template, request
import sqlite3

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
pwd = '1234'

app = Flask(__name__)

# database create
conn = sqlite3.connect('./data/wanggun.db')

cursor = conn.cursor()

sql = 'SELECT * from general' 
cursor.execute(sql)

print(cursor.fetchall())

conn.close()

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')

    c = conn.cursor()
    sql = 'SELECT * from general'
    c.execute(sql)
    rows = c.fetchall()

    conn.close()
    return render_template('board_index.html', rows=rows)

@app.route('/modi') # ?다음은 무시 되는건가/??
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    sql = 'SELECT * from general where id = (?)'
    c.execute(sql, (str(id),))
    rows = c.fetchall()
    print(rows)

    conn.close()
    return render_template('board_modi.html', rows=rows)

@app.route('/addrec',methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try :
            wars = request.form['war']
            ids = request.form['id']
            print(wars)
            print(ids)
            ids = ids[:-1]
            print(ids)
            with sqlite3.connect('./data/wanggun.db') as con:
                c1 = con.cursor()
                sql = 'UPDATE general SET war = (?)  WHERE id = (?)'
                c1.execute(sql, (str(wars),str(ids)))
                con.commit()
                msg = '정상적으로 입력되었습니다'
            
        except :
            con.rollback()
            msg = '입력과정에서 에러가 발생했습니다.'

        finally :
            return render_template('board_result.html', msg=msg)

if __name__ == '__main__':
    app.run(host=server, port=7000, debug=False)
