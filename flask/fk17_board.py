from flask import Flask, render_template, request
import sqlite3
import os
path = 'D:/Study/flask/template/'
os.chdir(path)

app = Flask(__name__)

# 데이터베이스
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall();
    return render_template(path + "board_index.html", rows=rows)

# /modi
@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id='+str(id))   # id 를 설정해주면 설정한 id에 대한 값만 출력. 
    rows = c.fetchall();
    return render_template(path + 'board_modi.html', rows=rows)

# /addrec
@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/wanggun.db") as conn:
                cur = conn.cursor()
                cur.execute("UPDATE general SET war="+str(war)+ "WHERE id="+str(id))
                conn.commit()
                msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '입력 과정에서 에러가 발생했습니다.'
        
        finally:
            return render_template(path + 'board_result.html', msg = msg)
            conn.close()

app.run(host='127.0.0.1', port=5001, debug=False)