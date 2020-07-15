## DB를 바탕으로 웹 게시판 만들기

from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터베이스
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")             # 실행하기
print(cursor.fetchall())                            # fetchall : 전체 출력

@app.route('/')                 # '/' : port 번호까지만 치면 웹을 보여준다
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall();
    return render_template("board_index.html", rows = rows)

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id = ' + str(id))
    rows = c.fetchall();
    return render_template('board_modi.html', rows = rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.methods == 'POST':
        try:                    # 예외 처리
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/wanggun.db") as conn:
                cur = conn.cursor()
                cur.execute("UPDATE general SET war=" + str(war) + "WHERE id=" + str(id))
                conn.commit()
                msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()                 # 에러 뜨면 되돌리기
            msg = '입력 과정에서 에러가 발생했습니다.'
        
        finally:
            return render_template('board_result.html', msg = msg)
            conn.close()

app.run(host='127.0.0.1', port=5001, debug=False)