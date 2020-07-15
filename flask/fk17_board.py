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
    return render_template("./template/board_index.html", rows = rows)

@app.route('/modi')


app.run(host = '127.0.0.1', port = 5001, debug = False)