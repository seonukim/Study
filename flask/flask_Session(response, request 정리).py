'''
## Session ##
 쿠키와 다르게 세션과 관련된 데이터는 서버에 저장된다. (쿠키는 시간이 지나면 소멸)
서버에서 관리할 수 있다는 점에서 안전성이 좋아서 보통 로그인 관련으로 사용됨
Flask에서 세션은 '딕셔너리' 형태로 저장되며 키를 통해 해당 값을 불러올 수 있다.

Session을 사용하기 위해서는 해당 값을 암호화하기 위한 Key 값을 코드에서 지정해주어야 한다.
'''

from datetime import timedelta
from flask import Flask, request, session, redirect, url_for, app
app = Flask(__name__)
app.secret_key = 'any random string'

# /URL, 조건문으로 Flask 세션 정보 안에 'username'이라는 세션 정보가
# 있냐, 없냐에 따라서 로그인을 했는지 안했는지 판단함
@app.route('/')
def index():
    if 'username' in session:
        username = session['username']
        return 'Logged in as ' + username + '<br>' + \
            "<b><a href = '/logout'>click here to log out</a></b>"
    
    return "You are not logged in <br><a href = '/login'></b>" + \
        "click here to log in</b></a>"


# 실제로 로그인을 하는 양식이 있는 /login URL
# 맨 처음 접속했을 때는 GET 메소드로 요청이 오게 되므로,
# 로그인을 하기 위한 양식을 전송한다.
# 양식을 통해 POST 요청이 오면 username이라는 세션을 생성하여
# 입력 받은 양식의 데이터를 세션에 저장한 후, 맨 처음 페이지로
# redirection한다.
@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''

    <form action = "" method = "post">
        <p><input type = text name = username/></p>
        <p<<input type = submit value = Login/></p>
    </form>

    '''

# 로그아웃의 경우 해당 세션 정보를 제거하는 것으로 연결을 끊는 것이 가능하다.
# 세션의 제거는 pop 메소드를 사용해서 아래처럼 코드를 작성한다.
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))

# Flask에서의 각 세션의 유효기간 : default 31일
# 세션 유효기간 직접 설정하기
@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes = 5)

if __name__ == '__main__':
    app.run(host = '127.0.0.1')