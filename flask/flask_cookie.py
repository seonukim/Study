'''
## Cookie ##
 쿠키는 클라이언트의 PC의 텍스트 파일 형태로 저장되는 것으로
일반적으로는 시간이 지나면 소멸한다. 보통 세션과 더불어 자동 로그인,
팝업 창에서 '오늘은 이 창을 더 이상 보지 않기' 등의 기능을
클라이언트에 저장해놓기 위해 사용된다.

 웹 페이지에서 폼을 전송 받으며, 클라이언트에 쿠키를 넘겨주는 코드이다.
'''
from flask import Flask, Response, make_response, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/setcookie', methods = ['POST', 'GET'])
def setcookie():
    if request.method == 'POST':
        user = request.form['nm']

        resp = make_response('Cookie Setting Complete')
        resp.set_cookie('userID', user)

        return resp

@app.route('/getcookie')
def getcookie():
    name = request.cookies.get('userID')
    return '<h1>welcome ' +name+'</h1>'

if __name__ == '__main__':
    app.run(host = '127.0.0.1')
