from flask import Flask, Response, make_response, render_template

app = Flask(__name__)

@app.route('/')
def response_test():
    custom_response = Response('[★] Custom Response', 200,
                      {"Program": "Flask Web Application"})
    return make_response(custom_response)

@app.before_first_request
def before_first_request():
    print('[1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.')

@app.before_request
def before_request():
    print('[2] 매 HTTP 요청이 처리되기 전에 실행됩니다.')

@app.after_request
def after_request(response):
    print('[3] 매 HTTP 요청이 처리되고 나서 실행됩니다.')
    return response         # response 반환해줌

@app.teardown_request
def teardown_request(exception):
    print('[4] 매 HTTP 요청의 결과가 브라우저에 응답하고나서 호출된다.')
    return exception

@app.teardown_appcontext
def teardown_appcontext(exception):
    print('[5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다.')
    return exception


if __name__ == '__main__':
    app.run(host = '127.0.0.1')

'''
과제 1: tf21 완성
과제 2: session, cookie
과제 3: response, request 
'''

'''
## Response ##
Request에서 보낸 요청에 대한 응답 변수

## Request Module ##
클라이언트에서 서버로 어떤 요청(GET, POST 등)을 보낼지 정해준다.
'''