import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('제목'),
      ),
      body: Align(
        alignment: Alignment.center,      // 가운데로 정렬
        child: RaisedButton(
          child: Text('RaisedButton'),
          color: Colors.orange,
          onPressed: () {
            // 실행될 코드 작성
          },
        ),
      ),
    );
  }
}
