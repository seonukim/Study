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
}     // 공통 코드

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return DefaultTabController(      // Scaffold를 감싸고
      length: 3,      // 탭 수 지정
      child: Scaffold(
        appBar: AppBar(
          title: Text('제목'),
          bottom: TabBar(       // Scaffold의 bottom 프로퍼티에 TabBar 지정
            tabs: <Widget>[     // tabs 프로퍼티에 Tab의 리스트 지정
              Tab(icon: Icon(Icons.tag_faces)),
              Tab(text: '메뉴2'),
              Tab(icon: Icon(Icons.info), text: '메뉴3'),
            ],
          ),
        ),
        body: TabBarView(
          children: <Widget>[
            Container(color: Colors.yellow,),
            Container(color: Colors.orange,),
            Container(color: Colors.red,),
          ],
        ),
      ),
    );
  }
}

