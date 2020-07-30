import 'package:flutter/material.dart';

class ButtonPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Button'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            RaisedButton(
              child: Text('RaisedButton'),
              color: Colors.orange, // 배경 색상
              onPressed: () {},
            ),
            FlatButton(
              child: Text('FlatButton'),
              onPressed: () {},
            ),
            IconButton(
              icon: Icon(Icons.add),
              color: Colors.red,  // 아이콘 색상
              iconSize: 100.0, // 기본값 24.0
              onPressed: () {},
            ),
            FloatingActionButton(
              child: Icon(Icons.add),
              onPressed: () {},
            ),
          ],
        ),
      ),
    );
  }
}
