import 'package:flutter/material.dart';
import 'package:flutter_basic/flutter02_main.dart';

class RaisedButtonPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('RaisedButtonPage'),
        actions: <Widget>[
          IconButton(
            onPressed: () {
              launchURL(
                  'https://github.com/junsuk5/flutter_basic/blob/3d00fee10e1c353df822cce0db6fa027958c251d/chapter04/lib/button/raised_button_page.dart');
            },
            icon: Image.asset('assets/github_icon.png'),
          ),
        ],
      ),
      body: Center(
        child:
        RaisedButton(
          child: Text('RaisedButton'),
          color: Colors.orange, // 배경 색상
          onPressed: () {},
        ),
      ),
    );
  }
}
