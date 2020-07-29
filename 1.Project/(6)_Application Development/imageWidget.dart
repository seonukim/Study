import 'package:flutter/material.dart';
import 'package:flutter/src/services/asset_bundle.dart';

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
    return Scaffold(
      appBar: AppBar(
        title: Text('Image'),
      ),
      body: Align(
        alignment: Alignment.center,      // 가운데 정렬
        child: Image.network('http://bit.ly/2Pvz4t8'),    // 이미지 URL
      ),
    );
  }
}
