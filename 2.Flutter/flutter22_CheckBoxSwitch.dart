import 'package:flutter/material.dart';

void main(){
  runApp(
    MyApp(),
  );
}

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


// StatefulWidget
class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  var _isChecked = false;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('CheckBox / Radio / Switch'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Checkbox(
                value: _isChecked,
                onChanged: (value){
                  setState(() {
                    _isChecked = value;
                  });
                },
              ),
              SizedBox(
                height: 40,
              ),
              Switch(
                value: _isChecked,
                onChanged: (value){
                  setState(() {
                    _isChecked = value;
                  });
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
