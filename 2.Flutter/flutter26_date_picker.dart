import 'package:flutter/material.dart';
import 'package:flutter_basic/dialog/date_picker_page.dart';
import 'package:flutter_basic/flutter02_main.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: DatePickerPage(),
    );
  }
}

class DatePickerPage extends StatefulWidget {
  @override
  _DatePickerPageState createState() => _DatePickerPageState();
}

class _DatePickerPageState extends State<DatePickerPage> {
  DateTime _selectedTime;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Date Picker'),
        actions: <Widget>[
          IconButton(
            onPressed: () {
              launchURL(
                'https://github.com/junsuk5/flutter_basic/blob/3d00fee10e1c353df822cce0db6fa027958c251d/chapter04/lib/dialog/date_picker_page.dart'
              );
            },
            icon: Image.asset('github_icon.png'),
          )
        ],
      ),
      body: Column(
        children: <Widget>[
          RaisedButton(
            onPressed: () {
              Future<DateTime> selectedDate = showDatePicker(
                context: context,
                initialDate: DateTime.now(),    // 초깃값
                firstDate: DateTime(2018),      // 시작일 2018.01.01
                lastDate: DateTime(2030),       // 마지막일 2030.01.01
                builder: (BuildContext context, Widget child) {
                  return Theme(
                    data: ThemeData.dark(),     // 다크 테마
                    child: child,
                  );
                },
              );
            },
          ),
        ],
      ),
    );
  }
}

