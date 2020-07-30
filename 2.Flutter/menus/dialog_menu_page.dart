import 'package:flutter/material.dart';
import 'package:flutter_basic/basic/circle_avatar_page.dart';
import 'package:flutter_basic/basic/icon_page.dart';
import 'package:flutter_basic/basic/image_page.dart';
import 'package:flutter_basic/basic/progress_page.dart';
import 'package:flutter_basic/basic/text_page.dart';
import 'package:flutter_basic/dialog/alert_dialog_page.dart';
import 'package:flutter_basic/dialog/date_picker_page.dart';
import 'package:flutter_basic/dialog/time_picker_page.dart';
import 'package:flutter_basic/input/check_switch_page.dart';
import 'package:flutter_basic/input/dropdown_page.dart';
import 'package:flutter_basic/input/radio_page.dart';
import 'package:flutter_basic/input/textfield_page.dart';

class DialogMenuPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('4.7 다이얼로그'),
      ),
      body: ListView(
        children: <Widget>[
          ListTile(
            title: Text('AlertDialog'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => AlertDialogPage()),
              );
            },
          ),
          ListTile(
            title: Text('DatePicker'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => DatePickerPage()),
              );
            },
          ),
          ListTile(
            title: Text('TimePicker'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => TimePickerPage()),
              );
            },
          ),
        ],
      ),
    );
  }
}
