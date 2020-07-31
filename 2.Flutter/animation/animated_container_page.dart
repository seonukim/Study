import 'package:url_launcher/url_launcher.dart';
import 'package:flutter/material.dart';
import 'dart:math';

class AnimatedContainerPage extends StatefulWidget {
  @override
  _AnimatedContainerPageState createState() => _AnimatedContainerPageState();
}

class _AnimatedContainerPageState extends State<AnimatedContainerPage> {
  var _size = 100.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AnimatedContainer'),
        actions: <Widget>[
          IconButton(
            onPressed: () {
              launchURL(
                  'https://github.com/junsuk5/flutter_basic/blob/3d00fee10e1c353df822cce0db6fa027958c251d/chapter04/lib/animation/animated_container_page.dart');
            },
            icon: Image.asset('assets/github_icon.png'),
          ),
        ],
      ),
      body: Center(
        child: GestureDetector(
          onTap: () {
            final random = Random();
            setState(() {
              _size = random.nextInt(200).toDouble() + 100;
            });
          },
          child: AnimatedContainer(
            duration: Duration(seconds: 1),
            width: _size,
            height: _size,
            child: Image.asset('assets/sample.jpg'),
            curve: Curves.fastOutSlowIn,
          ),
        ),
      ),
    );
  }

  launchURL(String url) async {
    if (await canLaunch(url)) {
      await launch(url);
    } else {
      throw 'Could not launch $url';
    }
  }}