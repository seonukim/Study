import 'package:flutter/material.dart';
import 'package:flutter_basic/animation/animated_container_page.dart';
import 'package:flutter_basic/animation/hero_page.dart';
import 'package:flutter_basic/animation/sliver_fillremaining_page.dart';
import 'package:flutter_basic/animation/sliver_list_page.dart';
import 'package:flutter_basic/basic/circle_avatar_page.dart';
import 'package:flutter_basic/basic/icon_page.dart';
import 'package:flutter_basic/basic/image_page.dart';
import 'package:flutter_basic/basic/progress_page.dart';
import 'package:flutter_basic/basic/text_page.dart';
import 'package:flutter_basic/flutter02_main.dart';

class AnimationMenuPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('4.9 애니메이션'),
      ),
      body: ListView(
        children: <Widget>[
          ListTile(
            title: Text('Hero'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => HeroPage()),
              );
            },
          ),
          ListTile(
            title: Text('AnimatedContainer'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => AnimatedContainerPage()),
              );
            },
          ),
          ListTile(
            title: Text('SliverAppBar / SliverList'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SliverListPage()),
              );
            },
          ),
          ListTile(
            title: Text('SliverAppBar / SliverFillRemaining'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SliverFillRemainingPage()),
              );
            },
          ),
        ],
      ),
    );
  }
}
