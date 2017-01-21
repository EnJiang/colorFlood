#What's color flood?
  Color flood is a little game I find very interesting.<br>
  In this game you have a n plus n borad (grid?) with multiple colors on it, each one grid has one color respectively. The main goal is to "flood" the borad in one color. The thing is, first take look at the color of the top-left grid(1,1), and if its neibors((1,2) and (2,1)) have the same color with it, they are joint as a area(I call it the "target area"). A move you make is to pick one kind of color and then the target area changes into it. If you do it right, now your target area should be larger than before. Say, if your target area is originally red, and you pick blue, all the blue grids that were at the edge of you red targrt area now join in and become a partner.<br>
  And there is a simple way to measure how well you play, which is the sum of moves your made to get a one-colored board.<br>
  [You can play it here.](http://unixpapa.com/floodit/)

#What is this repo?
  I am trying to do several things...<br>
  Firstly of course to right a color flood of myself, which I alread done (in game.py). Well and it turns out this one is not so efficient, so I might want to shorten the speed it needs every time to flood.<br>
  The import part is that I find it interesting to apply all kinds of algrithm on it to solve the problem, I mean it's always more fun to learn by practicing.<br>
  Any good way to improve the preformance or any elegent algrithm to get (or approch) a optimal solve, please start a issue.

