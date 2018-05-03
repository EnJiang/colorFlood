# What's color flood?
  Color flood is a little game I find very interesting.<br>
  In this game you have a n plus n borad (grid?) with multiple colors on it, each one grid has one color respectively. The main goal is to "flood" the borad in one color. The thing is, first take look at the color of the top-left grid(1,1), and if its neibors((1,2) and (2,1)) have the same color with it, they are joint as a area(I call it the "target area"). A move you make is to pick one kind of color and then the target area changes into it. If you do it right, now your target area should be larger than before. Say, if your target area is originally red, and you pick blue, all the blue grids that were at the edge of you red targrt area now join in and become a partner.<br>
  And there is a simple way to measure how well you play, which is the sum of moves your made to get a one-colored board.<br>
  [You can play it here.](http://unixpapa.com/floodit/)<br>
  Update at 2017/11/17: The problem is [proven](http://www.bris.ac.uk/news/2010/6945.html) to be NP-hard, meaning (almost...) no polynomial solution, however if a nice, namely admissible heuristic could be found, we might still get a optimal solution in acceptable time. As a result I might focus on A* algorithm from now on.<br>
  Update at 2018/05/02: I tried A* already and the result is fairly good. The admissible version of my heuristic is too slow, so I used a non-admissible one, but I still got result far better than greedy and myself.

# What is this repo?
  I am trying to do several things...<br>
  Firstly of course to implement a color flood of myself, which I alread done (in game.py). Well and it turns out this one is not so efficient, so I might want to shorten the speed it needs every time to flood.<br>
  The import part is that I find it interesting to apply all kinds of algrithm on it to solve the problem, I mean it's always more fun to learn by practicing.<br>
  Any good way to improve the preformance or any elegent algrithm to get (or approch) a optimal solution, please start a issue.
s
# algrithms
### Random
  Code in randomPIck.py<br>
  Pick one color randomly from all colors other than the previous one, as picking same color twice has no meaning.<br>
  The average steps is around 46.8 in 10000 tests.
### Greedy
  Code in greedy.py<br>
  The code examine all possible "n depth" continious choices(let's call it a path) and pick the best path until the board is filled. For example "red green blue" would be a 3 depth path. There is no guarantee that in this way we will get a best solution, in fact in my test, the larger I set the depth the shorter path I get averagely. At depth 1 the algrithm has a average 23-step long path in 29912 tests. Pretty awsome! However greedy algrithm can not give me a optimal path with a acceptable complexity(5^n for n depth).
### A*
  Code in A_star.py<br>
  I implemented an [A* search algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) where f(n) = g(n) + h(n). Here g(n) is the actual step taken at a certain state, h(n) is a estimated cost still needed.
  I used the formula h(n) = max(m, color_left - 1) +　(144 - flood_area) * 1.0 / 144.
  This is not an admissable formula so the result is (very likely) not optimal. Also, this approch is very time and memory consuming. In 203 random test, I get a average of 18.5 step per path.s

### DQN
  Code in dqn.py<br>
  I tried to use Deep Q-learning to solve this problem. However, it seems that with only the board as input, full connection and convolution network can't do the job right. I tried to add more features but things still did not work out. Might use an AlphaGo like system to solve this problem in the future...
