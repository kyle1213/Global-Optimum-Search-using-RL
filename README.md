# Global-Optimum-Search-using-RL
global optimum search using reinforcement learning on pytorch  

구현 방법:
+ RL model: DDPG(actor-critic)
+ reward: dy, -1 each step,
+ actions: x
+ state: (x, y, dx)
+ env: random 1 variable function

결과 분석:
+ optimum근처가면 더 빨리 도착해도, 멀리서 온 상황보다 score가 낮다. reward를 고쳐야하지 않을까.


# Train results
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/main/train%20result/env1.png)  
trained on env1 test env0  
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/main/gifs/env1%20env0.gif)  
trained on env1 test env1  
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/main/gifs/env1%20env1.gif)  
trained on env1 test env2  
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/main/gifs/env1%20env2.gif)  