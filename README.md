# Global-Optimum-Search-using-RL
global optimum search using reinforcement learning on pytorch  

기능: RL모델로 학습, train마다 trajectory를 gif로 기록(속도에 좋지 않지만, 시각적으로 보고 싶었음), train 후 loss, score기록, test시
model load를 하고 결과 trajectory를 gif로 기록

구현 방법: 간단한 actor critic 모델로 단순한 3차 공간에서 학습을 진행하였다. reward함수는 dz*abs(dz)를 사용하였는데, 단순히 
함수의 값의 변화량으로 reward를 주지 않고 변화량의 절대값을 한번 더 곱하여 더 큰 변화량에 더 큰 reward가 발생하도록 해보았다. 또한 매
스텝마다 -0.1의 reward를 부여함. 또한 actor의 action에 early stop을 위한 노드를 하나 더 만들어 학습을 해보았다.

결과 분석: 과거 정보 없이 state = (x, y, z, dx, dy)로 early stop을 학습하는 것은 어렵다는 직관이 있는 것 처럼 학습이 제대로 이루어지지
않았다(env0에서 test시 기울기를 따라가지 않고 env1의 optimum근처인 (0,0)을 향해 가는 것을 볼 수 있다. 근데 신기한 점은 env2에서는 그렇지 않고 
기울기를 따라가려고 하는듯 한 모습을 보인다). state memory를 state에 추가하여 학습해보아야 할 것 같다.


# Train results

graph  
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/early_stop_1.0/train%20result/env1.png)

env1에서 학습 중 early stop이 발생한 경우
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/early_stop_1.0/gifs/8540.gif)  
env1에서 학습 중 early stop이 발생하지 않은 경우
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/early_stop_1.0/gifs/8756.gif)  

Trained on env1 / Test on env0
![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/early_stop_1.0/gifs/env1%20env0.gif)  

Trained on env1 / Test on env1

![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/early_stop_1.0/gifs/env1%20env1.gif)  
Trained on env1 / Test on env2

![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/early_stop_1.0/gifs/env1%20env2.gif)

