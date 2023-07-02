# Global-Optimum-Search-using-RL
global optimum search using reinforcement learning on pytorch  

프로젝트 내용: 강화학습을 이용하여 3차원 공간에서 최저점(일종의 global optimum)을 찾는 에이전트를 학습시키기  
추후에 딥러닝의 경사하강법 대신 이 에이전트를 적용하여 학습시켜볼 예정  

내가 한 이상한 짓: 경사하강법과 다른 방법으로 해결하는 것을 보고 싶어서 state에 x, y, env(x, y)만 넣고
구현을 했는데 이렇게 하면 자신의 위치만 알고 주위 state를 알 방법이 없을 것이다(결국 학습이 안됨).  

이해가 안돼는 거: loss의 수치 해석, ddpg의 critic은, 결국 자신의 복제본인 target model의 q-value의 bellman equation을
loss의 지표로 사용하는데 메인 모델도 학습이 다 안됬는데 그걸 복제한 target model을 지표로 사용하는 이유를 모르겠음.  

Trained on X^2 + Y^2 / Test on X^2 + Y^2  

![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/main/gifs/env1env1.gif)

Trained on X^2 + Y^2 / Test on rastrigin  

![](https://github.com/kyle1213/Global-Optimum-Search-using-RL/blob/main/gifs/env1env2.gif)

해야할 것: 현재 x^2 + y^2에서 괜찮은 성능을 보이는 것을 확인했지만 rastrigin과 같은 복잡한 모델에서 성능이 안좋아보임, 또한 에이전트의 액션에 실수값이 아닌 (1, 1)이나 (-1, 1)처럼 부드럽지 않은 움직임만 보이는데 마음에 안드니까 개선해야 함
