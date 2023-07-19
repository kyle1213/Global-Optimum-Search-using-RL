# Global-Optimum-Search-using-RL
global optimum search using reinforcement learning on pytorch  

프로젝트 내용:
+ 강화학습을 이용하여 3차원 공간에서 최저점(일종의 global optimum)을 찾는 에이전트를 학습시키기  
+ 추후에 딥러닝의 경사하강법 대신 이 에이전트를 적용하여 학습시켜볼 예정  

기능:
+ RL모델 학습
+ train마다 trajectory를 gif로 기록(train 속도에 좋지 않지만, 시각적으로 보고 싶었음)
+ train 후 loss, score기록, test시 model load를 하고 결과 trajectory를 gif로 기록

구현 방법:
+ RL model: DDPG(actor-critic)
+ reward: dz * abs(dz), -1 each step,
+ actions: (x, y)
+ state: (x, y, z, dx, dy)
+ env: env0=x+y / env1=x^2+y^2 / env2=rastrigin

결과 분석:
+ env1에서 테스트할 때, global optimum 근처에서 동일한 action을 반복함(no exploration if it is local optimum, need early stop if global optimum)
+ env1에서만 학습해서 그런지 env0에서 env1의 global optimum좌표를 향하는 모습이 보인다
+ env2에서 test할 때는 local optimum에 빠지는 것을 보니, agent가 단순히 coordinate으로만 행동하지는 않고, slope도 활용하기는 하는 것 같다.

TODO:
+ 신경망으로 에이전트가 만족하면 탐색 중단하는 기능 추가해보기(early stop)
+ 다른 방법론 사용(PPO 등)
+ local optimum problem 해결
+ add agent memory of episode for more state info
+ train 단계에서 동일 env가 아닌 다양한 env에서 학습하게 하기(env memorize를 예방하기 위해)

내가 한 이상한 짓:
+ 경사하강법과 다른 방법으로 해결하는 것을 보고 싶어서 state에 x, y, env(x, y)만 넣고
구현을 했는데 이렇게 하면 자신의 위치만 알고 주위 state를 알 방법이 없을 것이다(결국 학습이 안됨, 환경 model을 학습하는 모델이라면 사용 가능할지도).  

이해가 안돼는 거:
+ loss의 수치 해석, ddpg의 critic은, 결국 자신의 복제본인 target model의 q-value의 bellman equation을
loss의 지표로 사용하는데 메인 모델도 학습이 다 안됬는데 그걸 복제한 target model을 지표로 사용하는 이유를 모르겠음.  

# Train results

trained on env1 test env0

trained on env1 test env1

trained on env1 test env2