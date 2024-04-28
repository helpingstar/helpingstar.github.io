So the goal in today's lecture is going to be to combine some of the policy gradient ideas that we discussed before with some more recent concepts that we cover in the course, like policy iteration, to provide a new perspective on policy gradient methods and a little bit of analysis about when and why we would expect policy gradients to work.

따라서 오늘 강의의 목표는 이전에 논의했던 정책 그래디언트 아이디어 중 일부를 정책 반복과 같이 이 과정에서 다루는 최신 개념과 결합하여 정책 그래디언트 방법에 대한 새로운 관점을 제공하고 정책 그래디언트가 언제, 왜 작동할 것으로 기대하는지에 대한 약간의 분석을 제공하는 것입니다.

So if you have an interest in reinforcement learning theory, in understanding how and why to design algorithms in a certain way, and if you just want to get more in-depth knowledge about policy gradients, this is the lecture for you.

따라서 강화 학습 이론에 관심이 있고, 알고리즘을 특정 방식으로 설계하는 방법과 이유를 이해하고, 정책 그래디언트에 대해 더 깊이 있는 지식을 얻고자 한다면 이 강의가 적합합니다.

And then we saw, for instance, in the Actor-Critic lecture that this reward to go could be computed in various ways, using the Monte Carlo estimator as shown here or with more sophisticated methods that involve actually learning value function estimators.

예를 들어, 액터-크리틱 강의에서 이 reward to go는 여기에 표시된 것처럼 몬테카를로 추정기를 사용하거나 실제로 가치 함수 추정기를 학습하는 더 정교한 방법으로 다양한 방식으로 계산할 수 있다는 것을 알 수 있었습니다.

They generate samples in the orange box, fit some estimate to the reward to go, either with a learned value function or just with Monte Carlo estimates in the green box, and then perform gradient ascent in the blue box.

They generate samples in the orange box, fit some estimate to the reward to go, either with a learned value function or just with Monte Carlo estimates in the green box, and then perform gradient ascent in the blue box.

