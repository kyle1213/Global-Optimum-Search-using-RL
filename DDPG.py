import torch
import numpy as np
import copy
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import math
import matplotlib.colors as mcolors
import random
import sympy as sy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent_num = 1

max_episode_steps = 500
batch_size = 128
mem_maxlen = 25000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 1e-5
tau = 1e-3

run_step = 300000
train_start_step = 5000

state_size = 5
action_size = 2

mu = 0
theta = 1e-3
sigma = 2e-3

load_model = True # True for test, False for train
train_mode = True if not load_model else False
test_step = 500
print_interval = 10
save_interval = 100

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{date_time}"
load_path = f"./saved_models/20230626191557"

x, y = sy.symbols('x y')

Z0 = x + y
Z1 = x ** 2 + y ** 2
Z2 = (x ** 2 - 10 * sy.cos(2 * sy.pi * x)) + (y ** 2 - 10 * sy.cos(2 * sy.pi * y)) + 20


def env0(x, y):  # environment
    return x + y


def env1(x, y):  # environment
    return x ** 2 + y ** 2


def env2(x, y):
    return (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y)) + 20


class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones((1, action_size), dtype=np.float32) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx

        return self.X


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        return torch.tanh(self.mu(x))/10


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat((state, torch.squeeze(action)), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.q(x)


class DDPGAgent():
    def __init__(self, i):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path) if load_model == False else None

        if load_model == True:
            print(f"... Load Model from {load_path}/"+str(i)+".ckpt ...")
            checkpoint = torch.load(load_path+"/"+str(i)+".ckpt", map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.target_actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def get_action(self, state, training=True):
        self.actor.train(training)

        action = (self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy())

        return action + self.OU.sample() if training else action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), [state, action, reward, next_state, done])

        next_actions = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_actions)
        target_q = torch.unsqueeze(reward + (1-done) * discount_factor * torch.squeeze(next_q), dim=1)
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, i):
        print(f" ... Save  "+str(i)+"th Model to {save_path}/ckpt ...")
        torch.save({"actor": self.actor.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(), "critic": self.critic.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),}, save_path+'/'+str(i)+'.ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)


if __name__ == "__main__":
    env = env1
    Z = Z1

    agents = [DDPGAgent(i) for i in range(agent_num)]

    actor_losses, critic_losses, scores, episode, score = [[] for _ in range(agent_num)], [[] for _ in range(
        agent_num)], [], 0, 0

    if train_mode:
        # initialize
        actions = [[] for i in range(agent_num)]
        reward = [0 for i in range(agent_num)]
        done = [0 for i in range(agent_num)]
        states = []
        for i in range(agent_num): #initialize agent's actions
            x_init = random.uniform(-5, 5)
            y_init = random.uniform(-5, 5)
            states.append([x_init, y_init, env(x_init, y_init), float(sy.diff(Z, x).evalf(subs={x: x_init, y: y_init})),
                           float(sy.diff(Z, y).evalf(subs={x: x_init, y: y_init}))])
            # states = [x,y,z, old_x, old_y, old_z]
        count_step = [0 for i in range(agent_num)]

        for step in range(run_step):
            for i in range(agent_num):
                if done[i] == 0:  # not finished
                    count_step[i] += 1
                    actions[i] = agents[i].get_action(states[i], train_mode)

                    old_states = copy.deepcopy(states)

                    states[i][0] += actions[i][0][0]
                    states[i][1] += actions[i][0][1]

                    states[i][2] = env(states[i][0], states[i][1])
                    states[i][3] = float(sy.diff(Z, x).evalf(subs={x: states[i][0], y: states[i][1]}))
                    states[i][4] = float(sy.diff(Z, y).evalf(subs={x: states[i][0], y: states[i][1]}))

                    reward[i] = env(old_states[i][0], old_states[i][1]) - env(states[i][0], states[i][1])

                    if env(states[i][0], states[i][1]) == 0:
                        reward[i] = 10000
                        done[i] = 1

                    if count_step[i] >= max_episode_steps:
                        done[i] = 1
                    agents[i].append_sample(old_states[i], copy.deepcopy(actions[i]), reward[i], copy.deepcopy(states[i]), done[i])

                    if step > train_start_step:
                        actor_loss, critic_loss = agents[i].train_model()
                        actor_losses[i].append(actor_loss)
                        critic_losses[i].append(critic_loss)

                        agents[i].soft_update_target()

            score += np.mean(reward) #score is sum of mean of all agent's reward

            if done == [1 for i in range(agent_num)]:
                actions = [[] for i in range(agent_num)]
                reward = [0 for i in range(agent_num)]
                done = [0 for i in range(agent_num)]
                states = []
                for i in range(agent_num):
                    x_init = random.uniform(-5, 5)
                    y_init = random.uniform(-5, 5)
                    states.append([x_init, y_init, env(x_init, y_init), float(sy.diff(Z, x).evalf(subs={x: x_init, y: y_init})),
                         float(sy.diff(Z, y).evalf(subs={x: x_init, y: y_init}))])
                count_step = [0 for i in range(agent_num)]

                episode += 1
                scores.append(score)
                score = 0

                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_actor_loss = np.mean(actor_losses)
                    mean_critic_loss = np.mean(critic_losses)
                    agents[0].write_summary(mean_score, np.mean(mean_actor_loss), np.mean(mean_critic_loss), step)


                    print(
                        f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " + f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")
                    actor_losses, critic_losses, scores = [[] for _ in range(agent_num)], [[] for _ in
                                                                                           range(agent_num)], []
                if train_mode and episode % save_interval == 0:
                    for i, agent in enumerate(agents):
                        agent.save_model(i)

    else:  # test mode
        actions = [[] for i in range(agent_num)]
        reward = [0 for i in range(agent_num)]
        done = [0 for i in range(agent_num)]
        states = []
        for i in range(agent_num):
            x_init = random.uniform(-5, 5)
            y_init = random.uniform(-5, 5)
            states.append([x_init, y_init, env(x_init, y_init), float(sy.diff(Z, x).evalf(subs={x: x_init, y: y_init})),
                           float(sy.diff(Z, y).evalf(subs={x: x_init, y: y_init}))])
        count_step = [0 for i in range(agent_num)]

        for step in range(test_step):
            for i in range(agent_num):
                if done[i] == 0:  # not finished
                    count_step[i] += 1
                    actions[i] = agents[i].get_action(states[i], train_mode)

                    old_states = copy.deepcopy(states)

                    states[i][0] += actions[i][0]
                    states[i][1] += actions[i][1]

                    states[i][2] = env(states[i][0], states[i][1])
                    states[i][3] = float(sy.diff(Z, x).evalf(subs={x: states[i][0], y: states[i][1]}))
                    states[i][4] = float(sy.diff(Z, y).evalf(subs={x: states[i][0], y: states[i][1]}))

                    reward[i] = reward[i] = env(old_states[i][0], old_states[i][1]) - env(states[i][0], states[i][1])

                    if count_step[i] >= test_step:
                        done[i] = 1
                    agents[i].append_sample(old_states[i], actions[i], reward[i], states[i], done[i])

            score += np.mean(reward)
        if done == [1 for i in range(agent_num)]:
            X = np.linspace(-5.12, 5.12, 100)
            Y = np.linspace(-5.12, 5.12, 100)
            X, Y = np.meshgrid(X, Y)

            fig = plt.figure()

            plt.contour(X, Y, env(X, Y), levels=15)
            cntr = plt.contourf(X, Y, env(X, Y), levels=15, cmap="RdBu_r")
            plt.colorbar(cntr)

            #  mcolors.CSS4_COLORS[list(mcolors.CSS4_COLORS.keys())[i]]
            #  'C'+str(i)+'o'
            d = [plt.plot([], [], 'C' + str(i) + 'o') for i in range(agent_num)]
            dx = []
            dy = []

            def animate(i):
                dx.append([agents[j].memory[i][0][0] for j in range(agent_num)])
                dy.append([agents[j].memory[i][0][1] for j in range(agent_num)])
                for j, d_ in enumerate(d):
                    d_[0].set_data(dx[i][j], dy[i][j])
                return d


            anim = animation.FuncAnimation(fig, animate, frames=test_step, interval=100)

            writer = animation.PillowWriter(fps=15,
                                             metadata=dict(artist='Me'),
                                             bitrate=1800)
            anim.save('./gifs/record.gif', writer=writer)
            plt.show()
# ##라인 지우기
# +- 5.12 넘으면 페널티 주기
# reward 개선
# z값 낮아지면 +1, 0되면 100, 올라가면 -1, 밖에 나가려하면 -5
#한명이라도 0이되면 끝낼지, 끝까지 할지 정하기
#서로 통신하지 않는 상황 (NL, naive learning)
# 자꾸 바깥으로 나가려는 문제 해결하기
# 쉬운 env로 테스트하기
# single ddpg부터 구현하기

#reward, state, env 등 환경과 상호작용과 관련된 부분을 개선해야함