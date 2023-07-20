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

max_episode_steps = 500
batch_size = 128
mem_maxlen = 25000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 1e-5
tau = 1e-3

run_step = 200000
train_start_step = 5000

state_size = 3
action_size = 1

mu = 0
theta = 1e-3
sigma = 2e-3

load_model = False  # True for test, False for train
load_param = False  # for continous learning
train_mode = True if not load_model else False
test_step = max_episode_steps
save_interval = 100

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{date_time}"
load_path = f"./saved_models/20230720213514"

x = sy.symbols('x')

Y0 = (x)**2


def env0(x):  # environment
    return (x)**2


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

        return torch.tanh(self.mu(x))


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat((state, torch.unsqueeze(torch.squeeze(action), dim=1)), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.q(x)


class DDPGAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path) if load_model == False else None

        if load_model:
            print(f"... Load Model from {load_path}/model.ckpt ...")
            checkpoint = torch.load(load_path+"/model.ckpt", map_location=device)
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

    def save_model(self):
        print(f" ... Save Model to {save_path}/ckpt ...")
        torch.save({"actor": self.actor.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(), "critic": self.critic.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),}, save_path+'/model.ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)


if __name__ == "__main__":
    env = env0
    Y = Y0

    agent = DDPGAgent()

    actor_losses_per_run, critic_losses_per_run = [], []
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    iterations = []

    if train_mode:
        # initialize
        actions = []
        reward = 0
        done = 0
        x_init = random.uniform(-100, 100)
        states = [x_init, env(x_init), float(sy.diff(Y, x).evalf(subs={x: x_init}))]
        # states = [x,y, dx]
        count_step = 0

        for step in range(run_step):
            if done == 0:  # not finished
                count_step += 1
                actions = agent.get_action(states, train_mode)

                old_states = copy.deepcopy(states)

                states[0] += actions[0][0]

                states[1] = env(states[0])
                states[2] = float(sy.diff(Y, x).evalf(subs={x: states[0]}))

                reward = env(old_states[0]) - env(states[0])
                reward -= 1

                ## if 모델이 충분히 최저점에 왔다고 판별을 하면 그만하기, +신경망으로 판별네트워크도 만들어야함

                if count_step >= max_episode_steps:
                    done = 1
                agent.append_sample(old_states, copy.deepcopy(actions), reward, copy.deepcopy(states), done)

                if step > train_start_step:
                    actor_loss, critic_loss = agent.train_model()
                    actor_losses_per_run.append(actor_loss)
                    critic_losses_per_run.append(critic_loss)

                    agent.soft_update_target()

            score += reward

            if done == 1:
                actions = []
                reward = 0
                done = 0
                x_init = random.uniform(-100, 100)
                states = [x_init, env(x_init), float(sy.diff(Y, x).evalf(subs={x: x_init}))]
                count_step = 0

                episode += 1
                scores.append(score)

                mean_actor_loss = np.mean(actor_losses_per_run)
                mean_critic_loss = np.mean(critic_losses_per_run)
                agent.write_summary(score, mean_actor_loss, mean_critic_loss, step)

                actor_losses.append(mean_actor_loss)
                critic_losses.append(mean_critic_loss)

                print(f"{episode} Episode / Step: {step} / Score: {score:.2f} / " + f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

                if train_mode and episode % save_interval == 0:
                    agent.save_model()

                iterations.append(step)
                score = 0

                # save train step gifs
                X_range = np.linspace(-100, 100, 2000)

                Y_values = env(X_range)

                fig = plt.figure()

                plt.plot(X_range, Y_values)
                plt.xlabel('X')
                plt.ylabel('Y')

                d, = plt.plot([], [], 'C0o')
                dx = deque(maxlen=5)
                dy = deque(maxlen=5)


                def animate(i):
                    dx.append(agent.memory[i+len(agent.memory)-500][0][0])
                    dy.append(agent.memory[i+len(agent.memory)-500][0][1])
                    d.set_data(dx, dy)

                    return d,

                anim = animation.FuncAnimation(fig, animate, frames=test_step, interval=100)

                writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

                anim.save('./gifs/'+str(episode)+'.gif', writer=writer)

                plt.close()

        plt.subplot(121)
        plt.plot(range(1, len(iterations) + 1), actor_losses, 'b--')
        plt.plot(range(1, len(iterations) + 1), critic_losses, 'r--')
        plt.subplot(122)
        plt.plot(range(1, len(iterations) + 1), scores, 'g-')
        plt.title('actor critic losses and scores')
        plt.savefig('train result/result.png')

    else:  # test mode
        # initialize
        actions = []
        reward = 0
        done = 0
        x_init = random.uniform(-100, 100)
        states = [x_init, env(x_init), float(sy.diff(Y, x).evalf(subs={x: x_init}))]

        for step in range(test_step):
            if done == 0:  # not finished
                actions = agent.get_action(states, train_mode)

                old_states = copy.deepcopy(states)

                states[0] += actions[0]

                states[1] = env(states[0])
                states[2] = float(sy.diff(Y, x).evalf(subs={x: states[0]}))

                reward = env(old_states[0]) - env(states[0])
                reward -= 1

                ## if 모델이 충분히 최저점에 왔다고 판별을 하면 그만하기, +신경망으로 판별네트워크도 만들어야함

                if step >= max_episode_steps - 1:
                    done = 1
                agent.append_sample(old_states, copy.deepcopy(actions), reward, copy.deepcopy(states), done)

            score += reward

            if done == 1:
                X_range = np.linspace(-100, 100, 2000)

                Y_values = env(X_range)

                fig = plt.figure()

                plt.plot(X_range, Y_values)
                plt.xlabel('X')
                plt.ylabel('Y')

                d, = plt.plot([], [], 'C0o')
                dx = deque(maxlen=5)
                dy = deque(maxlen=5)


                def animate(i):
                    dx.append(agent.memory[i + len(agent.memory) - 500][0][0])
                    dy.append(agent.memory[i + len(agent.memory) - 500][0][1])
                    d.set_data(dx, dy)

                    return d,


                anim = animation.FuncAnimation(fig, animate, frames=test_step, interval=100)

                writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

                anim.save('./gifs/record.gif', writer=writer)

                plt.close()

# ##라인 지우기

# reward, state, env 등 환경과 상호작용과 관련된 부분을 개선해야함