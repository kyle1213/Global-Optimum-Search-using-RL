import torch
import torch.nn.functional as F
import numpy as np
import copy
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class OU_noise:
    def __init__(self, mu, theta, sigma, action_size):
        self.reset()
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_size = action_size
        self.noise = None

    def reset(self):
        self.noise = np.ones((1, self.action_size), dtype=np.float32) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.noise) + self.sigma * np.random.randn(len(self.noise))
        self.noise += dx

        return self.noise


class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        return torch.tanh(self.mu(x))/10


class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size):
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
    def __init__(
            self, device, actor_lr, critic_lr, mem_maxlen, save_path, load_model, load_path,
            batch_size, state_size, action_size, mu, theta, sigma, discount_factor, tau
    ):
        self.actor = Actor(state_size=state_size, action_size=action_size).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_size=state_size, action_size=action_size).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.OU = OU_noise(mu=mu, theta=theta, sigma=sigma, action_size=action_size)
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path) if load_model == False else None
        self.device = device
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.save_path = save_path
        self.tau = tau

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

        action = (self.actor(torch.FloatTensor(state).to(self.device)).cpu().detach().numpy())

        return action + self.OU.sample() if training else action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), [state, action, reward, next_state, done])

        next_actions = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_actions)
        target_q = torch.unsqueeze(reward + (1-done) * self.discount_factor * torch.squeeze(next_q), dim=1)
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
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self):
        print(f" ... Save Model to {self.save_path}/ckpt ...")
        torch.save({"actor": self.actor.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic": self.critic.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(), },
                   self.save_path+'/model.ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)