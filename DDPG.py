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

config = {
    'OU_noise_mu': 0,
    'OU_noise_theta': 1e-3,
    'OU_noise_sigma': 2e-3,

}

max_episode_steps = 5000
batch_size = 128
mem_maxlen = 25000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 1e-5
tau = 1e-3

run_step = 3000000
train_start_step = 20000

state_size = 5
action_size = 2

load_model = False  # True for test, False for train
load_param = False  # for continous learning
train_mode = True if not load_model else False
test_step = max_episode_steps
print_interval = 10
save_interval = 100

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{date_time}"
load_path = f"./saved_models/main_env1"

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


if __name__ == "__main__":
    env = env1
    Z = Z1

    agent = DDPGAgent()

    actor_losses_per_run, critic_losses_per_run = [], []
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    iterations = []

    if train_mode:
        # initialize
        actions = []
        reward = 0
        done = 0
        x_init = random.uniform(-5, 5)
        y_init = random.uniform(-5, 5)
        states = [x_init, y_init, env(x_init, y_init), float(sy.diff(Z, x).evalf(subs={x: x_init, y: y_init})),
                  float(sy.diff(Z, y).evalf(subs={x: x_init, y: y_init}))]
        # states = [x,y,z, dx, dy]
        count_step = 0

        for step in range(run_step):
            if done == 0:  # not finished
                count_step += 1
                actions = agent.get_action(states, train_mode)

                old_states = copy.deepcopy(states)

                states[0] += actions[0][0]
                states[1] += actions[0][1]

                states[2] = env(states[0], states[1])
                states[3] = float(sy.diff(Z, x).evalf(subs={x: states[0], y: states[1]}))
                states[4] = float(sy.diff(Z, y).evalf(subs={x: states[0], y: states[1]}))

                reward = (env(old_states[0], old_states[1]) - env(states[0], states[1])) * abs(env(old_states[0], old_states[1]) - env(states[0], states[1]))
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
                x_init = random.uniform(-5, 5)
                y_init = random.uniform(-5, 5)
                states = [x_init, y_init, env(x_init, y_init), float(sy.diff(Z, x).evalf(subs={x: x_init, y: y_init})),
                               float(sy.diff(Z, y).evalf(subs={x: x_init, y: y_init}))]
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
                X = np.linspace(-5.12, 5.12, 100)
                Y = np.linspace(-5.12, 5.12, 100)
                X, Y = np.meshgrid(X, Y)

                fig = plt.figure()

                plt.contour(X, Y, env(X, Y), levels=15)
                cntr = plt.contourf(X, Y, env(X, Y), levels=15, cmap="RdBu_r")
                plt.colorbar(cntr)
                d, = plt.plot([], [], 'C0o')
                dx = []
                dy = []


                def animate(i):
                    dx.append(agent.memory[i+len(agent.memory)-5000][0][0])
                    dy.append(agent.memory[i+len(agent.memory)-5000][0][1])
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
        x_init = random.uniform(-5, 5)
        y_init = random.uniform(-5, 5)
        states = [x_init, y_init, env(x_init, y_init), float(sy.diff(Z, x).evalf(subs={x: x_init, y: y_init})),
                  float(sy.diff(Z, y).evalf(subs={x: x_init, y: y_init}))]

        for step in range(test_step):
            if done == 0:  # not finished
                actions = agent.get_action(states, train_mode)

                old_states = copy.deepcopy(states)

                states[0] += actions[0]
                states[1] += actions[1]

                states[2] = env(states[0], states[1])
                states[3] = float(sy.diff(Z, x).evalf(subs={x: states[0], y: states[1]}))
                states[4] = float(sy.diff(Z, y).evalf(subs={x: states[0], y: states[1]}))

                reward = (env(old_states[0], old_states[1]) - env(states[0], states[1])) * abs(env(old_states[0], old_states[1]) - env(states[0], states[1]))
                reward -= 1

                ## if 모델이 충분히 최저점에 왔다고 판별을 하면 그만하기, +신경망으로 판별네트워크도 만들어야함

                if step >= max_episode_steps - 1:
                    done = 1
                agent.append_sample(old_states, copy.deepcopy(actions), reward, copy.deepcopy(states), done)

            score += reward

            if done == 1:
                X = np.linspace(-5.12, 5.12, 100)
                Y = np.linspace(-5.12, 5.12, 100)
                X, Y = np.meshgrid(X, Y)

                fig = plt.figure()

                plt.contour(X, Y, env(X, Y), levels=15)
                cntr = plt.contourf(X, Y, env(X, Y), levels=15, cmap="RdBu_r")
                plt.colorbar(cntr)

                #  mcolors.CSS4_COLORS[list(mcolors.CSS4_COLORS.keys())[i]]
                #  'C'+str(i)+'o'
                d, = plt.plot([], [], 'C0o')
                dx = []
                dy = []


                def animate(i):
                    dx.append(agent.memory[i][0][0])
                    dy.append(agent.memory[i][0][1])
                    d.set_data(dx[i], dy[i])

                    return d,

                anim = animation.FuncAnimation(fig, animate, frames=test_step, interval=100)

                writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

                anim.save('./gifs/record.gif', writer=writer)

                plt.show()


# ##라인 지우기

# reward, state, env 등 환경과 상호작용과 관련된 부분을 개선해야함