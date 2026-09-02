# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C


# =========================
# Improvements needed
# =========================
# change outputs to be a dict of all agents
# Figure out how to save trained model
# decide if I want agents to use global or individual actorcritics



import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import environement.pettingZooEnvironement
import matplotlib.pyplot as plt
from multiprocessing import Queue
import time
import numpy as np
import random
plt.ion() # probably unused

#==========================
# Set seed for debuging
#==========================
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)


# =========================
# Constants
# =========================
N_GAMES = 200 # number of rounds
T_MAX = 256 # number of steps before updating model
ENTROPY_SCALAR = .02 # scales entropy value
PRINT_ACTION = False # print every action taken
PRINT_REWARD = True # print rewards and end of round
LR = 1e-3
GAMMA = .99


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=LR, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    
    def reset_optimizer(optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                state['step'] = 0
                state['exp_avg'].zero_()
                state['exp_avg_sq'].zero_()
    
    def load_memory(self, mem):
        self.state = [mem]

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state["agent_0"].astype(np.float32)) # this will need large changes to handel multi agent
        self.actions.append(action["agent_0"])
        self.rewards.append(reward["agent_0"])

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def reset_weights(model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)

        pi = self.pi(x)
        v = self.v(x)


        return pi, v

    def calc_R(self, next_state, done):
        R = T.tensor(0.0)
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        return T.stack(batch_return)
        

    def calc_loss(self, next_state, done):
        # Convert stored observations into a batch
        states = T.tensor(np.array(self.states), dtype=T.float32)
        # states shape:
        # (batch_size, 2, 7, 7)
        states = states / 3
        actions = T.tensor(self.actions, dtype=T.int64)
        # Calculate discounted returns
        returns = self.calc_R(next_state, done)
        # Run all states through CNN
        pi, values = self.forward(states)
        values = values.squeeze()
        # Advantage
        advantage = returns - values
        # Critic loss
        critic_loss = advantage.pow(2).mean()
        # Normalize advantage for actor
        if advantage.numel() > 1:
            actor_advantage = (
                advantage - advantage.mean()
            ) / (
                advantage.std(unbiased=False) + 1e-8
            )
        else:
            actor_advantage = advantage

        # Actor
        probs = T.softmax(pi, dim=1)

        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy()

        actor_loss = (
            -log_probs * actor_advantage.detach()
        ).mean()

        # Total loss
        total_loss = (
            critic_loss
            + actor_loss
            - ENTROPY_SCALAR * entropy.mean()
        )
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor(
        observation["agent_0"],
        dtype=T.float32
        )
        # Add batch dimension
        # (2, 7, 7) -> (1, 2, 7, 7)
        state = state.unsqueeze(0)

        state = state / 3
        pi, v = self.forward(state)


        probs = T.softmax(pi, dim=1)

        dist = Categorical(probs)

        action = dist.sample().item()
        return action, probs

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id, data_queue=None, control_queue=None):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
        self.name = 'agent_0'
        self.episode_idx = global_ep_idx
        print("start process")
        self.env = environement.pettingZooEnvironement.env(render_mode="none")
        self.optimizer = optimizer
        self.data_queue = data_queue
        self.control_queue = control_queue
        self.running = False
        self.canceled = False
        self.simulation_delay = .2
        self.lr = LR
        self.gamma = GAMMA
        self.entropy = ENTROPY_SCALAR
        self.t_max = T_MAX
        self.data_queue.put({
            "type": "log",
            "agent": self.name,
            "level": "info",   # info, warning, error
            "message": "Agent initialized"
        })

    def run(self):
        self.data_queue.put({
            "type": "log",
            "agent": self.name,
            "level": "info",   # info, warning, error
            "message": "Started Agent"
        })
        t_step = 1
        while True:
            # --- PAUSE ---
            if not self.running:
                time.sleep(.05)
                self.process_control_queue()
                self.data_queue.put({
                    "type": "status",
                    "agent": self.name,
                    "status": "stoped"
                })
                continue
            observation, reward, terminated, truncated, info = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            terminated = False
            truncated = False
            done = False
            self.path=[]
            self.data_queue.put({
                "type": "status",
                "agent": self.name,
                "status": "running"
            })
            while not done:
                self.process_control_queue()
                if not self.running:
                    time.sleep(.05)
                if self.simulation_delay > 0:
                    time.sleep(self.simulation_delay)
                action, probs = self.local_actor_critic.choose_action(observation) # (for multi agent) change to for each agent
                actions = {"agent_0": action}
                if (PRINT_ACTION):
                    print (action, end=" ")
                observation_, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated["agent_0"] or truncated["agent_0"]
                score += reward["agent_0"]
                self.path.append(self.env.player_pos)
                self.data_queue.put({
                    "type": "step",
                    "agent": self.name,
                    "position": self.env.player_pos,
                    "grid": self.env.grid,
                    "action prob": probs.detach().cpu().tolist()
                })
                self.local_actor_critic.remember(observation, actions, reward)
                if t_step % T_MAX == 0 or done:
                    if not self.canceled:
                        loss = self.local_actor_critic.calc_loss(observation_, done)
                        self.local_actor_critic.zero_grad()
                        self.optimizer.zero_grad()
                        loss.backward()
                        T.nn.utils.clip_grad_norm_(self.local_actor_critic.parameters(),40.0)
                        for local_param, global_param in zip(
                                self.local_actor_critic.parameters(),
                                self.global_actor_critic.parameters()):
                            global_param._grad = local_param.grad
                        self.optimizer.step()
                        self.local_actor_critic.load_state_dict(
                                self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            if (PRINT_REWARD):
                if(PRINT_ACTION):
                    print() # formatting
                print(self.name, 'episode', self.episode_idx.value, 'reward %.1f' % score)
            self.data_queue.put({
                "type": "episode",
                "agent": self.name,
                "episode": self.episode_idx.value,
                "rewards": score,
                "path": self.path.copy(),
                #"loss": loss
            })

    def process_control_queue(self):
         # --- CONTROL QUEUE ---
        while not self.control_queue.empty():
            cmd = self.control_queue.get()
            
            # action control
            if cmd["type"] == "action":
                if cmd["action"] == "start":
                    self.running = True

                elif cmd["action"] == "stop":
                    self.running = False

                elif cmd["action"] == "reset":
                    self.running = False
                    self.canceled = True
                    self.episode_idx.value = 0
                    self.local_actor_critic.clear_memory()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    
                
            # speed control
            elif cmd["type"] == "speed":
                self.simulation_delay = cmd["value"]

            # paramater control
            elif cmd["type"] == "param":
                name = cmd["name"]
                value = cmd["value"]

                if name == "lr":
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = value

                elif name == "gamma":
                    self.local_actor_critic.gamma = value

                elif name == "entropy":
                    global ENTROPY_SCALAR
                    ENTROPY_SCALAR = value

                elif name == "t_max":
                    self.t_max = int(value)
