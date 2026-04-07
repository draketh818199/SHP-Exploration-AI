# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C


# =========================
# Improvements needed
# =========================
# change outputs to be a dict of all agents
# better data display
# Figure out how to save trained model
# decide if I want agents to use global or individual actorcritics
# get visual environment working (maybe another program the just runs saved training)
# streatch goal - make UI for running / viewing agent


import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import environement.pettingZooEnvironement
import matplotlib.pyplot as plt
from multiprocessing import Queue
import time
plt.ion() # probably unused

# =========================
# Constants
# =========================
N_GAMES = 200 # number of rounds
T_MAX = 50 # number of steps before updating model
ENTROPY_SCALAR = .01 # scales entropy value
PRINT_ACTION = False # print every action taken
PRINT_REWARD = True # print rewards and end of round
PLOT_REWARD = True


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
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

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state["agent_0"]) # this will need large changes to handel multi agent
        self.actions.append(action["agent_0"])
        self.rewards.append(reward["agent_0"])

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float32)
        states = states.view(states.shape[0], -1)

        
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        states = states.view(states.shape[0], -1)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2        

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        actor_loss = -log_probs*(returns-values) - ENTROPY_SCALAR * entropy

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor(observation["agent_0"], dtype = T.float32) # hanndels dict by selecting only current, handel multiple later
        state = state.view(1, -1) #flattens input array to be one dimension
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id, data_queue=None, control_queue=None):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'agent_0'
        self.episode_idx = global_ep_idx
        print("start process")
        self.env = environement.pettingZooEnvironement.env(render_mode="none")
        self.optimizer = optimizer
        self.data_queue = data_queue
        self.control_queue = control_queue
        self.running = False
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
            # --- CONTROL QUEUE ---
            while not self.control_queue.empty():
                cmd = self.control_queue.get()

                if cmd["action"] == "start":
                    self.running = True

                elif cmd["action"] == "stop":
                    self.running = False

                elif cmd["action"] == "reset":
                    self.running = False
                    self.episode_idx.value = 0
                    self.local_actor_critic.clear_memory()
            # --- PAUSE ---
            if not self.running:
                time.sleep(0.05)
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
                action = self.local_actor_critic.choose_action(observation) # (for multi agent) change to for each agent
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
                    "grid": self.env.grid
                })
                self.local_actor_critic.remember(observation, actions, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
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
                    print()
                print(self.name, 'episode', self.episode_idx.value, 'reward %.1f' % score)
            self.data_queue.put({
                "type": "episode",
                "agent": self.name,
                "episode": self.episode_idx.value,
                "rewards": score,
                "path": self.path.copy()
            })




#if __name__ == '__main__':
#    lr = 1e-4
#    env_id = 'CartPole-v0'
#    n_actions = 4
#    input_dims = [49]
#    data_queue = Queue()
#    control_queue = Queue()
#    global_actor_critic = ActorCritic(input_dims, n_actions)
#    global_actor_critic.share_memory()
#    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
#    global_ep = mp.Value('i', 0)
#    workers = [Agent(global_actor_critic,
#                    optim,
#                    input_dims,
#                    n_actions,
#                    gamma=0.99,
#                    lr=lr,
#                    name='agent_0',
#                    global_ep_idx=global_ep,
#                    env_id=env_id,
#                    data_queue=data_queue,
#                    control_queue=control_queue)] #for i in range(mp.cpu_count())]
#    [w.start() for w in workers]
#    [w.join() for w in workers]
