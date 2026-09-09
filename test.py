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


advantage = T.tensor([1.7647865,   0.71734047,  0.70479167, -0.04157826, -0.53242177, -1.0083852,
 -0.4582486,  -1.119284,   -1.4105515,  -1.0583228,  -1.5514377,  -1.0511386,
 -1.4085492,  -0.527501,   -0.51380455, -0.4244735,  -1.1925439,  -0.25859705,
 -0.21090786, -0.12058788, -0.02712658,  0.13177961,  0.16003692,  0.2602111,
 -0.28749505,  0.43906853,  0.53987783,  0.47767106,  0.73688865,  1.1903485,
  2.1930966,   1.979677,    1.9073833 ], dtype=T.float32)
critic_loss = advantage.pow(2).mean()
print(critic_loss)
