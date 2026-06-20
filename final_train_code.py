import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

ip = "/kaggle/input"
for r, d, f in os.walk(ip):
    if "obelix.py" in f: sys.path.append(r)
try:
    from obelix import OBELIX
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WrapperEnv:
    def __init__(self, env):
        self.e = env

    def reset(self):
        return self.e.reset()

    def step(self, action, rd=False):
        n, r, dn = self.e.step(action, render=rd)
        
        if r <= -50: rs = -10
        elif r >= 1000: rs = 10
        elif r >= 100: rs = 5
        elif r > 1: rs = 1
        elif r < -1: rs = -1
        else: rs = r
        
        return n, rs, dn

class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.f(x)

def train():
    env = OBELIX(scaling_factor=1, arena_size=500, max_steps=1000, difficulty=3)
    we = WrapperEnv(env)
    ACTIONS = ("L45", "L22", "FW", "R22", "R45")
    
    actor = ActorNetwork().to(device)
    
    episodes = 500
    batch_size = 64
    noise_var = 0.1
    
    for ep in range(episodes):
        o = we.reset()
        dn = False
        tr = 0
        
        fs = [o, o, o, o]
        so = np.concatenate(fs)
        
        while not dn:
            with torch.no_grad():
                st = torch.FloatTensor(so).unsqueeze(0).to(device)
                a_c = actor(st).item()
            
            a_c = a_c + np.random.normal(0, noise_var)
            a_c = np.clip(a_c, -1.0, 1.0)
            
            if a_c <= -0.6: ai = 0
            elif a_c <= -0.2: ai = 1
            elif a_c <= 0.2: ai = 2
            elif a_c <= 0.6: ai = 3
            else: ai = 4
            
            an = ACTIONS[ai]
            n, r, dn = we.step(an, rd=False)
            
            fs.pop(0)
            fs.append(n)
            sn = np.concatenate(fs)
            
            so = sn
            tr += r

if __name__ == "__main__":
    pass