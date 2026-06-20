import os;import numpy as np;
ACTIONS=("L45","L22","FW","R22","R45");_MODEL=None;_STK=None;
def _load_once():
 global _MODEL,_STK;
 if(_MODEL is not None):return;
 sd=os.path.dirname(__file__);wp=os.path.join(sd,"weights.pth");
 import torch;import torch.nn as nn;
 class Net(nn.Module):
  def __init__(self):
   super().__init__();self.f=nn.Sequential(nn.Linear(72,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh());
  def forward(self,x):return self.f(x);
 m=Net();
 s=torch.load(wp,map_location="cpu",weights_only=False);
 if(type(s)==dict):
  if("w_a" in s):m.load_state_dict(s["w_a"]);
  else:m.load_state_dict(s);
 else:m.load_state_dict(s);
 m.eval();_MODEL=m;_STK=[];
def policy(obs:np.ndarray,rng:np.random.Generator)->str:
 _load_once();import torch;global _STK;
 if(len(_STK)==0):
  for i in range(4):_STK.append(obs);
 else:
  _STK.pop(0);_STK.append(obs);
 so=np.concatenate(_STK);x=torch.from_numpy(so.astype(np.float32)).unsqueeze(0);
 with torch.no_grad():a_c=_MODEL(x).item();
 if(a_c<=-0.6):ai=0;
 else:
  if(a_c<=-0.2):ai=1;
  else:
   if(a_c<=0.2):ai=2;
   else:
    if(a_c<=0.6):ai=3;
    else:ai=4;
 return ACTIONS[ai];