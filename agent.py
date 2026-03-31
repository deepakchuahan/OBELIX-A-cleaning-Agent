import os;import numpy as np;
import torch;import torch.nn as nn;

ACTIONS=("L45","L22","FW","R22","R45");
_MODEL=None;hx=None;cx=None;

def _load_once():
    global _MODEL;global hx;global cx;
    if(_MODEL!=None):return;
    s_dr=os.path.dirname(__file__);
    
    w1=os.path.join(s_dr,"weights.pth");
    w2=os.path.join(s_dr,"drqn.pt");
    w3=os.path.join(s_dr,"weights.pt");
    
    if(os.path.exists(w1)):wpt=w1;
    else:
        if(os.path.exists(w2)):wpt=w2;
        else:wpt=w3;
        
    ins=18;outs=5;
    
    class DRQN(nn.Module):
        def __init__(self):
            super(DRQN,self).__init__();
            self.lstm=nn.LSTM(ins,64,batch_first=True);
            self.l1=nn.Linear(64,outs);
        def forward(self,x,h,c):
            v,(nh,nc)=self.lstm(x,(h,c));
            q=self.l1(v);
            return q,nh,nc;
            
    mdl=DRQN();
    mdl.load_state_dict(torch.load(wpt,map_location="cpu"));
    mdl.eval();_MODEL=mdl;
    hx=torch.zeros(1,1,64);cx=torch.zeros(1,1,64);

def policy(obs,rng):
    global hx;global cx;
    _load_once();
    
    x=torch.tensor(obs,dtype=torch.float32).view(1,1,18);
    with torch.no_grad():
        q,nhx,ncx=_MODEL(x,hx,cx);
        
    hx=nhx;cx=ncx;
    lgt=q.squeeze(0).squeeze(0).numpy();
    act=int(np.argmax(lgt));
    return ACTIONS[act];




###---------------------------------------for running  the agent------------------------
from obelix import OBELIX   # import your environment
import numpy as np

if __name__ == "__main__":
    env = OBELIX(scaling_factor=1)
    obs = env.reset()

    rng = np.random.default_rng()
    done = False

    while not done:
        action = policy(obs, rng)
        obs, reward, done = env.step(action)

        print(f"Step: {env.current_step}, Reward: {reward}")
