import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class actor_network(nn.Module):
    def __init__(self,state_dim,action_dim,action_high):
        super(actor_network, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_lim = action_high
        #初始化网络结构参数
        self.fc1 = nn.Linear(self.s_dim,128)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(128,32)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(32,self.a_dim)
        self.fc3.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        out = x*2
        return out

class critic_network(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(critic_network, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.fcs1 = nn.Linear(self.s_dim,64)
        self.fcs1.weight.data.normal_(0,0.1)
        self.fcs2 = nn.Linear(64,32)
        self.fcs2.weight.data.normal_(0,0.1)
        self.fca = nn.Linear(self.a_dim,32)
        self.fca.weight.data.normal_(0,0.1)
        self.fc1 =nn.Linear(64,16)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 =nn.Linear(16,1)
        self.fc2.weight.data.normal_(0,0.1)

    def forward(self,state,action):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca(action))
        output1 =F.relu(self.fc1(torch.cat((s2,a1),dim =1)))
        output = self.fc2(output1)
        return output



