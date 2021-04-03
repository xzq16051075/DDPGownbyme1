import torch
import numpy as np
import torch.nn as nn
import random
from network import actor_network,critic_network
TAU = 0.1
LR = 0.001
GAMMA =0.9
TAU =0.01

EPISODES = 20
STEPS = 200
MEMORY_CAPACITY = 10000
MEMORY_SIZE = 32
class DDPG(object):
    def __init__(self,state_dim,action_dim,action_high,MODEL):
        super(DDPG, self).__init__()
        self.model = MODEL
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_high = action_high
        self.memory = np.zeros((MEMORY_CAPACITY,state_dim*2+action_dim+1),dtype=np.float)
        self.pointer = 0
        if self.model =='train':
            self.actor_eval = actor_network(self.s_dim, self.a_dim, self.a_high)
            self.actor_target = actor_network(self.s_dim, self.a_dim, self.a_high)
            self.critic_eval = critic_network(self.s_dim, self.a_dim)
            self.critic_target = critic_network(self.s_dim, self.a_dim)
        else:
            self.actor_eval = torch.load('actor_eval.pkl')
            self.actor_target = torch.load('actor_target.pkl')
            self.critic_eval = torch.load('critic_eval.pkl')
            self.critic_target = torch.load('critic_target.pkl')
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters() , lr = LR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr = LR)
        self.loss_func = nn.MSELoss()
    # 软更新
    def update_soft(self,target,source):
        for params_target,params in zip(target.parameters(),source.parameters()):
            params_target.data.copy_(
                params_target*(1-TAU) +params*TAU
            )
    # 硬更新
    # def update_hard(self,target,source):
    #     for params_target,params in zip(target.parameters(),source.parameters()):
    #         params_target.data.copy_(
    #             params
    #         )
    # 定义储存函数
    def store_transitions(self,s,a,r,s_):
        transition = np.hstack((s,a,[r],s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.pointer +=1

    def action_choose(self,state):
        state =torch.unsqueeze(torch.FloatTensor(state), 0)
        action = self.actor_eval(state)[0].detach()
        return action

    def learn(self):
        # 更新网络
        for params_target,params in zip(self.actor_target.parameters(),self.actor_eval.parameters()):
            params_target.data.copy_(
                params_target*(1-TAU) +params*TAU
            )
        for params_target,params in zip(self.critic_target.parameters(),self.critic_eval.parameters()):
            params_target.data.copy_(
                params_target*(1-TAU) +params*TAU
            )
        # 首先要批采样数据
        data_num = np.random.choice(MEMORY_CAPACITY,size = MEMORY_SIZE)
        batch_trans = self.memory[data_num,:]
        tran_s = torch.FloatTensor(batch_trans[:,:self.s_dim])
        tran_a = torch.FloatTensor(batch_trans[:,self.s_dim:self.s_dim+self.a_dim])
        tran_r = torch.FloatTensor(batch_trans[:,-self.s_dim-1:-self.s_dim])
        tran_s_ = torch.FloatTensor(batch_trans[:,-self.s_dim:])

        # 开始获得网络预测的参数
        # 更新actor参数
        a1 = self.actor_eval(tran_s)
        q1 = self.critic_eval(tran_s,a1)
        actor_loss = -torch.mean(q1)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新critic网络的参数
        a2 = self.actor_target(tran_s_)
        q2 = self.critic_target(tran_s_,a2)

        q_target  = tran_r + GAMMA * q2
        # 计算当前的价值
        q_eval = self.critic_eval(tran_s,tran_a)

        loss = self.loss_func(q_target,q_eval)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()





