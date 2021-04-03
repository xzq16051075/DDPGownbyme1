import gym
import numpy as np
from agent import DDPG
import time
import torch
from torch.utils.tensorboard import SummaryWriter

#定义全局变量
TAU = 0.1
LR = 0.001
GAMMA =0.9
TAU =0.01
ENV_NAME = 'Pendulum-v0'
EPISODES = 20
STEPS = 200
MEMORY_CAPACITY = 10000
MEMORY_SIZE = 32
MODEL = 'test'

writer = SummaryWriter('runs/fashion_mnist_experiment_1')
def main():
    RENDER = False
    env = gym.make(ENV_NAME)
    env.unwrapped
    env.seed(1)
    #获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low = env.action_space.low
    ddpg = DDPG(state_dim,action_dim,action_high,MODEL)
    var = 3
    for episode in range(EPISODES):
        ep_r = 0
        state = env.reset()

        for step in range(STEPS):
            if RENDER:
                env.render()
            action = ddpg.action_choose(state)
            action = np.clip(np.random.normal(action,var),action_low,action_high)

            state_,reward,done,info = env.step(action)

            ddpg.store_transitions(state,action,reward/10,state_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= 0.9995
                ddpg.learn()

            state = state_
            ep_r += reward

            if step == STEPS-1:
                print('Episode:',episode,'Average reward:',ep_r,"explore:",var)
                if ep_r> -300:
                    RENDER = True
                break
    if MODEL == 'train':
        torch.save(ddpg.actor_eval, 'actor_eval.pkl')
        torch.save(ddpg.actor_target, 'actor_target.pkl')
        torch.save(ddpg.critic_eval, 'critic_eval.pkl')
        torch.save(ddpg.critic_target,'critic_target.pkl')
    # writer.add_graph(ddpg.actor_eval,state)
    # writer.close()
    env.close()
if __name__ == '__main__':
    t1 = time.time()
    main()
    print('total train time:',time.time()-t1)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
