#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import scipy.io as sio
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from PIL import Image


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # file_folder_name = "../save_data"
    # if not os.path.exists(file_folder_name):
    #     os.makedirs(file_folder_name)
    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done)
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # print(env.world.agents[0].state.p_pos)
    import time
    time.sleep(1)
    # create interactive policies for each agent
    # policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    i = 0
    index = 0
    count = 0
    done_n = [False]
    while not any(done_n):
        # query for action from each agent's policy
        # act_n = np.zeros([2,5])
        act_n = [np.random.randn(5),[0,1,0,1,0.5]]
        target=env.world.check[index].state.p_pos

        # ----------------键盘操作-----------------------------
        # act_n = []
        # for i, policy in enumerate(policies):
        #     act_n.append(policy.action(obs_n[i]))
        # ------------------end键盘操作-------------------------------

        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n,target)

        # # render all agent views
        image = env.render("rgb_array")  # 使用这种方法读取图片信息
        # img = Image.fromarray(image[1].astype('uint8')).convert('RGB')
        # print(img)
        # print(image)  
        # print(np.shape(image))  #第一个维度2表示有两个画面，这两个画面信息是一样的。 [2,height, width, channels (3=RGB)]
    env.close()