import numpy as np
import gym
import envs.EnvDrone.classic_control.env_Drones2_sparse as search_grid
# import envs.EnvDrone.classic_control.env_Drones4_1channel as search_grid
class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self, map_set, map_num):
        self.env = search_grid.SearchGrid(map_set, map_num)
        
        self.agent_num = self.env.drone_num  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = self.env.observation_space.shape  # 设置智能体的观测纬度
        self.action_dim = self.env.action_space.n  # 设置智能体的动作纬度，这里假定为一个五个纬度的
        self.share_obs_dim = self.env.share_observation_space.shape
    
    def raise_difficulty(self):
        self.env.raise_difficulty()
        print("raise difficulty to ", self.env.run_time)
    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        sub_agent_obs = self.env.reset()
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, sub_agent_joint_map, sub_agent_rescue_masks, available_actions = self.env.step(actions)
        # sub_agent_obs = []
        # sub_agent_reward = []
        # sub_agent_done = []
        # sub_agent_info = []
        # for i in range(self.agent_num):
        #     sub_agent_obs.append(np.random.random(size=(14,)))
        #     sub_agent_reward.append([np.random.rand()])
        #     sub_agent_done.append(False)
        #     sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, sub_agent_joint_map, sub_agent_rescue_masks, available_actions]

