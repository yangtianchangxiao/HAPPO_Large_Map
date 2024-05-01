# 在这个环境里, 智能体一开始不会面临一条路是cutting road 的局面,也就是说, 一开始的几条路都可以遍历全图
# 不会出现一开始的必经之路就只有一条

from typing import Optional
import gym
import heapq
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
import random
import copy
from typing import List, Tuple
from itertools import product
from collections import deque
import os
import pickle
import sys
# from numba import njit, vectorize

sys.path.append(r"/home/cx/happo/envs/EnvDrone/classic_control/")
sys.path.append(r"/home/cx/envs/EnvDrone/classic_control/")
sys.path.append(r"/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/")



from rescue_function import rescue_action
from rescue_function import generate_path
from PIL import Image



# from gym.envs.classic_control import utils


class Drones(object):
    def __init__(self, pos, view_range, local_map_radius, id, map_size):
        self.id = id
        self.pos = pos
        self.last_pos = [None, None]
        self.view_range = view_range
        self.area = None
        self.communicate_list = [] # 记录可以通信的名单
        self.relative_pos = []
        self.relative_direction = []
        self.relatvie_coordinate = []
        self.individual_observed_zone = []
        self.observed_obs = []
        self.observed_drone = []
        self.individual_observed_obs = None
        self.unobserved = []
        self.communicate_rate = 0  # 添加了机器人通信频率奖励，使机器人在扩散探索的同时也注意信息的共享
        self.whole_map = np.zeros((4, 120, 120), dtype=np.float16)  # 每个机器人保存一个本地地图
        self.whole_map[1,pos[0],pos[1]] = 1 
        self.map_processed = 0.5 * np.ones((1, 120, 120), dtype=np.float16) # 将多个不同特征的地图融合到一张上
        self.map_condensed = np.zeros((1, 60, 60))
        
        self.last_whole_map = None
        self.grid_communication = 0
        self.obstacle_communication = 0
        self.last_obstacle_communication = 0
        self.coord_per_obs = np.empty((4*view_range**2, 2)) # 记录每次每个agent探索的空白区域的坐标，用于后续惩罚agents在一个step中，过多区域重合的现象
        # 该变量表示：智能体做出的action，带来的whole_map[1]这个已探索区域的时间戳的增加量
        self.open_information_gain = 0
        # 周围环境的空旷程度，可以用来表示避免碰撞的难度
        self.open_degree = 0
        # self.coord_per_obs_length = 0 # 记录agent一次探索的空白区域的坐标有几个
        self.repetition_flag = True
        self.repetition_count = 0
        self.find_grid_count = 0
        # 这个变量表示，智能体做出的action是否有助于帮助自己逃离稀疏奖励的困境
        # 如果是True，表示这个action是有助于逃离稀疏奖励的困境的
        # 如果是False，表示这个action是没有助于逃离稀疏奖励的困境的
        # 如果是None，表示现在没有在稀疏奖励的困境中
        self.rescue_path_reduce_flag = None
        self.communicate_update_flag = False
        # 有时候因为障碍物生成的不好，智能体会处在一个被障碍物包围的局面，这个时候就需要重开
        self.surrounded_flag = False
        # 救援动作的列表
        self.action_list = []
        self.path = None   
        self.map_condense_degree = 0
        self.local_map_radius = 2 * self.view_range
        # self.local_view_map = 0.5 * np.ones((2 * self.local_map_radius - 1, 2 * self.local_map_radius - 1), dtype=np.float16)


class Human(object):
    def __init__(self, pos):
        self.pos = pos


class Layout(object):
    def __init__(self, map_size, layout):
        self.map_size = map_size
        self.layout = layout


    def generate_obstacles_and_free_spaces(self, width: int, height: int, obstacle_percentage: float) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]]]:
        if obstacle_percentage < 0 or obstacle_percentage > 1:
            raise ValueError("Obstacle percentage must be between 0 and 1.")

        total_cells = width * height
        total_obstacles = int(total_cells * obstacle_percentage)

        obstacle_coordinates = set()
        available_positions = set((x, y) for x in range(1, width - 1) for y in range(1, height - 1))

        while len(obstacle_coordinates) < total_obstacles and available_positions:
            x, y = random.choice(tuple(available_positions))

            tree_pos = {(x + dx, y + dy) for dx, dy in product(range(-1, 2), repeat=2)}
            obstacle_coordinates.update(tree_pos)
            available_positions -= tree_pos

        free_spaces = set((x, y) for x in range(1, width - 1) for y in range(1, height - 1)) - obstacle_coordinates

        return list(obstacle_coordinates), list(free_spaces)
    def will_create_closed_shape(self, obstacle_coordinates: set, x: int, y: int, width: int, height: int) -> bool:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()
        queue = [(x, y)]
        visited.add((x, y))

        while queue:
            current_x, current_y = queue.pop(0)

            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy

                if 0 <= next_x < width and 0 <= next_y < height and (next_x, next_y) not in visited and (
                next_x, next_y) not in obstacle_coordinates:
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))

        return len(visited) + len(obstacle_coordinates) < width * height



class SearchGrid(gym.Env):
    def __init__(self, map_set, map_num):
        # self.observation_space = spaces.Box(low=0, high=1, shape=(4, 50, 50))
        # self.action_space = spaces.Discrete(4)
        # 注意，当我用evaluate的时候，使用的是下面两行，当我运行train的时候，暂时使用的是上面两行，下面两行是否可行，暂时没有测试
        # 补充，经过测试，发现似乎确实可行，那么就暂时决定grid_drone就这么用了
        # train.py 为每个env选择一份地图
        self.map_set = map_set
        self.map_num = map_num
        # self.choose_map = self.map_set[self.map_num]
        # When use mlp
        # self.view_range = 10
        self.view_range = 5
        self.local_map_radius = 2 * self.view_range # 10
        self.local_map_range = 2 * self.local_map_radius - 1 # 19
        
        # 三重课程学习的配置
        self.run_time = 4  # 10 25 40 45 55 65 75 100 110 140 200 Run run_time steps per game 当不禁止碰撞的时候，我用的参数是1000
        self.prob = 0
        
        
        self.init_param()
        self.observation_space = spaces.Box(low=0, high=1, shape=(1 * 60 * 60 + self.local_map_range**2,)) # 60*60是全局地图的大小，9*9是局部地图的大小
        self.share_observation_space = spaces.Box(low=0, high=1, shape=(1 * 60 * 60 + self.drone_num * self.local_map_range**2,))
        self.action_space = spaces.Discrete(4)
        # print("share_observation_space",self.share_observation_space)
        self.seed()
        self.reset()
        
    def raise_difficulty(self):
        self.run_time = self.run_time + 5
    
    def down_prob(self):
        self.prob = self.prob - 0.1
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.MC_iter = self.MC_iter + 1
        # print("MC_iter", self.MC_iter)
        explored_acreage = np.count_nonzero(self.joint_map[1])
        self.exploration_prop = explored_acreage /self.free_acreage
        
        # 救援模块
        # 1 表示该 step 没有使用 rescue， 0 表示该 step 使用了 rescue
        rescue_mask = np.ones(self.drone_num)
        # 计算当前地图探索的百分比
        explored_acreage = np.count_nonzero(self.joint_map[1])
        self.exploration_prop = explored_acreage /self.free_acreage
        
        # 分别记录两个agent没有探索出新地方的次数
        for i in range(self.drone_num):
            self.grid_agents[i] = self.average_list_true[i]
        for i in range(self.drone_num):
            if self.grid_agents[i] <= 0:
                # print("agent", i, "repetition", self.agent_repetition[i])
                self.agent_repetition[i] = self.agent_repetition[i] + 1
                self.agent_repetition_reward[i] = self.agent_repetition_reward[i] + 1
            else:
                # print("agent", i, "self.grid_agents[i]", self.grid_agents[i])
                self.agent_repetition[i] = 0
                self.agent_repetition_reward[i] = 0
        self.last_grid_agents = self.grid_agents.copy()
        


        # 环境变化模块
        self.drone_step(action)
        self.human_take_action()
        self.human_step(self.human_act_list)
        # self.get_full_obs()
        # print('开始执行', self.MC_iter)
        ax2_image = self.get_joint_obs(self.MC_iter)
        
        observation, reward, rescue_reward, done, info, available_actions = self.state_action_reward_done(rescue_mask)
        # # 是 使用 individual reward or shared reward
        #
        # reward = np.full_like(reward, np.mean(reward) * len(reward))  # 使用numpy的广播功能对所有奖励进行均值填充
        # reward = np.full_like(reward, np.mean(reward))  # 使用numpy的广播功能对所有奖励进行均值填充
        
        # 对single_map 和 joint_map进行处理：
        
        observation = [np.concatenate((o.flatten(), drone.local_view_map.flatten())) for o, drone in zip(observation, self.drone_list)] # 使用列表推导式对每个观测值进行扁平化处理
        self.joint_map_process = 0.5 * np.ones((120,120))
        
        # 整合 open grid 到 joint_map 里
        # open grid 最终的值在 0~0.3 之间
        self.joint_map_process[self.joint_map[1] > 0] = 1 - self.joint_map[1][self.joint_map[1] > 0]
        # 整合 occupied grid 到 joint_map 里
        self.joint_map_process[self.joint_map[2] > 0] = 0
        
        # 整合智能体自己的位置到 joint_map里
        for drone in self.drone_list:
            self.joint_map_process[drone.pos[0], drone.pos[1]] = 5.5
        
        
        # 压缩或者截取 joint_map
        if np.any(self.drone_list[i].map_condense_degree == 1 for i in range(self.drone_num)):
            division_factor = 2
            map_processed = self.joint_map_process
            reshaped_map = map_processed.reshape(map_processed.shape[0]//division_factor, division_factor, -1, division_factor).swapaxes(1, 2)
            self.joint_map_process = self.encode(reshaped_map[..., 0, 0], reshaped_map[..., 0, 1], reshaped_map[..., 1, 0], reshaped_map[..., 1, 1])           
        else:
            division_factor = 2
            map_size = self.map_size // division_factor
            # 判断象限并提取60x60的区域  
            if np.any(self.drone_list[i].pos[0] < map_size and self.drone_list[i].pos[1] < map_size for i in range(self.drone_num)):  # 第一象限
                quadrant_map = self.joint_map_process[:map_size, :map_size]
            elif np.any(self.drone_list[i].pos[0] < map_size and self.drone_list[i].pos[1] >= map_size for i in range(self.drone_num)):  # 第二象限
                quadrant_map = self.joint_map_process[:map_size, map_size:]
            elif np.ary(self.drone_list[i].pos[0] >= map_size and self.drone_list[i].pos[1] < map_size for i in range(self.drone_num)):  # 第三象限
                quadrant_map = self.joint_map_process[map_size:, :map_size]
            else:  # 第四象限
                quadrant_map = self.joint_map_process[map_size:, map_size:]
            self.joint_map_process = quadrant_map
        
        flattened_local_maps = np.concatenate([drone.local_view_map.ravel() for drone in self.drone_list])
        final_joint_map = np.concatenate((self.joint_map_process.ravel(), flattened_local_maps.ravel()), axis=0)
        # return observation, reward, done, info, self.joint_map.ravel(), rescue_mask
        # return observation, reward, done, info, final_joint_map, rescue_mask
        # return observation,final_joint_map, reward, rescue_reward,  done, info, available_actions
        
        # 对于 HAPPO的代码：step 的return 格式是：        
        # sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, sub_agent_joint_map, sub_agent_rescue_masks, available_actions = self.env.step(actions)

        return observation, reward, done, info, final_joint_map, rescue_mask,  available_actions

 
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None):
        self.init_param()
        self.MC_iter += 1
        if self.MC_iter > 1:
            print("error")
        self.get_joint_obs(self.MC_iter)
        for i in range(self.drone_num):
            self.drone_list[i].repetition_count = 0
        self.last_drone_pos += [drone.pos for drone in self.drone_list]
        observation, _, _, _, info, available_actions = self.state_action_reward_done(None)
        # 使用列表推导式对每个观测值进行扁平化处理
        observation = [np.concatenate((o.flatten(), drone.local_view_map.flatten())) for o, drone in zip(observation, self.drone_list)] # 使用列表推导式对每个观测值进行扁平化处理
                
        # 整合 open grid 到 joint_map 里
        # open grid 最终的值在 0~0.3 之间
        self.joint_map_process[self.joint_map[1] > 0] = 1 - self.joint_map[1][self.joint_map[1] > 0]
        # 整合 occupied grid 到 joint_map 里
        self.joint_map_process[self.joint_map[2] > 0] = 0
        # 整合智能体自己的位置到 joint_map里
        for drone in self.drone_list:
            self.joint_map_process[drone.pos[0], drone.pos[1]] = 5.5
        
        
        # 压缩或者截取 joint_map
        if np.any(self.drone_list[i].map_condense_degree == 1 for i in range(self.drone_num)):
            division_factor = 2
            map_processed = self.joint_map_process
            reshaped_map = map_processed.reshape(map_processed.shape[0]//division_factor, division_factor, -1, division_factor).swapaxes(1, 2)
            self.joint_map_process = self.encode(reshaped_map[..., 0, 0], reshaped_map[..., 0, 1], reshaped_map[..., 1, 0], reshaped_map[..., 1, 1])           
        else:
            division_factor = 2
            map_size = self.map_size // division_factor
            # 判断象限并提取60x60的区域  
            if np.any(self.drone_list[i].pos[0] < map_size and self.drone_list[i].pos[1] < map_size for i in range(self.drone_num)):  # 第一象限
                quadrant_map = self.joint_map_process[:map_size, :map_size]
            elif np.any(self.drone_list[i].pos[0] < map_size and self.drone_list[i].pos[1] >= map_size for i in range(self.drone_num)):  # 第二象限
                quadrant_map = self.joint_map_process[:map_size, map_size:]
            elif np.ary(self.drone_list[i].pos[0] >= map_size and self.drone_list[i].pos[1] < map_size for i in range(self.drone_num)):  # 第三象限
                quadrant_map = self.joint_map_process[map_size:, :map_size]
            else:  # 第四象限
                quadrant_map = self.joint_map_process[map_size:, map_size:]
            self.joint_map_process = quadrant_map
        

        try:
            drone_mean_local = np.mean([drone.local_view_map for drone in self.drone_list], axis=0)
        except:
            for i in range(self.drone_num):
                print("drone ", i, "local view shape is ", self.drone_list[i].local_view_map.shape)
                
        flattened_local_maps = np.concatenate([drone.local_view_map.ravel() for drone in self.drone_list])
        final_joint_map = np.concatenate((self.joint_map_process.ravel(), flattened_local_maps.ravel()), axis=0)

        # return observation, self.joint_map.flatten()
        return observation, final_joint_map, available_actions

        # return (observation[0])

    def render(self):
        pass

    def close(self):
        pass

    def drone_step(self, drone_act_list):
        # 定义每个方向的增量
        delta = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
        # 更新每个机器人的位置
        self.collision = np.zeros(self.drone_num)
        for k in range(self.drone_num):
            # 获取机器人要执行的操作
            # 当评价多机时，用这个
            action = drone_act_list[k]
            # 根据操作计算出机器人的新位置
            if type(action) == int:
                print("action is ", action)
                direction = action 
            elif len(action) == 1: # 获取机器人要移动的方向
                direction = action[0]
            elif len(action) == 4:
                direction = np.argmax(action)  # 获取机器人要移动的方向
            # print("the direction is ", direction)

            # 如果智能体陷入了周围都是被探索的区域的困境，那么我们就要判断当前动作是否有助于帮助智能体脱离困境·
            # 判断标准是：如果当前动作的执行，会使得智能体脱离困境的总路程减少，那么我们就将rescue_path_reduce_flag设置为True
            # 我们在state_action_reward那个函数中，当rescue_path_reduce_flag为True时，就会给予智能体一个奖励，之后将rescue_path_reduce_flag设置为False       
            
            # 禁止撞击
            
            # 测试是否在最大run_time = 200的情况下，是否是因为每次只移动一格，导致智能体很难脱离稀疏奖励区域
                
            # temp_pos = self.drone_list[k].pos + 2*delta[direction]  # 根据方向更新机器人的位置
            # temp_pos_1 = self.drone_list[k].pos + delta[direction] 
            # if self.land_mark_map[temp_pos[0], temp_pos[1]] > 0 or self.land_mark_map[temp_pos_1[0], temp_pos_1[1]] > 0:
            #     self.collision[k] = 1
            
            # 测试正常情况下，是否是因为最大run_time 过长，导致智能体的训练出现问题
            temp_pos = self.drone_list[k].pos + delta[direction]  # 根据方向更新机器人的位置
            if self.land_mark_map[temp_pos[0], temp_pos[1]] > 0:
                self.collision[k] = 1
            else:
                self.drone_list[k].pos = temp_pos
                # temp_pos = self.drone_list[k].pos
                # self.drone_list[k].pos = temp_pos # 有这个就不禁止撞击了
                if self.drone_list[k].whole_map[1, self.drone_list[k].pos[0], self.drone_list[k].pos[1]] > 0:
                    if self.drone_list[k].repetition_count > self.repetition_threshold_for_reward:
                        # try:
                        _, old_action_list, goal_r, goal_c = generate_path(env=self, id=k, free_zone = self.free_map_rescue, obstacle_map = self.joint_map[2], pos = self.drone_list[k].pos)
                        _, new_action_list, goal_r, goal_c = generate_path(env=self, id=k, free_zone = self.free_map_rescue, obstacle_map = self.joint_map[2], pos = temp_pos)
                        
                        if len(new_action_list) < len(old_action_list):
                            self.drone_list[k].rescue_path_reduce_flag = True
                            # print("好动作")
                        else:
                            self.drone_list[k].rescue_path_reduce_flag = False
                            # print("坏动作")
                       
                self.collision[k] = 0
                   

            # 不禁止撞击
            # self.drone_list[k].pos = temp_pos
        

    def human_take_action(self):
        self.human_act_list = [0] * self.human_num
        for i in range(self.human_num):
            self.human_act_list[i] = random.randint(0, 3)

    def human_step(self, human_act_list):
        for k in range(self.human_num):
            human_pos = self.human_list[k].pos
            human_init_pos = self.human_init_pos[k]

            if human_act_list[k] == 0:
                new_pos = human_pos[0] - 1
                if new_pos > 0 and new_pos - human_init_pos[0] > -self.move_threshold:
                    free_space = self.land_mark_map[new_pos, human_pos[1]]
                    if free_space == 0:
                        human_pos[0] = new_pos
            elif human_act_list[k] == 1:
                new_pos = human_pos[0] + 1
                if new_pos < self.map_size - 1 and new_pos - human_init_pos[0] < self.move_threshold:
                    free_space = self.land_mark_map[new_pos, human_pos[1]]
                    if free_space == 0:
                        human_pos[0] = new_pos
            elif human_act_list[k] == 2:
                new_pos = human_pos[1] - 1
                if new_pos > 0 and new_pos - human_init_pos[1] > -self.move_threshold:
                    free_space = self.land_mark_map[human_pos[0], new_pos]
                    if free_space == 0:
                        human_pos[1] = new_pos
            elif human_act_list[k] == 3:
                new_pos = human_pos[1] + 1
                if new_pos < self.map_size - 1 and new_pos - human_init_pos[1] < self.move_threshold:
                    free_space = self.land_mark_map[human_pos[0], new_pos]
                    if free_space == 0:
                        human_pos[1] = new_pos

    def get_full_obs(self):
        # Initialize an array with ones
        obs = np.ones((self.map_size, self.map_size, 3))

        # Set [0, 0, 0] for wall and tree locations
        wall_tree_mask = np.logical_or(self.land_mark_map == 1, self.land_mark_map == 2)
        obs[wall_tree_mask] = 0

        # Set [1, 0, 0] for human locations
        for i in range(self.human_num):
            human_pos = tuple(self.human_list[i].pos)
            obs[human_pos] = [1, 0, 0]

        # Set [0.5*i, 0, 0.5*i] for drone locations
        for i in range(self.drone_num):
            drone_pos = tuple(self.drone_list[i].pos)
            obs[drone_pos] = [0.5 * i, 0, 0.5 * i]

        return obs

    def get_drone_obs(self, drone):  # 获得无人机的观测，这里的drone是类
        # print("执行get_drone_obs", drone.id)
        drone.observed_obs = []
        drone.unobserved = []
        drone.individual_observed_obs = 0
        drone.observed_drone = []
        drone.communicate_rate = 0
        index = random.randint(self.sensing_threshold[0], self.sensing_threshold[1])
        obs_size = 2 * drone.view_range - 1
        sensing_size = 2 * (drone.view_range + index) - 1
        obs = np.ones((obs_size, obs_size, 3))
        # 这里是给机器人感知其他机器人的位置加了波动
        # 对于单个agent，第一层记录自己的观测
        # 第二层记录自己对障碍物的观测
        #对于joint obs，是 obs 的 joint整合



        # 先通过通信更新得图：
        # 在观测范围内进行信息更新，更新时间戳地图1，轨迹地图2和障碍物地图3
        # 禁止通信
        drone.whole_map[2] = 0
        drone.communicate_list = []

        for k in range(self.drone_num):
            if self.drone_list[k].id != drone.id and (self.drone_list[k].pos[0] - drone.pos[0]) ** 2 \
                    + (self.drone_list[k].pos[1] - drone.pos[1]) ** 2 <= sensing_size ** 2:
                drone.communicate_list.append(k)
                # 记录其他智能体的位置，以后可以变成记录其他智能体的轨迹
                drone.whole_map[2, self.drone_list[k].pos[0], self.drone_list[k].pos[1]] = 1
                # print("信息交换")
                # print("距离的平方是", (self.drone_list[k].pos[0]-drone.pos[0])**2\
                #     + (self.drone_list[k].pos[1]-drone.pos[1])**2)
                # print("距离阈值是",sensing_size**2)
        drone.grid_communication = 0
        drone.obstacle_communication = 0

        if drone.communicate_list:
            # Combine maps from all drones, including the current one
            maps = np.array([self.drone_list[i].whole_map for i in drone.communicate_list] + [drone.whole_map])
            # Compute the maximum values for channels 1, 2, and 3
            max_channels = np.max(maps[:, [1, 2, 3], :, :], axis=0)

            # Update grid and obstacle communication values
            drone.grid_communication += np.count_nonzero(max_channels[0]) - np.count_nonzero(drone.whole_map[1])
            drone.obstacle_communication += np.sum(max_channels[2] > drone.whole_map[3, :, :])

            # Update the drone's whole_map with the maximum values
            drone.whole_map[[1, 2, 3], :, :] = max_channels


        # Large map
        drone.whole_map[0] = 0
        drone.whole_map[0, :drone.pos[0]+1, :drone.pos[1]+1] = self.memory_step


        # # 确定观测到的区域, 代替上面注释掉的部分
        # x_indices = np.arange(drone.pos[0] - (drone.view_range + index), drone.pos[0] + drone.view_range + index + 1)
        # y_indices = np.arange(drone.pos[1] - (drone.view_range + index), drone.pos[1] + drone.view_range + index + 1)
        # # 确定在观察区域内是否有其他无人机，以及在哪里
        # for other_drone in self.drone_list:
        #     if other_drone.id != drone.id and other_drone.pos[0] in x_indices and other_drone.pos[1] in y_indices:
        #         drone.observed_drone.append([other_drone.pos[0], other_drone.pos[1]])
        #         drone.whole_map[2, other_drone.pos[0], other_drone.pos[1]] = self.memory_step  # add other agent's history positions to the map
        #         drone.communicate_rate += 1

        # 这里循环的目的是构建障碍物地图

        # Create a meshgrid with the correct dimensions
        xx, yy = np.meshgrid(np.arange(obs_size), np.arange(obs_size), indexing='ij')

        # Calculate the actual x and y coordinates in the land_mark_map
        actual_x = np.clip(drone.pos[0] - obs_size // 2 + xx, 0, self.map_size - 1)
        actual_y = np.clip(drone.pos[1] - obs_size // 2 + yy, 0, self.map_size - 1)

        # Find the indices in the land_mark_map where the value is 2
        mask = (self.land_mark_map[actual_x, actual_y] == 2)
        # Set the corresponding indices in the obs array to 0
        obs[mask] = 0


        coord_per_obs_length = 0 # 记录agent一次探索的空白区域的坐标有几个
        drone.coord_per_obs = np.empty((obs_size**2, 2)) # 记录每次每个agent探索的空白区域的坐标，用于后续惩罚agents在一个step中，过多区域重合的现象
        for i in range(obs_size):
            for j in range(obs_size):
                # obs_size = 2 * r - 1
                # 最左侧的位置 = pos - (r-1)
                x = i + drone.pos[0] - (drone.view_range - 1)
                y = j + drone.pos[1] - (drone.view_range - 1)
                drone_positions = {tuple(drone.pos) for drone in self.drone_list}
                human_position = {tuple(human.pos) for human in self.human_list}

                if (x, y) in human_position:# 是否有目标点在观测范围内
                    obs[i, j, 0] = 1
                    obs[i, j, 1] = 0
                    obs[i, j, 2] = 0

                if (x, y) in drone_positions:# 是否有其他机器人在观测范围内
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0
                    obs[i, j, 2] = 0.5
                if 0 <= x <= self.map_size - 1 and 0 <= y <= self.map_size - 1 :  # 是否有障碍物在观测范围内
                    if self.land_mark_map[x, y] == 1:
                        obs[i, j] = 0
                    # if self.land_mark_map[x, y] == 2 or ((x, y) in drone_positions and (drone.pos[0]!=x and drone.pos[1]!= y)) :  # 不透明 在发现障碍物后对观测进行处理
                    if self.land_mark_map[x, y] == 2:  # 透明 transparent 在发现障碍物后对观测进行处理

                        obs[i, j] = 0
                        drone.observed_obs.append([x, y])
                        gap = [drone.observed_obs[-1][0] - drone.pos[0], \
                               drone.observed_obs[-1][1] - drone.pos[1]]
                        gap_abs = [abs(drone.observed_obs[-1][0] - drone.pos[0]), \
                                   abs(drone.observed_obs[-1][1] - drone.pos[1])]
                        chosen_gap = max(gap_abs)

                        if chosen_gap < drone.view_range:
                            if gap[0] >= 0 and gap[1] > 0:
                                if gap[0] == 0:
                                    if obs[i + 1, j, 0] == 0 and obs[i + 1, j, 1] == 0 and \
                                            obs[i + 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num_1, j + num + 1])
                                    if obs[i - 1, j, 0] == 0 and obs[i - 1, j, 1] == 0 and \
                                            obs[i - 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num_1, j + num + 1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + num + 1, j + num + 1])
                            if gap[0] > 0 and gap[1] <= 0:
                                if gap[1] == 0:
                                    if obs[i, j + 1, 0] == 0 and obs[i, j + 1, 1] == 0 and \
                                            obs[i, j + 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num + 1, j + num_1])
                                    if obs[i, j - 1, 0] == 0 and obs[i, j - 1, 1] == 0 and \
                                            obs[i, j - 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num + 1, j - num_1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + num + 1, j - num - 1])
                            if gap[0] < 0 and gap[1] >= 0:
                                if gap[1] == 0:
                                    if obs[i, j + 1, 0] == 0 and obs[i, j + 1, 1] == 0 and \
                                            obs[i, j + 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num - 1, j + num_1])
                                    if obs[i, j - 1, 0] == 0 and obs[i, j - 1, 1] == 0 and \
                                            obs[i, j - 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num - 1, j - num_1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i - num - 1, j + num + 1])
                            if gap[0] <= 0 and gap[1] < 0:
                                if gap[0] == 0:
                                    if obs[i + 1, j, 0] == 0 and obs[i + 1, j, 1] == 0 and \
                                            obs[i + 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num_1, j - num - 1])
                                    if obs[i - 1, j, 0] == 0 and obs[i - 1, j, 1] == 0 and \
                                            obs[i - 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num_1, j - num - 1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i - num - 1, j - num - 1])


                else:  # 其他情况
                    obs[i, j] = 0.5
                    # print("obs i j", obs[i,j])
                # 这里是设置圆形观测区域
                if (drone.view_range - 1 - i) * (drone.view_range - 1 - i) + (drone.view_range - 1 - j) * (
                        drone.view_range - 1 - j) > drone.view_range * drone.view_range:
                    obs[i, j] = 0.5



        for pos in drone.unobserved:  # 这里处理后得到的obs是能观测到的标志物地图
            obs[pos[0], pos[1]] = 0.5

        # 对观测到的区域添加时间戳, 并且记录具体的长度
        drone.open_information_gain = 0 # 记录探索区域的更新程度
        # 统计在当前位置中，空白区域占多大的百分比，从而得到障碍物占多大的百分比
        # 这个障碍物百分比，可以用来描述”不碰撞的难度“，障碍物的百分比越高，不碰撞的难度越高，碰撞的惩罚应该越小
        count = np.sum(np.all(obs == [1, 1, 1], axis=-1))
        drone.open_degree = count/obs_size/obs_size

        # 计算新找的了多少个区域
        drone.find_grid_count = 0
        # 刷新local_view_map
        # drone.local_view_map = np.ones((obs_size, obs_size)) * 0.5
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1
                # 对观测到的区域添加时间戳
                # print("obs i j", obs[i, j])
                if np.array_equal(obs[i, j], [1, 1, 1]):
                    # 观测到了新区域，新区域的值被赋为1，新值与旧值的差就是这一个step的information gain
                    # If find new area, then the repetation_flag is set to False.
                    if drone.whole_map[1, x, y] == 0:
                        drone.repetition_flag =  False
                        drone.find_grid_count = drone.find_grid_count + 1
                    drone.open_information_gain = drone.open_information_gain + 1 - drone.whole_map[1, x, y]
                    drone.whole_map[1, x, y] = 1 
                    # print("droe.coord_per_obs shape",drone.coord_per_obs.shape)
                    drone.coord_per_obs[coord_per_obs_length] = (x, y)
                    coord_per_obs_length = coord_per_obs_length + 1
                    
        # drone.local_view_map= drone.whole_map[] =  # 局部观测地图，添加open grid

        # If not find new area
        # for buffer1 curriculum learning 
        # if drone.repetition_flag is True:
        #     drone.repetition_count = drone.repetition_count + 1
        #     # print("drone id", drone.id,"repetition_count", drone.repetition_count)
        # else:
        #     drone.repetition_count = 0
            # print("drone id", drone.id,"无重复")
        drone.rescue_path_reduce_flag = None
        

        
        drone.repetition_flag = True
        
            
            
        # 进行轨迹的衰减, 是上面那段注释的优化
        # drone.whole_map[2, :, :] -= 1 / self.t_u
        # drone.whole_map[2, drone.whole_map[2, :, :] < 1 / self.t_u] = 0
        # 进行自身探索地图的衰减
        # 最多也就衰减到0.3, 毕竟是已经探索过的地方，能够确定是open的。如果衰减到0， 就相当于从来没有探索过了
        # drone.whole_map[1, :, :] = np.maximum(0, drone.whole_map[1, :, :] - 0.0025)
        
        # 轨迹衰减
        # drone.whole_map[1, :, :] = np.where(drone.whole_map[1, :, :] > 0.7, drone.whole_map[1, :, :] - 0.01, drone.whole_map[1, :, :])
        
        # print("relative_direction:",drone.relative_direction)
        return obs

    def get_joint_obs(self, time_stamp):
        # Modification record which aqt find the target
        # One target can be found by multi agents at the smae time point.
        # Target_per_agent defines how many targets are found by each agent at thie time point.
        self.target_per_agent = np.zeros(self.drone_num)
        human_del_list = []
        len_human_del_list = 0
        # Record obstacle gain of each agent
        # self.obstacle_gain_per_agent = np.zeros(self.drone_num)
        len_obstacle_gain = len(self.obstacles)

        # self.obstacles_temp = copy.deepcopy(self.obstacles)
        self.per_observed_goal_num = 0
        self.time_stamp = time_stamp
        obs = np.full((self.map_size, self.map_size, 3), 0.5)
        # self.obstacle_multi_agent = [self.obstacles for k in range(self.drone_num)]
        # 打乱智能体的决策顺序，随机抽取智能体来决策
        allowed_values = list(range(self.drone_num))
        k_list = random.sample(allowed_values, self.drone_num)
        # print("开始获得联合观测")
        
        for k in k_list:
            self.drone_list[k].individual_observed_obs = 0
            # print("drone id is k", k)
            temp = self.get_drone_obs(self.drone_list[k])
            size = temp.shape[0]
            temp_list_individual = []

            for i in range(size):
                for j in range(size):
                    x = i + self.drone_list[k].pos[0] - self.drone_list[k].view_range + 1
                    y = j + self.drone_list[k].pos[1] - self.drone_list[k].view_range + 1
                    # 如果一个位置根本没有被观测到，就不执行赋值
                    if np.all(temp[i, j] == (0.5,0.5,0.5)):
                        continue
                    else:
                        obs[x, y] = temp[i, j]
                        temp_list_individual.append([x, y])
                        # 这里为了判断观测中有多少障碍物，并更新障碍物地图
                        # 如果temp[i,j] = (0,0,0)
                        if not temp[i, j].any():
                            if [x, y] not in self.obstacles:
                                self.drone_list[k].individual_observed_obs += 1
                                # self.obstacle_multi_agent[k].append([x,y])
                                self.obstacles.append([x, y])  # 所有机器人观测过的障碍物
                            self.drone_list[k].whole_map[3, x, y] = 1  # add obstacle information to each agent's whole map
                            # self.drone_list[k].local_view_map[i, j] = 1 # 局部观测地图，添加障碍物
                            # self.joint_map[2, x, y] = 1
                        # 如果观测中有目标，则清除被观测到的目标
                        if all(obs[x, y] == [1, 0, 0]):
                            self.per_observed_goal_num += 1
                            for num, goal in enumerate(self.human_list):
                                if goal.pos[0] == x and goal.pos[1] == y:
                                    human_del_list.append(num)
                                    # print("11111111111111111111")


            self.obstacle_gain_per_agent[k] = len(self.obstacles) - len_obstacle_gain
            len_obstacle_gain = len(self.obstacles)
            self.target_per_agent[k] = self.target_per_agent[k] + len(human_del_list) - len_human_del_list
            # print("self.target_per_agent:",k,self.target_per_agent[k])
            len_human_del_list = len(human_del_list)
            # self.drone_list[k].individual_observed_zone = temp_list_individual
            # 这里计算观测区域去掉障碍物的面积
            # self.drone_list[k].area = len(self.drone_list[k].individual_observed_zone) - \
            #                           self.drone_list[k].individual_observed_obs
            # print(len(self.drone_list[k].individual_observed_zone))

        # 去掉重复检测到的target
        # Convert human_del_list to a set to remove duplicates
        human_del_set = set(human_del_list)
        # Use list comprehension to create new_human_list and new_human_init_pos
        new_human_list = [h for i, h in enumerate(self.human_list) if i not in human_del_set]
        new_human_init_pos = [h for i, h in enumerate(self.human_init_pos) if i not in human_del_set]
        # Update human_count and human_list
        self.human_num = len(new_human_list)
        self.human_list = new_human_list
        self.human_init_pos = new_human_init_pos

        # 合并所有无人机的整个地图
        for drone in self.drone_list:
            drone.communicate_update_flag = False
        for drone in self.drone_list:
            self.joint_map[0, drone.pos[0], drone.pos[1]] = 5
            # 如果无人机的通信列表不为空，且无人机的通信更新标志位为False
            # 则将所有通信列表中的无人机的地图全部弄相同
            # 引入通信更新标志位的原因是，如果一个无人机的地图已经更新过了，就不需要再更新了
            if drone.communicate_list and drone.communicate_update_flag == False:
                maps = np.array([self.drone_list[i].whole_map for i in drone.communicate_list] + [drone.whole_map])
                # Compute the maximum values for channels 1, 2, and 3
                max_channels = np.max(maps[:, [1, 2, 3], :, :], axis=0)
                drone.whole_map[[1, 2, 3], :, :] = max_channels
                drone.communicate_update_flag = True
                for drone_id in drone.communicate_list:
                    self.drone_list[drone_id].whole_map[[1, 2, 3], :, :] = max_channels
                    self.drone_list[drone_id].communicate_update_flag = True

        # self.joint_map[0, :, :] = np.maximum(0,  self.joint_map[0, :, :] - 0.01)
        # self.joint_map[0] = np.max([drone.whole_map[0] for drone in self.drone_list], axis=0)
        self.joint_map[1] = np.max([drone.whole_map[1] for drone in self.drone_list], axis=0)
        # self.joint_map[2] = np.max([drone.whole_map[3] for drone in self.drone_list], axis=0)
        return obs

    # @vectorize
    def encode(self, v1, v2, v3, v4):
        return (v1 * 3**3 + v2 * 3**2 + v3 * 3 + v4) / (3**4 - 1)
    def condense_map(self, drone, soft_boundary=0, division_factor=2):
        # boundary 指的是一个比 drone 现在的地图边界更严格的地图边界，一个虚拟的边界
        # 严格指的是，这个边界更靠内，如果把地图边界叫做外环，那么这个虚拟的边界就是内环
        # 当超出这个“内环”时，drone 就要拓展地图了
        w_open = 0.2
        w_occupied = 0.2
        w_unexplored = 0.6
        
        if drone.map_condense_degree == 0:
            map_size = self.map_size // division_factor
            # 总体的View range 是否超出了 "内环"
            rows = drone.map_processed[0][[map_size, map_size+1], :]
            columns = drone.map_processed[0][:, [map_size, map_size+1]]
            
            if np.any(np.sum(rows != 0.5, axis=1) >= 2) or np.any(np.sum(columns != 0.5, axis=0) >= 2):
                drone.map_condense_degree = 1
            # 使用大小为2*2的框，通过三进制编码，将map_porcessed的大小压缩到原来的1/2
            # 压缩的代码如下：
        if drone.map_condense_degree == 1:
            map_processed = drone.map_processed[0]
            reshaped_map = map_processed.reshape(map_processed.shape[0]//division_factor, division_factor, -1, division_factor).swapaxes(1, 2)
            condensed_map = self.encode(
                reshaped_map[..., 0, 0],  # First value of the last two dimensions
                reshaped_map[..., 0, 1],  # Second value of the penultimate dimension
                reshaped_map[..., 1, 0],  # First value of the last dimension, second value of the penultimate dimension
                reshaped_map[..., 1, 1]   # Second value of the last two dimensions
)            
            
            drone.map_condensed[0] = condensed_map
        else:
            # 获取智能体位置
            drone_pos = drone.pos

            # 判断象限并提取60x60的区域  
            if drone_pos[0] < map_size and drone_pos[1] < map_size:  # 第一象限
                quadrant_map = drone.map_processed[0][:map_size, :map_size]
            elif drone_pos[0] < map_size and drone_pos[1] >= map_size:  # 第二象限
                quadrant_map = drone.map_processed[0][:map_size, map_size:]
            elif drone_pos[0] >= map_size and drone_pos[1] < map_size:  # 第三象限
                quadrant_map = drone.map_processed[0][map_size:, :map_size]
            else:  # 第四象限
                quadrant_map = drone.map_processed[0][map_size:, map_size:]

            drone.map_condensed[0] = quadrant_map
    
    def state_action_reward_done(self, rescue_masks):  # 这里返回状态值，奖励值，以及游戏是否结束
        # print("reward is")
        # reward = 0  # 合作任务，只设置单一奖励
        # reward_list = np.zeros(self.drone_num, dtype=np.float32)
        ####################设置奖励的增益
        target_factor = 0
        # 发现障碍物的奖励系数
        information_gain = 1
        # time step factor 变成 0, 取消时间惩罚
        # 时间惩罚
        time_step_factor = 1 # 从0.1升到0.3 优化一下轨迹
        # 发现新区域的奖励系数
        average_time_stamp_factor = 0.5 # 当禁止碰撞时，是4；不禁止碰撞时，是10
        time_stamp_update_gain = 1
        collision_factor = 10
        close_penalty = 20
        collision_decay = 0
        # 单步内，智能体探索区域重复的惩罚系数
        overlap_factor = 0 # 0.05
        # 对reward进行缩放，降低回报的波动，减少学习的难度。（深度强化学习落地指南 p77.
        reward_scale = 0.1

        done = False
        available_action = np.ones((self.drone_num, 4))
        # 检查每个pos的四个方向是否有障碍物，如果有，就将对应的avaliable action置为0
        
        # directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        # for i in range(self.drone_num):
        #     x, y = self.drone_list[i].pos
        #     self.drone_list[i].path, self.drone_list[i].action_list, goal_r, goal_c = generate_path(env=self, id=i, free_zone = self.free_map_rescue, obstacle_map = self.joint_map[2], pos = self.drone_list[i].pos)                
        #     current_rescue_path_length =  len(self.drone_list[i].action_list)
        #     for d, (dx, dy) in enumerate(directions):
        #         if self.drone_list[i].whole_map[3,x+dx, y+dy] > 0:
        #             available_action[i, d] = 0
        #     if self.target_per_agent[i] < 0:
        #         # print("上次没看到目标，没看到目标")
        #         for d, (dx, dy) in enumerate(directions):
        #             candidate_path, candiate_action_list, goal_r, goal_c = generate_path(env=self, id=i, free_zone = self.free_map_rescue, obstacle_map = self.joint_map[2], pos = [x+dx, y+dy])                
        #             candidate_path_length =  len(candiate_action_list)
        #             if candidate_path_length >= current_rescue_path_length:
        #                 prob = np.random.rand()
        #                 if prob < self.prob:
        #                     available_action[i, d] = 0
        #         if all(available_action[i] == 0):
        #             print("drone", i, "is trapped")
                    
        
        
        # single_map_set = [self.drone_list[k].whole_map for k in range(self.drone_num)]
        
        # 压缩状态空间
        # single_map_set = [self.drone_list[k].whole_map[1:].copy() for k in range(dself.drone_num)]
        single_map_set = [drone.whole_map for drone in self.drone_list]
        for drone_count, each_map in enumerate(single_map_set):
            # 整合 open grid 到最终的地图里
            self.drone_list[drone_count].map_processed[0][each_map[1] > 0] = 1 - each_map[1][each_map[1] > 0]
            # 整合 occupied grid 到最终的地图里
            self.drone_list[drone_count].map_processed[0][each_map[2] > 0] = 1
            # 整合智能体自己的位置到最终的地图里
            drone_pos = self.drone_list[drone_count].pos
            self.drone_list[drone_count].map_processed[0][drone_pos[0], drone_pos[1]] -= -4.5
            # 整合其他智能体的位置到最终的地图里
            self.drone_list[drone_count].map_processed[0][each_map[3] > 0] += 5.5
            
            drone_pos = self.drone_list[drone_count].pos
            last_drone_pos = self.drone_list[drone_count].last_pos[0]
            last_last_drone_pos = self.drone_list[drone_count].last_pos[1]
            if last_last_drone_pos is not None:
                self.drone_list[drone_count].map_processed[0][last_last_drone_pos[0], last_last_drone_pos[1]] -= -6.5
            if last_drone_pos is not None:
                self.drone_list[drone_count].map_processed[0][last_drone_pos[0], last_drone_pos[1]] -= -5.5
            
            # 判断opne grid 和 occupied grid 是否超过了图的四分之一分割的线
            # print("local map radius", self.drone_list[drone_count].local_map_radius)
            x_min = max(0, self.drone_list[drone_count].pos[0] - self.drone_list[drone_count].local_map_radius + 1)
            x_max = min(120, self.drone_list[drone_count].pos[0] + self.drone_list[drone_count].local_map_radius)
            y_min = max(0, self.drone_list[drone_count].pos[1] - self.drone_list[drone_count].local_map_radius + 1)
            y_max = min(120, self.drone_list[drone_count].pos[1] + self.drone_list[drone_count].local_map_radius)

            self.drone_list[drone_count].local_view_map = self.drone_list[drone_count].map_processed[:, x_min:x_max, y_min:y_max]
            
            
            
            
            # 创建一个全是 1 的数组，大小满足你的要求
            full_ones_array = np.zeros((1, self.local_map_range, self.local_map_range)) # 这里是你希望得到的图像的大小，你需要用实际值替代这些
            left_boundary = self.local_map_radius -1 - (self.drone_list[drone_count].pos[0] - x_min)
            right_boundary = self.local_map_radius -1 + x_max - self.drone_list[drone_count].pos[0]
            up_boundary = self.local_map_radius -1 - (self.drone_list[drone_count].pos[1] - y_min)
            down_boundary = self.local_map_radius- 1 + y_max - self.drone_list[drone_count].pos[1]
            
            # print("right bounday - left boundary is", left_boundary, right_boundary, down_boundary, up_boundary)
            
            # 将机器人局部观测的地图复制到“局部观测地图”上，并且，机器人自身的位置永远是在full_ones_array的正中间
            full_ones_array[:, left_boundary:right_boundary, up_boundary:down_boundary] = self.drone_list[drone_count].local_view_map

            self.drone_list[drone_count].local_view_map = full_ones_array
            
            # 然后加入自己在压缩网格中的相对位置的信息 12月1日 0：42 待处理
            
            
            # self.drone_list[drone_count].local_view_map = self.drone_list[drone_count].map_processed[:, 
            #     self.drone_list[drone_count].pos[0] - self.drone_list[drone_count].local_map_radius + 1: self.drone_list[drone_count].pos[0] + self.drone_list[drone_count].local_map_radius, 
            #     self.drone_list[drone_count].pos[1] - self.drone_list[drone_count].local_map_radius + 1: self.drone_list[drone_count].pos[1] + self.drone_list[drone_count].local_map_radius]
            self.condense_map(self.drone_list[drone_count])
           
                
            
        final_single_map_set = [drone.map_condensed for drone in self.drone_list]

        reward_list = [0 for i_agent in self.target_per_agent]  # 这里计算发现目标点的数量
  

        # 机器人的每个时间戳平均尽可能大，保证尽可能有多的区域被探索到
        # 记录 rescue reward
        rescue_reward = [0 for i in range(self.drone_num)]
        for i, drone in enumerate(self.drone_list):
            self.average_list_true[i] = self.average_list[i] = self.drone_list[i].find_grid_count
            # print("drone", i, "find grid count is", self.drone_list[i].find_grid_count)
            # print("drone.pos is", drone.pos)
            reward_list[i] = reward_list[i] + self.average_list[i] * average_time_stamp_factor + information_gain * self.obstacle_gain_per_agent[i]
            if drone.rescue_path_reduce_flag is True: # 怀疑这里有问题
                # print("Start guide: ", "agent", i, "repetition", drone.repetition_count, 'Mc_iter', self.MC_iter)   
                # print("timestamp gain is", self.drone_list[i].open_information_gain * time_stamp_update_gain )
                # print("origin reward is ", reward_list[i])
                # reward_list[i] = reward_list[i] +  self.drone_list[i].open_information_gain * time_stamp_update_gain 
                rescue_reward[i] =  time_stamp_update_gain
                reward_list[i] = reward_list[i] +  rescue_reward[i] 
            elif drone.rescue_path_reduce_flag is False:
                rescue_reward[i] = -1*time_stamp_update_gain
                reward_list[i] = reward_list[i] + rescue_reward[i]
            else:                
                rescue_reward[i] = 0
                
            drone.rescue_path_reduce_flag = None
                # print("该动作有利于脱困")
        # print("reward_list new area", reward_list)

        # 单步内探索区域重叠的惩罚：
        # 将所有坐标堆叠到一个数组中
        all_coordinates = np.vstack([self.drone_list[i].coord_per_obs for i in range(self.drone_num)])
        # 使用 np.unique 函数找到唯一的坐标
        unique_coordinates, counts = np.unique(all_coordinates, axis=0, return_counts=True)
        # 计算重复坐标的数量
        num_duplicates = np.sum(counts > 1)
        # print("reward list before duplicate is", reward_list)
        # print("num duplicates is", num_duplicates*overlap_factor)
        reward_list = list(map(lambda x: x - num_duplicates*overlap_factor, reward_list))  # 单步惩罚
        # reward_list = list(map(lambda x: x - min(num_duplicates*overlap_factor, max(abs(x)-1, 0)), reward_list))  # 单步惩罚
        # average = sum(average_list) / self.drone_num * average_time_stamp_factor
        # print("average is",average)
        # done_list = [done for i_agent in range(self.drone_num)]
        # reward_list = list(map(lambda x: x+average, reward_list))
        # print("reward list is", reward_list)
        target_found_num = self.human_num_copy - self.human_num

        # 时间惩罚
        reward_list = [x - min(time_step_factor, abs(x) / 2) if x > 0 else x - time_step_factor for x in reward_list]
        # print("加上时间惩罚后", reward_list)

        # # 发现目标的奖励
        # reward_list = [x + target_factor * i_agent for x, i_agent in zip(reward_list, self.target_per_agent)]
        # # 发现所有目标
        # if self.human_num == 0 and self.generate_human is True:
        #     # reward_list = list(map(lambda x: x + 500, reward_list))
        #     done = True
        #     print("Map index is", self.random_index)
        #     with open ("/home/cx/envs/EnvDrone/classic_control/map_index8.txt","a") as w:
        #         w.write(str(self.random_index)+"\n")
        #     # info['0'] = "find all target"

        # 机器人碰撞惩罚，
        for i in range(self.drone_num - 1):  # 如果机器人发生碰撞
            for j in range(i + 1, self.drone_num):
                distance = np.linalg.norm(np.array(self.drone_list[i].pos) - np.array(self.drone_list[j].pos))
                if distance <= 1:
                    # done = True
                    # print("robot collision")
                    reward_list[i] -= collision_factor
                    reward_list[j] -= collision_factor

        # 机器人和障碍物碰撞的惩罚
        for i, drone in enumerate(self.drone_list):
            if self.land_mark_map[drone.pos[0], drone.pos[1]] > 0 or self.collision[i] == 1:
                # done = True
                # print("obstacle collison")
                # reward_list[drone.id] -= (collision_factor - min(collision_decay, self.MC_iter))* max(drone.open_degree, 0.6)
                reward_list[drone.id] -= collision_factor
            if self.drone_list[i].surrounded_flag is True:
                done = True
 
        # 时间用尽但是还没有完成任务的惩罚
        if self.time_stamp > self.run_time:  # 超时
            done = True

        
        if done is True or self.time_stamp > self.run_time:
            self.reset()

        # 对 reward 进行放缩，使得 Q 更新不需要太大
        reward_list = list(map(lambda x: reward_scale * x, reward_list))
       
        done_list = [done]*self.drone_num
        
        # return final_single_map_set, reward_list, done_list, target_found_num
        return final_single_map_set, reward_list, rescue_reward, done_list, target_found_num, available_action


    def get_neighboring_free_spaces(self, pos: Tuple[int, int], free_spaces: List[Tuple[int, int]], distance: int) -> \
        List[Tuple[int, int]]:

        def is_valid_neighbor(free_space):
            x_diff = abs(pos[0] - free_space[0])
            y_diff = abs(pos[1] - free_space[1])
            neigobor_distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
            return x_diff < distance and y_diff < distance and neigobor_distance > 1

        random.shuffle(free_spaces)
        neighbors = [free_space for free_space in free_spaces if is_valid_neighbor(free_space)]

        return neighbors[:self.drone_num - 1]

    def pick_random_positions(self, free_spaces: List[Tuple[int, int]], x: int) -> Tuple[
        Tuple[int, int], List[Tuple[int, int]]]:
        tried_positions = set()
        while len(tried_positions) < len(free_spaces):
            temp_pos = tuple(random.choice(free_spaces))
            if temp_pos in tried_positions:
                continue
            tried_positions.add(temp_pos)
            distance = x + 2
            eligible_positions = self.get_neighboring_free_spaces(temp_pos, free_spaces, distance)

            # x 是总智能体数量-1。因为第一个agent已经被选择了，然后基于此，我们选择剩下的智能体
            if len(eligible_positions) >= x:
                return temp_pos, eligible_positions

        raise ValueError("Cannot find a temp_pos with enough eligible positions.")


    def is_valid_coord(self, coord, width, height, obstacle_coordinates):
        x, y = coord
        return 1 <= x < width - 1 and 1 <= y < height - 1 and coord not in obstacle_coordinates

    def get_neighbors(self, coord, width, height, obstacle_coordinates):
        x, y = coord
        return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if
                self.is_valid_coord((x + dx, y + dy), width, height, obstacle_coordinates)]

    def heuristic(self, coord, target_coord):
        x1, y1 = coord
        x2, y2 = target_coord
        return abs(x1 - x2) + abs(y1 - y2)

    def is_path_available(self, agent_coord, target_coord, width, height, obstacle_coordinates):
        visited = set()
        queue = deque([(0, agent_coord, 0)])

        obstacle_coordinates_set = set(obstacle_coordinates)

        while queue:
            _, current_coord, g_value = queue.popleft()

            if current_coord == target_coord:
                return True

            if current_coord not in visited:
                visited.add(current_coord)
                neighbors = self.get_neighbors(current_coord, width, height, obstacle_coordinates_set)

                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_g_value = g_value + 1
                        f_value = new_g_value + self.heuristic(neighbor, target_coord)
                        queue.append((f_value, neighbor, new_g_value))
                        queue = deque(sorted(queue, key=lambda x: x[0]))

        return False

    def init_param(self):
        self.MC_iter = 0
        self.target_occur_iter = 100
        self.offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右四个方向的偏移
        self.map_size = 120
        self.drone_num = 2
        # Rescue paragrams
        # 已探索区域的比例
        self.exploration_prop = 0
        # rescue 的动作列表
        self.rescue_action_list = []
        # 当发现rescue的目标点被其他智能体探索后，就重新生成目标点
        # 新生成的目标点有一定的“保质期”， 在这段时间内后，如果目标点及时被其他智能体探索了，我们也暂时不切换目标点
        # 这样做的好处是，可以避免频繁的切换目标点，导致智能体的运动不稳定
        self.rescue_target_keep_time = 1
        self.rescue_target_keep_time_count = self.rescue_target_keep_time
        # rescue找到的目标点的位置，agent使用广度优先搜索来找到目标点，会移动到搜索序列的最后一个点，也就是紧挨着目标点的点
        # 初始化目标点的位置，在不触发rescue的情况下，目标点的位置是不会被使用的
        self.goal_r = [0 for i in range(self.drone_num)]
        self.goal_c = [0 for i in range(self.drone_num)]
        # 在每个时间步，每个agent独自探索的网格数
        self.grid_agents = []
        for i in range(self.drone_num):
            # id starts from 0.
            self.rescue_action_list.append(rescue_action(actions=[], id=i))
            self.grid_agents.append(0)
        self.last_grid_agents = np.zeros(self.drone_num)
        self.agent_repetition = np.zeros(self.drone_num)
        self.agent_repetition_reward = np.zeros(self.drone_num)
        self.repetition_threshold = 100000000 # 基础款MIXER里用的是5
        self.repetition_threshold_for_reward = -1  # if self.run_time < 30 else 3 # 用于指导智能体从死胡同等不好的地方出来
        self.rescue_flag = False
        # The area explored by each agent each step
        self.average_list = [0] * self.drone_num
        self.average_list_true = [0] * self.drone_num
        self.last_drone_pos = []
        self.obstacle_gain_per_agent = np.zeros(self.drone_num)
        self.last_obstacle_gain_per_agent = np.zeros(self.drone_num)
        self.find_grid_count = np.zeros(self.drone_num)
        self.last_find_grid_cout = np.zeros(self.drone_num)
        self.tree_num = 3
        
        self.human_init_pos = []
        # 一开始不生成目标点，探索范围过了阈值之后再生成
        self.generate_human = False
        self.generate_threshold = 1.1
        self.human_num = 0
        self.human_num_temp = self.human_num
        self.human_num_copy = self.human_num
        self.sensing_threshold = [60, 61]
        self.time_stamp = None
        self.observed_zone = {}  # 带有时序的已观测点
        self.global_reward = []
        self.global_done = []
        self.per_observed_goal_num = None
        self.obstacles = []  # 记录所有机器人观测到的障碍物
        self.joint_map_process = 0.5 * np.ones((120,120))
        # 障碍物在地图中的面积占比
        self.obstacle_percentage = np.random.uniform(0.1, 0.3)
        self.obstacles_temp = []
        self.human_act_list = []
        self.drone_act_list = []
        self.joint_map = np.zeros((3, self.map_size, self.map_size))
        # initialize trees
        self.land_mark_map = np.zeros((self.map_size, self.map_size))  # 地标地图
        self.memory_step = 1
        self.global_obs_num = 0
        self.t_u = 10
        self.move_threshold = 2
        self.random_pos_robot = True
        self.random_pos_target = True
        self.last_n_step_pos = None
        self.n_step = 4
        view_range_2 = (self.view_range -2) ** 2
        self.view_range_2 = view_range_2
        self.collision = np.zeros(self.drone_num) # 记录which drone take an action that will cause collision
        self.random_index = np.random.randint(0, self.map_num)
        self.erosion_prob = 0.0
        self.choose_map = self.map_set[self.random_index]
        # 使用 NumPy 切片，隔一个采样一个
        # self.choose_map = self.choose_map[::2, ::2]
        
        # self.choose_map  = np.load('/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/obs_data.npy')
        
        # self.choose_map  = np.load('/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/downsampled_map.npy')
        # self.choose_map  = self.choose_map.reshape(60,60)
        # print('choose_map', self.choose_map.shape)
        # self.choose_map = np.where(self.choose_map == 0.5, 1, self.choose_map)

        # self.choose_map = 1 - self.choose_map

        # self.choose_map = self.get_Map()
        # if np.random.rand() >self.erosion_prob:
        # self.choose_map = self.erode_and_add_obstacles(map=self.choose_map, erosion_probability=0.3, obstacle_probability=0.05, erosion_times=2)

      
        # 生成一个随机整数（0 或 1）
        rand_num = np.random.randint(0, 2)

        # 如果随机数为 1，则旋转地图
        if rand_num == 1:
            self.choose_map = np.rot90(self.choose_map, k=-1)  # 沿顺时针方向旋转90°

        # 生成一个随机整数（0、1、2 或 3）
        rand_num = np.random.randint(0, 4)

        # 使用随机数作为旋转次数（每次沿顺时针方向旋转90°）
        # choose_map = np.rot90(self.choose_map, k=-rand_num)
        wall = np.argwhere(self.choose_map == 0)  # 获取障碍物的坐标
        self.free_zones = np.argwhere(self.choose_map == 1)  # 获取空白区域的坐标
        self.free_acreage = len(self.free_zones)

        # wall, free_zones = layout.generate_obstacles_and_free_spaces(height=self.map_size, width=self.map_size, obstacle_percentage=self.obstacle_percentage)
        # for pos in inverse_wall:
        for pos in wall:
            # tree_pos = []
            x, y = pos
            self.land_mark_map[x, y] = 2
            self.joint_map[2, x, y] = 1
        self.free_map_rescue = np.zeros((self.map_size, self.map_size))
        for pos in self.free_zones:
            # tree_pos = []
            x, y = pos
            self.free_map_rescue[x, y] = 1
    
        # 计算全局观察数
        self.global_obs_num = np.sum(self.land_mark_map == 2)

        # 初始化无人机
        if self.random_pos_robot:
            self.drone_list = []
            id = 0
            temp_pos, selected_positions = self.pick_random_positions(self.free_zones, self.drone_num - 1)
            self.drone_list.append(Drones(temp_pos, self.view_range, self.local_map_radius, id, self.map_size))
            self.drone_list.extend(
                [Drones(position, self.view_range, self.local_map_radius, id, self.map_size) for id, position in
                 enumerate(selected_positions, start=1)]
            )
        else:
            temp_pos = [[35, 5], [46, 10], [25, 23], [23, 25], [27, 25]]
            self.drone_list = [
                Drones(pos, self.view_range, self.local_map_radius, i, self.map_size) for i, pos in enumerate(temp_pos)
            ]

        # randomly initialize humans
        if self.random_pos_target:
            self.human_list = []

            for i in range(self.human_num):
                temp_pos = random.choice(self.free_zones)
                flag_in_agent_range = False  # 判断目标的是否一开始初始化在了智能体初始就能检测到的范围内
                for i_agent in range(self.drone_num):
                    try:
                        if (temp_pos[0] - self.drone_list[i_agent].pos[0]) ** 2 + (
                                temp_pos[1] - self.drone_list[i_agent].pos[1]) ** 2 <= view_range_2:
                            flag_in_agent_range = True
                            break
                    except:
                       print("self.drone_num",self.drone_num)
                       print("len temp_pos",len(temp_pos))


                # avaliable = self.is_path_available(agent_coord=self.drone_list[0].pos, target_coord=temp_pos,
                #                                    width=self.map_size, height=self.map_size,
                #                                    obstacle_coordinates=wall)

                # while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0 or flag_in_agent_range or not avaliable:

                # 计算所有智能体的 view_range_2
                view_range_2_list = [agent.view_range ** 2 for agent in self.drone_list]

                # 使用列表解析构建一个满足条件的点的列表
                filtered_free_zones = [
                    pos for pos in self.free_zones
                    if not any(
                        (pos[0] - agent.pos[0]) ** 2 + (pos[1] - agent.pos[1]) ** 2 <= view_range_2
                        for agent, view_range_2 in zip(self.drone_list, view_range_2_list)
                    )
                ]

                # 在循环中使用 filtered_free_zones 而不是 free_zones
                for i in range(self.human_num):
                    temp_pos = random.choice(filtered_free_zones)
                self.human_init_pos.append(np.array(temp_pos).copy())
                temp_human = Human(np.array(temp_pos))
                self.human_list.append(temp_human)
        # fixedly initialize humans
        else:
            self.human_list = []
            # temp_pos = [[16, 14], [34, 36], [16, 46], [40, 37], [48, 3]]
            temp_pos = [[45, 13], [38, 38], [16, 46], [35, 10], [48, 3]]
            for i in range(self.human_num):
                temp_human = Human(temp_pos[i])
                self.human_init_pos.append(temp_pos[i].copy())
                self.human_list.append(temp_human)
                
                
                

    def load_pgm_map(self, file_path):
        # 使用PIL库打开PGM文件
        image = Image.open(file_path)

        # 转换为灰度图像模式
        image = image.convert("L")

        # 获取图像的宽度和高度
        width, height = image.size

        # 创建一个二维列表来表示栅格地图
        grid_map = [[0] * width for _ in range(height)]

        # 遍历图像的每个像素，将像素值映射到栅格地图中
        for y in range(height):
            for x in range(width):
                pixel_value = image.getpixel((x, y))
                # 假设PGM图像中较亮的像素代表障碍物，将其标记为1，其他像素标记为0
                if pixel_value > 210:
                    grid_map[y][x] = 1

        grid_map = np.array(grid_map)

        return grid_map


    def downsample_map(self,grid_map, shape, downsample_factor):
        height, width = shape[0], shape[1]

        # Calculate new map dimensions
        new_height = height // downsample_factor
        new_width = width // downsample_factor
        print(f'new_height:{new_height}, new_width:{new_width}!')

        # Create new map
        downsampled_map = np.zeros((new_height, new_width))

        # Traverse the new map
        for i in range(new_height):
            for j in range(new_width):
                # Calculate the value of each grid in the new map
                    downsampled_map[i, j] = np.min(grid_map[i*downsample_factor : min((i+1)*downsample_factor, height), j*downsample_factor : min((j+1)*downsample_factor, width)])

        print(f'downsampled_map shape:{downsampled_map.shape}!')
        return downsampled_map


    def get_Map(self):
        grid_map = self.load_pgm_map('test.pgm')

        # Create downsampled map
        downsampled_map = self.downsample_map(grid_map, grid_map.shape, 4)

        downsampled_map = downsampled_map[10:45, 0:50]

        desired_shape = (60, 60)

        # Compute the padding widths
        pad_width_y = desired_shape[0] - downsampled_map.shape[0]
        pad_width_x = desired_shape[1] - downsampled_map.shape[1]

        # Generate random padding widths for each side
        # np.random.seed(0)  # you can set the seed to make the random numbers reproducible
        random_pad_y1 = np.random.randint(0, pad_width_y + 1)
        random_pad_y2 = pad_width_y - random_pad_y1
        random_pad_x1 = np.random.randint(0, pad_width_x + 1)
        random_pad_x2 = pad_width_x - random_pad_x1

        # Compute the padding for each dimension
        pad_width = ((random_pad_y1, random_pad_y2), (random_pad_x1, random_pad_x2))

        # Apply the padding
        expanded_map = np.pad(downsampled_map, pad_width, mode='constant', constant_values=0)

        return expanded_map
    
    
    def erode_and_add_obstacles(self, map, erosion_probability=0.3, obstacle_probability=0.1, erosion_times=2):
        map_width, map_height = map.shape

        # Define the area of interest
        start_x, end_x, start_y, end_y = 5, 55, 5, 55

        # Randomly add obstacles within the area of interest
        mask = np.random.rand(map_width, map_height) < obstacle_probability
        mask[:start_x, :] = mask[end_x:, :] = mask[:, :start_y] = mask[:, end_y:] = False
        map = np.where(mask & (map == 1), 0, map)

        for _ in range(erosion_times):
            # Create boolean masks for the map borders and obstacles within the area of interest
            top_mask = (map[start_x:end_x-1, start_y:end_y] == 0) & (map[start_x+1:end_x, start_y:end_y] == 1)
            bottom_mask = (map[start_x+1:end_x, start_y:end_y] == 0) & (map[start_x:end_x-1, start_y:end_y] == 1)
            left_mask = (map[start_x:end_x, start_y:end_y-1] == 0) & (map[start_x:end_x, start_y+1:end_y] == 1)
            right_mask = (map[start_x:end_x, start_y+1:end_y] == 0) & (map[start_x:end_x, start_y:end_y-1] == 1)

            # Combine all border cells into a single boolean mask
            border_mask = np.zeros_like(map, dtype=bool)
            border_mask[start_x:end_x-1, start_y:end_y][top_mask] = True
            border_mask[start_x+1:end_x, start_y:end_y][bottom_mask] = True
            border_mask[start_x:end_x, start_y:end_y-1][left_mask] = True
            border_mask[start_x:end_x, start_y+1:end_y][right_mask] = True

            # Erode border cells with a given probability
            erosion_mask = (np.random.rand(map_width, map_height) < erosion_probability) & border_mask
            map[erosion_mask] = 1

        return map
