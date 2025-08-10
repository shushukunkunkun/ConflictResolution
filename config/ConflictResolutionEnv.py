'''
Author: Shukun
Date: 2025-03-28 15:44:06
LastEditors: Shukun
LastEditTime: 2025-08-10 19:41:35
Description: test the new personality
'''
import json
import math
import os
import sys
import numpy as np
from gym.spaces import Box
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 返回上一级目录，即项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from xuance.environment import RawMultiAgentEnv

from scripts.other_function import normalization,compute_shortest_distance,is_collision,is_overlap,discretize_circle, discretize_square, discretize_line,compute_outside_length,numpy_to_list
class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        # 定义一些无法在yaml中实现的东西
        self.env_config = env_config
        # self.env_config.test_mode 即可判断
        self.args = env_config.args
        self.args.max_theta = math.pi / 2
        self.args.normalization_scale = np.array([-5, -5, -90, -90, 60, math.pi/2])
        self.args.normalization_scale4state= np.array([5, 5, -900, -900, 60, math.pi/2])
        self.args.start_point = np.array([[50, 750], [750, 750], [50, 50], [750, 50]])
        self.args.target_point = np.array([[475, 325], [325, 325], [475, 475], [325, 475]])
        if self.args.map_filename == 'NarrowCorridor_3':
            line1_points = [(-50, -100), (850, -100)]
            # line2_points = [(-50, 100), (850, 100)]
            line2_points = [(-50, 100), (250, 100), (375, 0), (575, 0),
                            (700, 100), (850, 100)]
        if self.args.map_filename == 'NarrowCorridor_2':
            # 离散化各个障碍物的点
            line1_points = discretize_line(
                [(200, 20), (600, 20), (500, 250), (150, 150), (200, 20)]
            )
            Square1 = discretize_square([700, 150], 30)
            Square2 = discretize_square([620, 450], 80)
            Square3 = discretize_square([650, 650], 50)
            Triangle = discretize_line([(400, 600), (600, 750), (200, 750), (400, 600)])
            Circle1 = discretize_circle([100, 550], 60)
            Circle2 = discretize_circle([150, 350], 50)
            Wall = discretize_line([(0, 0), (0, 800), (800, 800), (800, 0), (0, 0)])
            # 合并所有障碍物点
            obstacle_coor = np.vstack((line1_points, Square1, Square2, Square3, Triangle, Circle1, Circle2, Wall))
        self.args.obstacle_coor = obstacle_coor
        self.env_id = env_config.env_id
        self.num_agents = 4
        self.num_robots = 10
        # self.neighbor_attentionnetwork = AttentionNetwork(6, self.args.embed_dim4neighbor, self.args.num_heads4neighbor, 6)
        # self.obs_attentionnetwork = AttentionNetwork(2, self.args.embed_dim4obs, self.args.num_heads4obs, 2)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agents_dis2obs = {agent: 30.0 for i,agent in enumerate(self.agents)}
        self.agents_working_state = {agent: 'working' for agent in self.agents}
        self.nearest_neighbor = {agent: None for i,agent in enumerate(self.agents)}
        self.agents_last_dis2obs = self.agents_dis2obs.copy()
        self.agents_state = {agent: np.concatenate((np.array([0.0, 0.0]),self.args.start_point[i].flatten(),np.array([30.0, 0.0]))) for i,agent in enumerate(self.agents)}
        self.agents_last_state = self.agents_state.copy()
        self.neighbor_state = {agent: np.zeros(12) for agent in self.agents}
        self.obs_state = {agent: np.zeros(2) for agent in self.agents}
        self.agents_obs = {agent: np.concatenate((normalization(self.agents_state[agent], self.args.normalization_scale4state, np.array([0.0,0.0,-self.args.target_point[i][0],-self.args.target_point[i][1],0.0,0.0])),self.neighbor_state[agent],self.obs_state[agent])) for i,agent in enumerate(self.agents)}
        # 每个智能体的观测为20维度  包括 self-state[Velocity,Position,Ellipse-length,Theta],neighbor-visablestate[Relative-velocity,Relative-position,Ellipse-length,Theta],obs[Position]
        # 定义单个智能体的观测低维和高维边界
        single_agent_low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0])
        single_agent_high = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        # 全局状态空间为所有智能体观测空间的拼接
        self.state_space = Box(
            low=np.concatenate([single_agent_low] * self.num_agents),
            high=np.concatenate([single_agent_high] * self.num_agents),
            dtype=np.float32
        )
        # 每个智能体的观测空间
        self.observation_space = {
            agent: Box(low=single_agent_low, high=single_agent_high, dtype=np.float32)
            for agent in self.agents
        }
        # 每个智能体的动作空间
        self.action_space = {
            agent: Box(
                low=np.array([-4.0, -4.0, 15.0, -math.pi/2]),
                high=np.array([4.0, 4.0, 60.0, math.pi/2]),
                dtype=np.float32
            )
            for agent in self.agents
        }
        self.max_episode_steps = 1200
        self.sampling_time = float(0.2)
        self._current_step = 0
        self.attention = env_config.attention
        if self.env_config.test_mode:
            self.agents_state_whole_process = np.zeros((self.max_episode_steps,self.num_agents,8)) #八个维度依次为 vx,vy,x,y,ax,ay,L,θ
            # agents_reward_list 主要记载每个时刻下每个智能体的reward细则
            self.agents_reward_list = [
            [ {} for _ in range(self.num_agents) ]
            for _ in range(self.max_episode_steps)]
            if self.env_config.with_robot:
                from basic_class.GroundRobot import GroundRobot
                np.random.seed(1)
                self.robots = {}
                for uav_index in range(self.num_agents):
                    ground_robots = []
                    for i in range(self.num_robots):
                        position = np.random.uniform(low=-20.0, high=20.0, size=2) + self.args.start_point[uav_index]
                        robot = GroundRobot(self.args.obstacle_coor, robot_id= uav_index * self.num_robots + i, position=position)
                        ground_robots.append(robot)
                    self.robots[f'agent{uav_index}'] = ground_robots
                self.robots_state_whole_process = np.zeros((self.max_episode_steps,self.num_agents*self.num_robots,6)) #六个维度依次为 vx,vy,x,y,ax,ay
    def pay_attention(self):
        "Updating the Observation of Each Agent Using the Attention Mechanism"
        "Neighbor_Attention"
        for i,agent in enumerate(self.agents):
            state_i = self.agents_state[agent]
            # 提取位置
            pos_i = state_i[2:4]  # 这里假设位置存储在第 3 和第 4 个元素
            visible_neighbors = []
            min_neighbor_distance = float('inf')  # 初始化最近距离为无穷大
            second_min_neighbor_distance = float('inf')
            neighbor_state = []
            # 遍历所有其他智能体，寻找邻居
            nearest_neighbor = None
            second_nearest_neighbor = None
            # 遍历所有其他智能体，寻找邻居
            for j, other_agent in enumerate(self.agents):
                if other_agent == agent:
                    continue  # 忽略自己

                state_j = self.agents_state[other_agent]
                pos_j = state_j[2:4]
                # 计算欧氏距离
                distance = np.linalg.norm(pos_i - pos_j)
                # 如果距离小于等于感知半径，则认为 other_agent 是可见邻居
                if distance <= self.args.perception_radius:
                    visible_neighbors.append(other_agent)

                    # 先检查是否比“当前最近”更近
                    if distance < min_neighbor_distance:
                        # 原来的最近变成第二近
                        second_min_neighbor_distance = min_neighbor_distance
                        second_nearest_neighbor = nearest_neighbor

                        # 更新最近
                        min_neighbor_distance = distance
                        nearest_neighbor = other_agent

                    # 否则，如果不属于最近并且比“当前第二近”更近，就更新第二近
                    elif distance < second_min_neighbor_distance:
                        # 注意：这里不需要额外判断 distance == min_neighbor_distance，因为 distance < min 的情况已排除
                        second_min_neighbor_distance = distance
                        second_nearest_neighbor = other_agent
            self.nearest_neighbor[agent] = nearest_neighbor
            # 将考虑的邻居拓展为两个
            nearest_neighbor_state = normalization(self.agents_state[nearest_neighbor],self.args.normalization_scale,np.concatenate((-state_i[:4].flatten(), np.array([0, 0])))) if nearest_neighbor != None else np.zeros(6)
            second_nearest_neighbor_state = normalization(self.agents_state[second_nearest_neighbor],self.args.normalization_scale,np.concatenate((-state_i[:4].flatten(), np.array([0, 0])))) if second_nearest_neighbor != None else np.zeros(6)
            neighbor_state = np.concatenate((nearest_neighbor_state,second_nearest_neighbor_state))
            # # 将列表堆叠成一个张量：形状 (num_neighbors, 6)
            # neighbor_tensor = torch.stack(neighbor_state, dim=0)
            # # 增加 batch 维度，变成 (1, num_neighbors, 6)
            # neighbor_tensor = neighbor_tensor.unsqueeze(0)
            # # 使用注意力网络进行前向传播，输出形状为 (1, 6)
            self.neighbor_state[agent] = neighbor_state
            # self.neighbor_state[agent] = self.neighbor_attentionnetwork(neighbor_tensor).squeeze(0).detach().cpu().numpy()
        "Obstacle_Attention"
        for i, agent in enumerate(self.agents):
            # 获取当前智能体状态，state 格式为 [x_vel, y_vel, x_pos, y_pos, L, theta]
            state_i = self.agents_state[agent]
            pos_i = state_i[2:4]  
            min_obs = None
            # 筛选出在感知范围内的障碍物
            obstacles_in_range = []
            min_obs_distance = float('inf')  # 初始设置为正无穷
            for obs in self.args.obstacle_coor:  
                distance = np.linalg.norm(pos_i-obs)#这里的dis计算设计为距离实际椭圆的距离
                if distance <= self.args.perception_radius:
                    obstacles_in_range.append(obs)
            for obs in obstacles_in_range:
                dis = compute_outside_length(state_i,obs)
                if  dis < min_obs_distance:
                        min_obs_distance = dis
                        min_obs = obs
            # 如果没有障碍物在范围内，使用一个零向量作为默认值
            if len(obstacles_in_range) == 0:
                # obstacle_state = [torch.zeros(2)]\
                obstacle_state = np.zeros(2)
                min_obs_distance = 30
            else:
                obstacle_state = normalization(min_obs,np.array([self.args.perception_radius,self.args.perception_radius]),-pos_i)
                # 将每个障碍物坐标转换为 torch 张量
                # obstacle_state = [torch.tensor(normalization(obs,np.array([self.args.perception_radius,self.args.perception_radius]),-pos_i), dtype=torch.float32) for obs in obstacles_in_range]
            # # 将列表堆叠成一个张量：形状 (num_filtered_obs, 2)
            # obstacle_tensor = torch.stack(obstacle_state, dim=0)
            # # 增加 batch 维度，变成 (1, num_filtered_obs, 2)
            # obstacle_tensor = obstacle_tensor.unsqueeze(0)
            # # 使用障碍物注意力网络进行前向传播，输出形状为 (1, 2)
            # aggregated_obs = self.obs_attentionnetwork(obstacle_tensor)
            # 将输出转换为 NumPy 数组，并去除 batch 维度
            self.agents_last_dis2obs[agent] = self.agents_dis2obs[agent]
            self.agents_dis2obs[agent] = min_obs_distance
            self.obs_state[agent] = obstacle_state
            # aggregated_obs.squeeze(0).detach().cpu().numpy()
    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        # 按照离目标距离（即 self.agents_obs[agent][2:4] 的模长）从近到远排序
        sorted_agents = sorted(self.agents, key=lambda agent: np.linalg.norm(self.agents_obs[agent][2:4]))

        # 根据排序顺序，将各 agent 的观测拼接成全局状态
        global_state = np.concatenate([self.agents_obs[agent] for agent in sorted_agents], axis=0)
        return global_state

    def reset(self):
        self.agents_state = {agent: np.array([0.0, 0.0, self.args.start_point[i][0],self.args.start_point[i][1], 30, 0.0]) for i,agent in enumerate(self.agents)}
        self.neighbor_state = {agent: np.zeros(12) for agent in self.agents}
        self.obs_state = {agent: np.zeros(2) for agent in self.agents}
        self.agents_obs = {agent: np.concatenate((normalization(self.agents_state[agent], self.args.normalization_scale4state, np.array([0.0,0.0,-self.args.target_point[i][0],-self.args.target_point[i][1],0.0,0.0])),self.neighbor_state[agent],self.obs_state[agent])) for i,agent in enumerate(self.agents)}
        self.agents_working_state = {agent: 'working' for agent in self.agents}
        self.agents_dis2obs = {agent: 30.0 for i,agent in enumerate(self.agents)}
        self.agents_last_dis2obs = self.agents_dis2obs.copy()
        observation = {agent: self.agents_obs[agent] for agent in self.agents}
        info = {}
        self._current_step = 0
        return observation, info

    def step(self, action_dict):
        # 首先根据Action进行状态更新
        if self.env_config.render == True:
            self.render(self.env_config.render_mode)
        # 第一步更新自身状态
        self.update_agents_state(action_dict)
        # 第二步根据自身状态更新hidden_state(incluidng obs_hidden_state and neighbor_state)
        self.pay_attention()
        if self.env_config.with_robot and self.env_config.test_mode:
            for ii in range(self.num_agents):
                for i in range(self.num_robots):
                    robot = self.robots[f'agent{ii}'][i]
                    all_robots = [r for robots in self.robots.values() for r in robots]
                    robot.update_control_input(all_robots,
                                               self.robots[f'agent{ii}'],
                                               self.agents_state[f'agent_{ii}'][2:4],
                                               self.agents_state[f'agent_{ii}'][0:2],
                                               self.agents_state[f'agent_{ii}'][4],
                                               900/self.agents_state[f'agent_{ii}'][4],
                                               self.agents_state[f'agent_{ii}'][5],
                                               self.sampling_time
                                               )
                    robot.update_state(self.sampling_time)
        # 第三步根据自身状态组合self.obs_state
        self.agents_obs = {agent: np.concatenate((np.array(normalization(self.agents_state[agent], self.args.normalization_scale4state, np.array([0,0,-self.args.target_point[i][0],-self.args.target_point[i][1],0,0]))),self.neighbor_state[agent],self.obs_state[agent])) for i,agent in enumerate(self.agents)}
        
        observation = {agent: self.agents_obs[agent] for agent in self.agents}
        rewards = {agent: self.calculate_reward(agent) for agent in self.agents}
        terminated = {agent: True if self.agents_working_state[agent] != 'working' else False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        self._current_step += 1
        return observation, rewards, terminated, truncated, info
    def calculate_reward(self,agent):
        "Required Information Representation"
        index = self.agents.index(agent)
        state = 'working'
        current_state = self.agents_state[agent]
        last_state = self.agents_last_state[agent]
        pos_current = current_state[2:4]
        pos_last = last_state[2:4]
        ellipse_length_current = current_state[4]
        ellipse_length_last = last_state[4]
        theta_current = current_state[5]
        theta_last = last_state[5]
        dis2obstacle = self.agents_dis2obs[agent]
        last_dis2obstacle = self.agents_last_dis2obs[agent]
        target = self.args.target_point[index].flatten() if hasattr(self.args.target_point[index], 'flatten') else self.args.target_point[index]
        alpha, beta, gamma, kappa, epsilon = 1, 3, 3, 1, 1  # 奖励权重
        normilized_scale = 100
        if self.agents_working_state[agent] != 'working' and self.env_config.test_mode:
            self.agents_reward_list[self._current_step][index] = {
                    'distance_reward': 0,
                    'ellipse_change_penalty': 0,
                    'collision_penalty': 0 ,
                    'overlap_penalty': 0,
                    'theta_change_penalty': 0,
                    'sum_reward': 0
                }
            return 0
        if self.agents_working_state[agent] == 'working':
        
            "(1) Positive rewards for approaching the target"
            if np.linalg.norm(pos_current - target) <= (self.args.col_dis + 25):
                distance_reward = 2000 / normilized_scale
                state = 'success'
            else:
                distance_to_target = np.linalg.norm(pos_current - target)
                distance_reward = 100 * (np.linalg.norm(pos_last - target) - distance_to_target) / normilized_scale
            "(2) Elliptical planning deformation penalty"
            # ellipse_change_penalty = -10 * abs(ellipse_length_current - ellipse_length_last) / normilized_scale
            ellipse_change_penalty_now = -abs(ellipse_length_current - (900/ellipse_length_current))/45
            # ellipse_change_penalty_last = -abs(ellipse_length_last - (900/ellipse_length_last))/45
            # ellipse_change_penalty = self.args.decay_gamma * ellipse_change_penalty_now - ellipse_change_penalty_last 
            
            # reward shaping
            ellipse_change_penalty = ellipse_change_penalty_now
            theta_change_penalty = -2*(abs(theta_current - theta_last))/np.pi
            
            "(3) Outside collision penalty"
            collision_penalty_now = 0
            collision_penalty_last = 0
            if dis2obstacle  <= 10:
                collision_penalty_now = -10*((10 - dis2obstacle)**2) / normilized_scale
                # if dis2obstacle == 0:
                #     print(f'{agent} crash!')
            if last_dis2obstacle <= 10:
                collision_penalty_last = -10 * (10 - last_dis2obstacle)**2 / normilized_scale
            if  is_collision(self.agents_state[agent],self.args.obstacle_coor) == True:
                collision_penalty = -1000 / normilized_scale
                state = 'dead'
            else:
                collision_penalty = self.args.decay_gamma * collision_penalty_now - collision_penalty_last  # reward shaping
                # collision_penalty = collision_penalty_now 
            "(4) Interior collision penalty"
            "Internal collision penalty should be related to the area of the interaction"
            other_agent = self.nearest_neighbor[agent]
            overlap_penalty = 0
            overlap = False
            if other_agent is not None:
                overlap, intersection_area = is_overlap(self.agents_state[agent],self.agents_state[other_agent])
                if  overlap:
                    "If they overlap, the penalties should be added to the penalties related to the charging area"
                    if intersection_area <= (100 * np.pi):
                        overlap_penalty = -2
                    elif intersection_area >(100 * np.pi) and intersection_area <= (450 * np.pi):
                        overlap_penalty = -5
                    elif intersection_area >(450 * np.pi):
                        overlap_penalty = -10
            reward = alpha * distance_reward + beta * ellipse_change_penalty + gamma * collision_penalty + kappa * overlap_penalty + epsilon * theta_change_penalty
        else:
            reward = 0
        if self.env_config.test_mode:
            self.agents_reward_list[self._current_step][index] = {
                'distance_reward': alpha * distance_reward,
                'ellipse_change_penalty':beta * ellipse_change_penalty,
                'collision_penalty':gamma * collision_penalty ,
                'overlap_penalty':kappa * overlap_penalty,
                'theta_change_penalty':epsilon * theta_change_penalty,
                'sum_reward': reward
            }
        self.agents_working_state[agent] = state
        return reward
    def update_agents_state(self, action_dict):
        """
        根据 action_dict 更新每个代理的状态。
        state 格式：[x_vel, y_vel, pos (二维), L, theta]
        action 格式:(acc,new_L,new_theta)
        """
        # 默认Actor最后一层使用sigmod函数  所以要先将action_dict与实际动作进行映射
        for agent in self.agents:
            low = self.action_space[agent].low
            high = self.action_space[agent].high
            # 将 (0,1) 输出映射到实际动作空间
            actual_action = low + action_dict[agent] * (high - low)
            action_dict[agent] = actual_action
        self.agents_last_state = self.agents_state.copy()
        dt = self.sampling_time  # 采样时间
        for agent in self.agents:
            if self.agents_working_state[agent] == 'success':
                self.agents_state[agent][:2] = np.array([0,0])
            if self.agents_working_state[agent] != 'working':
                continue
            acc_x, acc_y , new_L, new_theta = action_dict[agent]
            state = self.agents_state[agent]
            x_vel, y_vel = state[0], state[1]
            pos = state[2:4]  # 二维位置
            L = state[4]
            theta = state[5]
            # 1. 更新速度（假设加速度沿当前角度方向作用）
            new_x_vel = x_vel + acc_x * dt
            new_y_vel = y_vel + acc_y * dt 
            
            current_speed = np.linalg.norm(np.array([new_x_vel, new_y_vel]))
            if current_speed > self.args.max_speed:
                scaling_factor = self.args.max_speed / current_speed
                new_x_vel *= scaling_factor
                new_y_vel *= scaling_factor
            new_pos = pos + np.array([new_x_vel, new_y_vel]) * dt
            new_state = np.concatenate((
                np.array([new_x_vel, new_y_vel]),
                new_pos,
                np.array([new_L, new_theta])
            ))
            self.agents_state[agent] = new_state
            if self.env_config.test_mode:
                for i,agent in enumerate(self.agents):
                    self.agents_state_whole_process[self._current_step,i,:] = np.concatenate((self.agents_state[agent],np.array([acc_x , acc_y])))
                    if self.env_config.with_robot:
                        for ii in range(self.num_robots):
                            pos_r, vel_r, acc_r = self.robots[f'agent{i}'][ii].position,self.robots[f'agent{i}'][ii].velocity,self.robots[f'agent{i}'][ii].acceleration
                            self.robots_state_whole_process[self._current_step,i*self.num_robots+ii,:] = np.concatenate((pos_r,vel_r,acc_r))
    def render(self, mode='human', close=False):
        """
        Renders the current state of the environment, drawing obstacles, each agent's ellipse, and its target point (color-consistent).

        Request:
        - self.args.obstacle_coor: NumPy array, shaped (n, 2), representing obstacle coordinates.
        - self.args.target_point: A list or array, each element is a [x, y] coordinate, and the target point corresponds to the agent.
        - self.agents_state: A dictionary with the state of each agent as a one-dimensional NumPy array, where:
            state[2] and state[3] are the central locations of the agent;
            state[4] is the semi-major axis of the ellipse;
            state[5] is the elliptic inclination in radians.
            The elliptical semi-minor axis is given by 900 / state[4] (state[4] is not 0).
        """
        # 初始化 figure 和 axis
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        ax = self.ax
        ax.clear()

        # 绘制障碍物（红色）
        ax.scatter(self.args.obstacle_coor[:, 0], self.args.obstacle_coor[:, 1],
                color='red', s=20, label='Obstacles')

        # 定义颜色列表，用于区分不同 agent
        colors = ['b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink', 'gray']

        # 遍历每个 agent，绘制其椭圆和对应目标点
        for i, (agent, state) in enumerate(self.agents_state.items()):
            # 选择颜色
            color = colors[i % len(colors)]
            # agent 中心坐标
            center = (state[2], state[3])
            # 椭圆参数
            a = state[4]
            b = 900 / a if a != 0 else 0
            angle_deg = state[5] * 180 / math.pi  # 转换为度数

            # 绘制椭圆：width 和 height 为整个轴长
            ellipse = Ellipse(
                xy=center,
                width=2 * a,
                height=2 * b,
                angle=angle_deg,
                edgecolor=color,
                facecolor='none',
                linewidth=2,
                label=f'Agent {agent} Region'  # 可以根据需要定制标签
            )
            ax.add_patch(ellipse)
            # 绘制 agent 中心（与椭圆颜色一致）
            ax.scatter(center[0], center[1], color=color, s=50)
            # 绘制目标点：假设每个 agent 对应的目标点在 self.args.target_point 中，按顺序对应
            target = np.array(self.args.target_point[i])
            ax.scatter(target[0], target[1], color=color, s=50, marker='*', 
                    label=f'Agent {agent} Target')

        # ax.legend()
        ax.set_title("Environment Render")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # 设置 x 和 y 轴刻度一致，避免形状失真
        ax.axis('equal')
        if mode == 'human':
            plt.draw()
            plt.pause(0.01)
            if close:
                plt.close(self.fig)
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
    def save_data(self):
        """
        保存训练数据到指定目录。如果指定目录不存在，则保存到默认目录。
        
        参数:
        - path_on_f_drive (str): F盘路径,用于保存数据。
        - replay_buffer: 包含所有训练数据的 replay buffer。
        - Args: 包含 map_filename 的对象，用于生成文件名。
        """
        map_filename = self.env_config.args.map_filename
        path_on_f_drive = fr"{self.env_config.save_path}/{map_filename}_train_data/{self.env_config.method}"
        filename_state = f"all_episodes_data_{map_filename}.json"
        filename_robots = f"{map_filename}_robots.json"
        filename_reward = f"reward_data_{map_filename}.json"
        filename_obs = f"{map_filename}_obs.json"
        filename_target = f"{map_filename}_target.json"
        # 默认保存目录
        default_save_dir = "all_train_data"

        if os.path.exists(path_on_f_drive):
            # 如果 F 盘路径存在，则保存到 F 盘
            save_path_state = os.path.join(path_on_f_drive, filename_state)
            save_path_robots = os.path.join(path_on_f_drive, filename_robots)
            save_path_obs = os.path.join(path_on_f_drive, filename_obs)
            save_path_target = os.path.join(path_on_f_drive, filename_target)
            save_path_reward = os.path.join(path_on_f_drive, filename_reward)
            print(f"Saving data to {save_path_state} on F drive.")
        else:
            # 如果 F 盘路径不存在，则保存到当前目录
            save_path_state = os.path.join(default_save_dir, filename_state)
            save_path_robots = os.path.join(default_save_dir, filename_robots)
            save_path_obs = os.path.join(default_save_dir, filename_obs)
            save_path_target = os.path.join(default_save_dir, filename_target)
            save_path_reward = os.path.join(default_save_dir, filename_reward)
            print(
                f"{path_on_f_drive} path not found. Saving data to {save_path_state} in current directory."
            )
        # 确保路径存在
        os.makedirs(os.path.dirname(save_path_state), exist_ok=True)
        
        # 获取所有经验数据并转换格式
        state_data = numpy_to_list(self.agents_state_whole_process)
        if self.env_config.with_robot:
            robots_data = numpy_to_list(self.robots_state_whole_process)
        reward_data = numpy_to_list(self.agents_reward_list)
        target_data = numpy_to_list(self.args.target_point)
        obs_data = numpy_to_list(self.args.obstacle_coor)
        # 将数据保存到指定路径
        with open(save_path_state, "w") as f:
            json.dump(state_data, f)
        with open(save_path_reward, "w") as f:
            json.dump(reward_data, f)
        with open(save_path_obs, "w") as f:
            json.dump(obs_data, f)
        with open(save_path_target, "w") as f:
            json.dump(target_data, f)
        if self.env_config.with_robot:
            with open(save_path_robots, "w") as f:
                json.dump(robots_data, f)  
        print(f"数据已保存到 {save_path_state}")
    def close(self):
        if self.env_config.test_mode:
            # 尾部填充
            mask = np.any(self.agents_state_whole_process != 0, axis=(1,2))
            if np.any(mask):
                last_valid = np.where(mask)[0][-1]
                if last_valid + 1 < self.agents_state_whole_process.shape[0]:
                    self.agents_state_whole_process[last_valid+1:] = (
                        self.agents_state_whole_process[last_valid])
                if self.env_config.with_robot:
                    mask = np.any(self.robots_state_whole_process != 0, axis=(1,2))
                    if np.any(mask):
                        last_valid = np.where(mask)[0][-1]
                        if last_valid + 1 < self.robots_state_whole_process.shape[0]:
                            self.robots_state_whole_process[last_valid+1:] = (
                                self.robots_state_whole_process[last_valid])
            self.save_data()
        return