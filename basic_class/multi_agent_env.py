import math
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
import sys  # 导入numpy用于数值计算  # 导入map_setting模块中的函数
from basic_class.uav_control import UAVControl

# 设定other_function所在路径 (假设其在my_project/other_scripts/下)
my_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
other_function_path = os.path.join(my_project_root, "scripts")
sys.path.append(other_function_path)  # 假设UAVControl定义在另一个文件中
from scripts.other_function import (
    compute_outside_length,
    discretize_circle,
    discretize_square,
    discretize_line,
    compute_shortest_distance,
    is_collision,
    is_overlap,
    normalization,
)
import matplotlib

# 设置matplotlib的非交互式后端


class MultiAgentEnv:

    def __init__(self, args):
        self.col_dis = 5
        # self.start_point = np.array([[200, 60], [200, -60], [200, 0]])
        # self.target_point = np.array([[900, 60], [900, -60], [900, 0]])

        self.device = args.device
        if args.map_filename == "NarrowCorridor_3":
            self.start_point = np.array([[50, 350], [350, 350], [50, 50], [350, 50]])
            self.target_point = np.array([[250, 150], [150, 150], [250, 250], [150, 250]])
            line1_points = discretize_line([
                    (100, 10),
                    (300, 10),
                    (250, 125),
                    (75, 75),
                    (100, 10)
                ])

            Square1 = discretize_square([350, 150], 15)
            Square2 = discretize_square([310, 225], 40)
            Square3 = discretize_square([350, 300], 15)

            Triangle = discretize_line([
                (200, 300),
                (300, 375),
                (100, 375),
                (200, 300)
            ])

            Circle1 = discretize_circle([50, 275], 30)
            Circle2 = discretize_circle([75, 175], 25)

            Wall = discretize_line([(0, 0), (0, 400), (400, 400), (400, 0), (0, 0)])
            obstacle_coor = np.vstack(
            (
                line1_points,
                Square1,
                Square2,
                Square3,
                Triangle,
                Circle1,
                Circle2,
                Wall,
            )
                )
        if args.map_filename == "NarrowCorridor_2":
            # 离散化各个障碍物的点
            self.start_point = np.array([[50, 750], [750, 750], [50, 50], [750, 50]])
            self.target_point = np.array([[500, 300], [300, 300], [500, 500], [300, 500]])
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
        self.method = args.method
        # 离散化折线
        self.obstacle_coor = obstacle_coor
        self.agents = [
            UAVControl.create_uav(
                method=args.method,
                epsilon=args.epsilon,  # 关键字参数
                index=i,
                position=self.start_point[i],
                velocity=[0, 0],
                target_pos=self.target_point[i],
                obs_coor=self.obstacle_coor,  # 关键字参数
                map = args.map_filename,
                device=torch.device("cuda"),
            )
            for i in range(args.n_agents)
        ]
        if self.method == "None":
            if args.map_filename == "NarrowCorridor_2":
                self.normalization_scale = np.array([-5, -5, -90, -90, 60, math.pi / 2])
                self.normalization_scale4state = np.array(
                    [5, 5, -900, -900, 60, math.pi / 2]
                )
            elif args.map_filename == "NarrowCorridor_3":
                self.normalization_scale = np.array([-5, -5, -90, -90, 40, math.pi / 2])
                self.normalization_scale4state = np.array(
                    [5, 5, -900, -900, 40, math.pi / 2]
                )
            self.num_agents = 4
            self.uavs = [f"agent_{i}" for i in range(self.num_agents)]
            self.agents_dis2obs = {agent: 30.0 for i, agent in enumerate(self.uavs)}
            self.agents_working_state = {agent: "working" for agent in self.uavs}
            self.nearest_neighbor = {agent: None for i, agent in enumerate(self.uavs)}
            self.agents_last_dis2obs = self.agents_dis2obs.copy()
            self.agents_state = {
                agent: np.concatenate(
                    (
                        np.array([0.0, 0.0]),
                        self.start_point[i].flatten(),
                        np.array([math.sqrt(self.agents[0].ellipse_area/math.pi), 0.0]),
                    )
                )
                for i, agent in enumerate(self.uavs)
            }
            self.agents_last_state = self.agents_state.copy()
            self.neighbor_state = {agent: np.zeros(12) for agent in self.uavs}
            self.obs_state = {agent: np.zeros(2) for agent in self.uavs}
            self.agents_obs = {
                agent: np.concatenate(
                    (
                        normalization(
                            self.agents_state[agent],
                            self.normalization_scale4state,
                            np.array(
                                [
                                    0.0,
                                    0.0,
                                    -self.target_point[i][0],
                                    -self.target_point[i][1],
                                    0.0,
                                    0.0,
                                ]
                            ),
                        ),
                        self.neighbor_state[agent],
                        self.obs_state[agent],
                    )
                )
                for i, agent in enumerate(self.uavs)
            }
            # 每个智能体的观测为14维度  包括 self-state[Velocity,Position,Ellipse-length,Theta],neighbor-visablestate[Relative-velocity,Relative-position,Ellipse-length,Theta],obs[Position]
            # 定义单个智能体的观测低维和高维边界
            self.max_episode_steps = 1200
            self.sampling_time = float(0.2)
            self.decay_gamma = 0.95
    def reset(self):
        for i, agent in enumerate(self.agents):
            agent.reset(self.start_point[i])
        for i, agent in enumerate(self.agents):
            other_agent = self.agents[1 - i]
            agent.update_relative_state(other_agent, self.obstacle_coor)

        return [agent.get_state() for agent in self.agents]

    def step(self, actions, gamma):
        next_states = []
        rewards = []
        control_inputs = []
        done = False
        # TODO Change the original agent.update_state input directly from the action space index to the control quantity
        if self.method == "qmix":
            for action in actions:
                control_input = self.agent[0].action_space[action]
                control_inputs.append(control_input)
        elif self.method == "sac":
            # TODO sac action space
            for action in actions:
                control_input = self.agent[0].action_space[action]
                control_inputs.append(control_input)
        elif self.method == "None":
            control_inputs = actions
        for i, (agent, control_input) in enumerate(zip(self.agents, control_inputs)):
            Agent = f'agent_{i}'
            if self.agents_working_state[Agent]!='working':
                continue
            agent.update_state(control_input)
        # for i, (agent, control_input) in enumerate(zip(self.agents, control_inputs)):
        #     other_agent = self.agents[1 - i]
        #     agent.update_relative_state(other_agent, self.obstacle_coor)
        for i, (agent, control_input) in enumerate(zip(self.agents, control_inputs)):
            next_state = agent.get_state()
            next_state = np.array(next_state, dtype=float).flatten()
            flat_next_state = np.expand_dims(next_state, axis=0)
            next_states.append(flat_next_state)
            reward = self.calculate_reward(agent, gamma)
            rewards.append(reward)
        # for i in range(len(self.agents)):
        #     print(fr'Uav{i},位置{self.agents[i].position},速度{self.agents[i].velocity}')

        # 在循环外部检查终止条件
        if self.method != "None":
            done = False
            # 检查每个 agent 是否发生碰撞
            for agent in self.agents:
                if self.is_collision(agent):
                    done = True
                    break
            if not done and self.is_overlap():
                done = True

            # 当所有 agent 都接近目标时，认为任务完成
            if not done:
                if all(
                    np.linalg.norm(agent.position - agent.target_pos)
                    <= (self.col_dis + 25)
                    for agent in self.agents
                ):
                    done = True
            # 检查所有 agent 是否超出边界（这里只检查 x 方向）
            if not done:
                for agent in self.agents:
                    if agent.position[0] <= -50 or agent.position[0] >= 920:
                        done = True
                        break
            for agent in self.agents:
                if np.linalg.norm(agent.position - agent.target_pos) <= (
                    self.col_dis + 15
                ):
                    agent.Gotta_Go = True
        return next_states, rewards, done
    def get_global_state4draw(self):
        global_state = []
        for agent in self.agents:
                agent_data = [
                    agent.position[0], agent.position[1], agent.velocity[0],
                    agent.velocity[1], agent.ellipse_length, agent.theta
                ]
                global_state.extend(agent_data)
        return np.array(global_state, dtype=float)
    def get_global_state(self):

        global_state = []
        if self.method == "qmix":
            index_of_negative_relative_pos_agent = None
            for i, agent in enumerate(self.agents):
                if agent.get_state()[2] <= 0:
                    index_of_negative_relative_pos_agent = i
                    break
            if index_of_negative_relative_pos_agent is not None:
                negative_agent_state = self.agents[
                    index_of_negative_relative_pos_agent
                ].get_state()
            else:
                print(self.agents[0].get_state()[2])
                print(self.agents[1].get_state()[2])
                raise ValueError(
                    "No agent with a negative or zero third state value was found."
                )
            global_state.extend(negative_agent_state[:2])
            index_of_negative_relative_pos_agent = None
            for i, agent in enumerate(self.agents):
                if agent.get_state()[2] <= 0:
                    index_of_negative_relative_pos_agent = i
                    break

            if index_of_negative_relative_pos_agent is not None:
                negative_agent_state = self.agents[
                    index_of_negative_relative_pos_agent
                ].get_state()
            else:
                print(self.agents[0].get_state()[2])
                print(self.agents[1].get_state()[2])
                raise ValueError(
                    "No agent with a negative or zero third state value was found."
                )

            global_state.extend(negative_agent_state[:2])  # 添加 target_relative_pos
            global_state.extend(negative_agent_state[2:4])  # 添加 relative_pos
            global_state.extend(negative_agent_state[4:6])  # 添加 relative_vel
            global_state.extend(negative_agent_state[6:8])  # 添加 pos2obs
            global_state.append(negative_agent_state[8])  # 添加 relative_ellipse_length
            global_state.append(
                negative_agent_state[9]
            )  # 添加 another_relative_ellipse_length

            another_agent_state = self.agents[
                1 - index_of_negative_relative_pos_agent
            ].get_state()
            global_state.extend(
                another_agent_state[:2]
            )  # 添加 another_target_relative_pos
            global_state.extend(another_agent_state[6:8])  # 添加 another_pos2obs

            return np.array(global_state, dtype=float)
        elif self.method == "sac":
            print("Oops something went wrong!")
        elif self.method == "None":
            """Returns the global state of the environment."""
            # 按照离目标距离（即 self.agents_obs[agent][2:4] 的模长）从近到远排序
            sorted_agents = sorted(self.uavs, key=lambda agent: np.linalg.norm(self.agents_obs[agent][2:4]))
            # 根据排序顺序，将各 agent 的观测拼接成全局状态
            global_state = np.concatenate([self.agents_obs[agent] for agent in sorted_agents], axis=0)
            return np.array(global_state, dtype=float)

    def calculate_reward(self, agent, decay_gamma):
        if self.method != 'None':
            alpha, beta, gamma, kappa = 1, 1, 1, 1  # 奖励权重
            normilized_scale = 100
            if np.linalg.norm(agent.position - agent.target_pos) <= (self.col_dis + 5):
                distance_reward = 2000 / normilized_scale
            else:
                distance_to_target = np.linalg.norm(agent.position - agent.target_pos)
                distance_reward = (
                    10
                    * (
                        np.linalg.norm(agent.last_pos - agent.target_pos)
                        - distance_to_target
                    )
                    / normilized_scale
                )

            ellipse_change_penalty = (
                -20
                * abs(agent.ellipse_length - agent.last_ellipse_length)
                / normilized_scale
            )
            collision_penalty_now = 0
            collision_penalty_last = 0
            if (agent.dis2obstacle * agent.perception_radius) <= 5:
                collision_penalty_now = (
                    -2 * (5 - (agent.dis2obstacle * agent.perception_radius)) ** 2
                ) / normilized_scale
            if (agent.last_dis2obstacle * agent.perception_radius) <= 5:
                collision_penalty_last = (
                    -2 * (5 - (agent.last_dis2obstacle * agent.perception_radius)) ** 2
                ) / normilized_scale
            if self.is_collision(agent) == True:
                collision_penalty = -2000 / normilized_scale
            else:
                collision_penalty = (
                    decay_gamma * collision_penalty_now - collision_penalty_last
                )  # reward shaping
                # collision_penalty = collision_penalty_now
            """
            计算两个矩形各自两点之间距离的最小值,通过判断这个最小距离计算penalty
            """
            other_agent = self.agents[1 - agent.index]
            min_dis_now, min_dis_last = compute_shortest_distance(agent, other_agent)
            overlap_penalty_now = 0
            overlap_penalty_last = 0
            if min_dis_now <= 3:
                overlap_penalty_now = (-2 * (3 - min_dis_now) ** 2) / normilized_scale
            if min_dis_last <= 3:
                overlap_penalty_last = (-2 * (3 - min_dis_last) ** 2) / normilized_scale
            if self.is_overlap() == True:
                overlap_penalty = -1000 / normilized_scale
            else:
                overlap_penalty = (
                    decay_gamma * overlap_penalty_now - overlap_penalty_last
                )  # reward shaping
                # overlap_penalty = overlap_penalty_now
            lazy_penalty = 0
            if agent.position[0] <= -50 or agent.position[0] >= 920:  ##限制边界
                lazy_penalty = -1000 / normilized_scale
            if self.agents[agent.index].Gotta_Go == False:
                reward = (
                    alpha * distance_reward
                    + beta * ellipse_change_penalty
                    + gamma * collision_penalty
                    + kappa * overlap_penalty
                    + lazy_penalty
                )
            else:
                reward = 0
        else:
            "Required Information Representation"
            agent = f"agent_{agent.index}"
            index = self.uavs.index(agent)
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
            target = self.target_point[index].flatten() if hasattr(self.target_point[index], 'flatten') else self.target_point[index]
            alpha, beta, gamma, kappa, epsilon = 1, 5, 3, 1, 1  # 奖励权重
            normilized_scale = 100
            if self.agents_working_state[agent] != 'working':
                return 0
            " (1) Positive rewards for approaching the target"
            if np.linalg.norm(pos_current - target) <= (self.col_dis + 25):
                distance_reward = 2000 / normilized_scale
                state = 'success'
            else:
                distance_to_target = np.linalg.norm(pos_current - target)
                distance_reward = 100 * (np.linalg.norm(pos_last - target) - distance_to_target) / normilized_scale
            "(2) Elliptical planning deformation penalty"
           
            # ellipse_change_penalty = -10 * abs(ellipse_length_current - ellipse_length_last) / normilized_scale
            ellipse_change_penalty_now = -abs(ellipse_length_current - (900/ellipse_length_current))/45
            # ellipse_change_penalty_last = -abs(ellipse_length_last - (900/ellipse_length_last))/45
            # ellipse_change_penalty = self.decay_gamma * ellipse_change_penalty_now - ellipse_change_penalty_last  # reward shaping
            ellipse_change_penalty = ellipse_change_penalty_now
            theta_change_penalty = -2*(abs(theta_current - theta_last))/np.pi
            
            "(3) Outside collision penalty"
            collision_penalty_now = 0
            collision_penalty_last = 0
            if dis2obstacle  <= 10:
                collision_penalty_now = -10*((10 - dis2obstacle)**2) / normilized_scale
            if last_dis2obstacle <= 10:
                collision_penalty_last = (-10 * (10 - last_dis2obstacle)**2) / normilized_scale
            if  is_collision(self.agents[0].ellipse_area,self.agents_state[agent],self.obstacle_coor) == True:
                collision_penalty = -1000 / normilized_scale
                state = 'dead'
            else:
                collision_penalty = self.decay_gamma * collision_penalty_now - collision_penalty_last  # reward shaping
                # collision_penalty = collision_penalty_now 
            "(4) Interior collision penalty"
            "Internal collision penalty should be related to the area of the interaction"
            other_agent = self.nearest_neighbor[agent]
            overlap_penalty = 0
            if other_agent is not None:
                overlap, intersection_area = is_overlap(self.agents[0].ellipse_area,self.agents_state[agent],self.agents_state[other_agent])
                if  overlap:
                    "If they overlap, the penalties should be added to the penalties related to the charging area"
                    if intersection_area <= (100 * np.pi):
                        overlap_penalty = -2
                    elif intersection_area >(100 * np.pi) and intersection_area <= (450 * np.pi):
                        overlap_penalty = -5
                    elif intersection_area >(450 * np.pi):
                        overlap_penalty = -10
            if self.agents_working_state[agent] == 'working':
                reward = alpha * distance_reward + beta * ellipse_change_penalty + gamma * collision_penalty + kappa * overlap_penalty + epsilon * theta_change_penalty
            else:
                reward = 0
            if reward < -10:
                print(f"agent{agent},reward:{reward},distance_reward:{alpha * distance_reward},ellipse_change_penalty:{beta * ellipse_change_penalty},collision_penalty{gamma * collision_penalty},  overlap_penalty:{kappa * overlap_penalty},theta_change_penalty:{epsilon * theta_change_penalty}")
            self.agents_working_state[agent] = state
        return reward

    def calculate_global_reward(self, gamma):
        total_reward = sum(
            [self.calculate_reward(agent, gamma) for agent in self.agents]
        )
        return total_reward
    def pay_attention(self):
        "Updating the Observation of Each Agent Using the Attention Mechanism"
        "Neighbor_Attention"
        for i,agent in enumerate(self.uavs):
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
        for j, other_agent in enumerate(self.uavs):
            if other_agent == agent:
                continue  # 忽略自己
            state_j = self.agents_state[other_agent]
            pos_j = state_j[2:4]
            # 计算欧氏距离
            distance = np.linalg.norm(pos_i - pos_j)

            # 如果距离小于等于感知半径，则认为 other_agent 是可见邻居
            if distance <= self.agents[0].perception_radius:
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
            nearest_neighbor_state = normalization(self.agents_state[nearest_neighbor],self.normalization_scale,np.concatenate((-state_i[:4].flatten(), np.array([0, 0])))) if nearest_neighbor != None else np.zeros(6)
            second_nearest_neighbor_state = normalization(self.agents_state[second_nearest_neighbor],self.normalization_scale,np.concatenate((-state_i[:4].flatten(), np.array([0, 0])))) if second_nearest_neighbor != None else np.zeros(6)
            neighbor_state = np.concatenate((nearest_neighbor_state,second_nearest_neighbor_state))
            # # 将列表堆叠成一个张量：形状 (num_neighbors, 6)
            # neighbor_tensor = torch.stack(neighbor_state, dim=0)
            # # 增加 batch 维度，变成 (1, num_neighbors, 6)
            # neighbor_tensor = neighbor_tensor.unsqueeze(0)
            # # 使用注意力网络进行前向传播，输出形状为 (1, 6)
            self.neighbor_state[agent] = neighbor_state
        "Obstacle_Attention"
        for i, agent in enumerate(self.uavs):
            # 获取当前智能体状态，state 格式为 [x_vel, y_vel, x_pos, y_pos, L, theta]
            state_i = self.agents_state[agent]
            pos_i = state_i[2:4]  
            min_obs = None
            # 筛选出在感知范围内的障碍物
            obstacles_in_range = []
            min_obs_distance = float('inf')  # 初始设置为正无穷
            for obs in self.obstacle_coor:  
                distance = np.linalg.norm(pos_i-obs)#这里的dis计算设计为距离实际椭圆的距离
                if distance <= self.agents[0].perception_radius:
                    obstacles_in_range.append(obs)
            for obs in obstacles_in_range:
                dis = compute_outside_length(state_i,obs)
                if  dis < min_obs_distance:
                        min_obs_distance = dis
                        min_obs = obs
            # 如果没有障碍物在范围内，使用一个零向量作为默认值
            if len(obstacles_in_range) == 0:
                obstacle_state = np.zeros(2)
                min_obs_distance = 30
            else:
                obstacle_state = normalization(min_obs,np.array([self.agents[0].perception_radius,self.agents[0].perception_radius]),-pos_i)
            self.agents_last_dis2obs[agent] = self.agents_dis2obs[agent]
            self.agents_dis2obs[agent] = min_obs_distance
            self.obs_state[agent] = obstacle_state
    def is_collision(self, agent):
        # 椭圆的中心位置
        cx, cy = agent.position
        # 椭圆的长轴半径和短轴半径
        a = agent.ellipse_length  # 长轴半径
        b = agent.ellipse_area / (math.pi * a)  # 根据面积计算短轴半径
        for obstacle in self.obstacle_coor:
            ox, oy = obstacle
            # 计算障碍物相对于椭圆中心的位置
            if ((ox - cx) ** 2) / (a**2) + ((oy - cy) ** 2) / (b**2) <= 1:
                return True  # 障碍物在椭圆内部
        return False  # 没有障碍物在椭圆内部

    def is_overlap(self):
        # 假设您的环境中有两个智能体
        agent1 = self.agents[0]
        agent2 = self.agents[1]

        # 获取第一个椭圆的参数
        h1, k1 = agent1.position  # 椭圆1的中心坐标 (x, y)
        a1 = agent1.ellipse_length  # 椭圆1的长轴半径（半轴长度）
        b1 = agent1.ellipse_area / (math.pi * a1)  # 椭圆1的短轴半径，利用面积公式计算

        # 获取第二个椭圆的参数
        h2, k2 = agent2.position  # 椭圆2的中心坐标
        a2 = agent2.ellipse_length  # 椭圆2的长轴半径
        b2 = agent2.ellipse_area / (math.pi * a2)  # 椭圆2的短轴半径

        # 计算两个椭圆外接矩形的重叠区域
        x_min = max(h1 - a1, h2 - a2)
        x_max = min(h1 + a1, h2 + a2)
        y_min = max(k1 - b1, k2 - b2)
        y_max = min(k1 + b1, k2 + b2)

        # 如果外接矩形不重叠，则椭圆一定不重叠
        if x_min >= x_max or y_min >= y_max:
            return False

        # 在重叠区域内创建采样网格
        num_samples = 100  # 采样点数，可根据需要调整
        x_samples = np.linspace(x_min, x_max, num_samples)
        y_samples = np.linspace(y_min, y_max, num_samples)
        xv, yv = np.meshgrid(x_samples, y_samples)
        xv_flat = xv.flatten()
        yv_flat = yv.flatten()

        # 计算采样点在椭圆方程中的值
        lhs1 = ((xv_flat - h1) / a1) ** 2 + ((yv_flat - k1) / b1) ** 2
        lhs2 = ((xv_flat - h2) / a2) ** 2 + ((yv_flat - k2) / b2) ** 2

        # 判断是否存在同时位于两个椭圆内的点
        overlap = np.any((lhs1 <= 1) & (lhs2 <= 1))

        return overlap
