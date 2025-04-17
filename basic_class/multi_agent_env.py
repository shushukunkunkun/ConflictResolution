import math
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
import sys  # 导入numpy用于数值计算  # 导入map_setting模块中的函数
from basic_class.uav_control import UAVControl 
# 设定other_function所在路径 (假设其在my_project/other_scripts/下)
my_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
other_function_path = os.path.join(my_project_root, 'scripts')
sys.path.append(other_function_path)# 假设UAVControl定义在另一个文件中
from scripts.other_function import discretize_circle, discretize_square, discretize_line,compute_shortest_distance
import matplotlib
# 设置matplotlib的非交互式后端


class MultiAgentEnv:

    def __init__(self, args):
        self.col_dis = 5
        # self.start_point = np.array([[200, 60], [200, -60], [200, 0]])
        # self.target_point = np.array([[900, 60], [900, -60], [900, 0]])
        self.start_point = np.array([[50, 750], [750, 750], [50, 50], [750, 50]])
        self.target_point = np.array([[500, 300], [300, 300], [500, 500], [300, 500]])
        self.device = args.device
        if args.map_filename == 'NarrowCorridor_3':
            line1_points = [(-50, -100), (850, -100)]
            # line2_points = [(-50, 100), (850, 100)]
            line2_points = [(-50, 100), (250, 100), (375, 0), (575, 0),
                            (700, 100), (850, 100)]
        if args.map_filename == 'NarrowCorridor_2':
            # line1_points = [(-50, -100), (850, -100)]
            # line2_points = [(-50, 100), (850, 100)]
            # 离散化各个障碍物的点
            line1_points = discretize_line([(200, 20), (600, 20), (500, 250), (150, 150), (200, 20)])
            Square1 = discretize_square([700, 300], 30)
            Square2 = discretize_square([620, 450], 80)
            Square3 = discretize_square([700, 600], 30)
            Triangle = discretize_line([(400, 600), (600, 750), (200, 750), (400, 600)])
            Circle1 = discretize_circle([100, 550], 60)
            Circle2 = discretize_circle([150, 350], 50)
            Wall = discretize_line([(0, 0), (0, 800), (800, 800), (800, 0), (0, 0)])
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
                device=torch.device("cuda")) for i in range(args.n_agents)
        ]

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
        elif self.method == 'sac':
            # TODO sac action space
            for action in actions:
                control_input = self.agent[0].action_space[action]
                control_inputs.append(control_input)
        elif self.method == 'None':
            control_inputs = actions
        for i, (agent,
                control_input) in enumerate(zip(self.agents, control_inputs)):
            agent.update_state(control_input)
        for i, (agent,
                control_input) in enumerate(zip(self.agents, control_inputs)):
            other_agent = self.agents[1 - i]
            agent.update_relative_state(other_agent, self.obstacle_coor)
        for i, (agent,
                control_input) in enumerate(zip(self.agents, control_inputs)):
            next_state = agent.get_state()
            next_state = np.array(next_state, dtype=float).flatten()
            flat_next_state = np.expand_dims(next_state, axis=0)
            next_states.append(flat_next_state)
            reward = self.calculate_reward(agent, gamma)
            rewards.append(reward)
        # for i in range(len(self.agents)):
        #     print(fr'Uav{i},位置{self.agents[i].position},速度{self.agents[i].velocity}')

        # 在循环外部检查终止条件
        if self.method != 'None':
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
                if all(np.linalg.norm(agent.position - agent.target_pos) <= (self.col_dis + 5)
                for agent in self.agents):
                    done = True
            # 检查所有 agent 是否超出边界（这里只检查 x 方向）
            if not done:
                for agent in self.agents:
                    if agent.position[0] <= -50 or agent.position[0] >= 920:
                        done = True
                        break
            for agent in self.agents:
                if np.linalg.norm(agent.position -
                                  agent.target_pos) <= (self.col_dis + 5):
                    agent.Gotta_Go = True
        return next_states, rewards, done

    def get_global_state(self):

        global_state = []
        if self.method == 'qmix':
            index_of_negative_relative_pos_agent = None
            for i, agent in enumerate(self.agents):
                if agent.get_state()[2] <= 0:
                    index_of_negative_relative_pos_agent = i
                    break
            if index_of_negative_relative_pos_agent is not None:
                negative_agent_state = self.agents[
                    index_of_negative_relative_pos_agent].get_state()
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
                    index_of_negative_relative_pos_agent].get_state()
            else:
                print(self.agents[0].get_state()[2])
                print(self.agents[1].get_state()[2])
                raise ValueError(
                    "No agent with a negative or zero third state value was found."
                )

            global_state.extend(
                negative_agent_state[:2])  # 添加 target_relative_pos
            global_state.extend(negative_agent_state[2:4])  # 添加 relative_pos
            global_state.extend(negative_agent_state[4:6])  # 添加 relative_vel
            global_state.extend(negative_agent_state[6:8])  # 添加 pos2obs
            global_state.append(
                negative_agent_state[8])  # 添加 relative_ellipse_length
            global_state.append(
                negative_agent_state[9])  # 添加 another_relative_ellipse_length

            another_agent_state = self.agents[
                1 - index_of_negative_relative_pos_agent].get_state()
            global_state.extend(
                another_agent_state[:2])  # 添加 another_target_relative_pos
            global_state.extend(another_agent_state[6:8])  # 添加 another_pos2obs

            return np.array(global_state, dtype=float)
        elif self.method == 'sac':
            print('Oops something went wrong!')
        elif self.method == 'None':
            for agent in self.agents:
                agent_data = [
                    agent.position[0], agent.position[1], agent.velocity[0],
                    agent.velocity[1], agent.ellipse_length, agent.theta
                ]
                global_state.extend(agent_data)
            return np.array(global_state, dtype=float)

    def calculate_reward(self, agent, decay_gamma):
        """
                V5.0
                智能体简单学会基本任务后  需要更进一步优化
                去除reward_shaping
                加入优化项
                加入速度方向变化惩罚
                加入gamma 时间惩罚
                V5.1
                加入时间惩罚gamma  下调了惩罚的权重
                V5.2
                加入reward_shaping
                V5.3
                下调了惩罚的权重
        """
        alpha, beta, gamma, kappa = 1, 1, 1, 1  # 奖励权重
        normilized_scale = 100
        if np.linalg.norm(agent.position - agent.target_pos) <= (self.col_dis +
                                                                 5):
            distance_reward = 2000 / normilized_scale
        else:
            distance_to_target = np.linalg.norm(agent.position -
                                                agent.target_pos)
            distance_reward = 10 * (
                np.linalg.norm(agent.last_pos - agent.target_pos) -
                distance_to_target) / normilized_scale

        ellipse_change_penalty = -20 * abs(
            agent.ellipse_length -
            agent.last_ellipse_length) / normilized_scale
        collision_penalty_now = 0
        collision_penalty_last = 0
        if (agent.dis2obstacle * agent.perception_radius) <= 5:
            collision_penalty_now = (
                -2 * (5 - (agent.dis2obstacle * agent.perception_radius))**
                2) / normilized_scale
        if (agent.last_dis2obstacle * agent.perception_radius) <= 5:
            collision_penalty_last = (
                -2 * (5 - (agent.last_dis2obstacle * agent.perception_radius))
                **2) / normilized_scale
        if self.is_collision(agent) == True:
            collision_penalty = -2000 / normilized_scale
        else:
            collision_penalty = decay_gamma * collision_penalty_now - collision_penalty_last  # reward shaping
            # collision_penalty = collision_penalty_now
        """
        计算两个矩形各自两点之间距离的最小值,通过判断这个最小距离计算penalty
        """
        other_agent = self.agents[1 - agent.index]
        min_dis_now, min_dis_last = compute_shortest_distance(
            agent, other_agent)
        overlap_penalty_now = 0
        overlap_penalty_last = 0
        if min_dis_now <= 3:
            overlap_penalty_now = (-2 *
                                   (3 - min_dis_now)**2) / normilized_scale
        if min_dis_last <= 3:
            overlap_penalty_last = (-2 *
                                    (3 - min_dis_last)**2) / normilized_scale
        if self.is_overlap() == True:
            overlap_penalty = -1000 / normilized_scale
        else:
            overlap_penalty = decay_gamma * overlap_penalty_now - overlap_penalty_last  # reward shaping
            # overlap_penalty = overlap_penalty_now
        lazy_penalty = 0
        if agent.position[0] <= -50 or agent.position[0] >= 920:  ##限制边界
            lazy_penalty = -1000 / normilized_scale
        if self.agents[agent.index].Gotta_Go == False:
            reward = alpha * distance_reward + beta * ellipse_change_penalty + gamma * collision_penalty + kappa * overlap_penalty + lazy_penalty
        else:
            reward = 0
        return reward

    def calculate_global_reward(self, gamma):
        total_reward = sum(
            [self.calculate_reward(agent, gamma) for agent in self.agents])
        return total_reward

    def is_collision(self, agent):
        # 椭圆的中心位置
        cx, cy = agent.position
        # 椭圆的长轴半径和短轴半径
        a = agent.ellipse_length  # 长轴半径
        b = (agent.ellipse_area / (math.pi * a))  # 根据面积计算短轴半径
        for obstacle in self.obstacle_coor:
            ox, oy = obstacle
            # 计算障碍物相对于椭圆中心的位置
            if ((ox - cx)**2) / (a**2) + ((oy - cy)**2) / (b**2) <= 1:
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
        lhs1 = ((xv_flat - h1) / a1)**2 + ((yv_flat - k1) / b1)**2
        lhs2 = ((xv_flat - h2) / a2)**2 + ((yv_flat - k2) / b2)**2

        # 判断是否存在同时位于两个椭圆内的点
        overlap = np.any((lhs1 <= 1) & (lhs2 <= 1))

        return overlap
