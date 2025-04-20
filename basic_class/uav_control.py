"""
Author: Shukun
Date: 2024-12-30 20:11:57
LastEditors: Shukun
LastEditTime: 2024-12-31 15:02:34
Description: UAV类的定义
"""
import itertools
import numpy as np
import math
import torch
from abc import ABC, abstractmethod
from scripts.other_function import (
    discretize_line,
    generate_ellipse_points,
    process_draw_state,
    calculate_angle,
    filter_ellipse_points_by_angle,
    calculate_obstacle_distances,
    ellipse_radius_in_direction,
    ellipses_minimum_distance,
)
import casadi as ca  # 导入 CasADi 库，用于非线性优化


class UAVControlBase:

    def __init__(
            self,
            method,
            epsilon,
            index,
            position,
            velocity,
            target_pos,
            obs_coor,
            device=torch.device("cpu"),
    ):
        """
        Here I need to define a traditional method as well as a specific reinforcement learning algorithm
        For subsequent comparison and parallel use, the attributes should be divided into traditional method attributes and RL attributes.
        """
        # 公有基础属性
        self.method = method
        self.index = index
        self.position = np.array(position, dtype=float)
        self.last_pos = self.position
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.ellipse_length = 30  # 椭圆长轴
        self.last_ellipse_length = 30  # 椭圆上一时刻长轴
        self.max_ellipse_length = 60  # 椭圆长轴的最大长度
        self.min_ellipse_length = 15
        self.last_theta = 0  # 上一时刻椭圆的倾斜角
        self.theta = 0  # 椭圆的倾斜角
        self.max_theta = math.pi / 2  # 椭圆的最大倾斜角
        self.target_pos = np.array(target_pos, dtype=float)
        self.max_acceleration = 4.0  # 最大加速度
        self.max_speed = 5.0  # 最大速度
        self.perception_radius = 90.0  # 感知半径
        self.ellipse_area = 900.0 * math.pi  # 椭圆面积
        self.communication_radius = 900  # 通讯半径
        self.obs_coor = obs_coor  # 障碍物坐标
        self.dis2obstacle = 0  #距离最近障碍物的距离
        self.target_relative_pos = np.zeros(2, dtype=float)  # 无人机与目标的相对位置
        self.relative_vel = np.zeros(2, dtype=float)  # 无人机相对归一化速度
        self.relative_ellipse_length = (
            self.ellipse_length / self.max_ellipse_length)  # 无人机归一化后的椭圆长轴
        self.epsilon = epsilon
        self.device = device
        # 传统方法属性
        self.sampling_time = float(1)  # 采样时间
        self.horizon = 5  # MPC的预测序列长度
        self.traditional_acceleration = np.zeros(2, dtype=float)
        self.traditional_length = np.zeros(2, dtype=float)
        self.traditional_theta = 0

    @abstractmethod
    def select_action(self, *args, **kwargs):
        """
        Abstract method, subclasses must implement it.
        And ensure that the input parameters of the calling method of the subclass are consistent
        """
        pass

    def calculate_action(self, all_agents):
        """
        A decentralized MPC is used to calculate the agent's acceleration, elliptical major axis length, and inclination control.
        
        Returns:
            acceleration (2D),
            ellipse_length (scalar),
            theta (scalar)
        """

        # 1. 感知可见邻居与障碍物
        visible_neighbors, visible_obstacles,min_obs = self.sense_neighbors_and_obstacles(
            all_agents)

        # ============ 参数设置 ============
        H = self.horizon  # 预测时域长度
        dt = self.sampling_time  # 离散时间步
        w_pos = 0.5  # 位置误差权
        w_acc = 0  # 加速度权重
        w_length = 10  # 椭圆长轴变化权重
        w_theta = 1000 # 椭圆角度变化权重
        w_obs = -5000 # 外部避障惩罚权重
        w_n = -10000 # 内部避碰权重
        w_round = 0.5 #尽量保持为圆
        d_obs = 8.0  # 避碰安全距离
        d_n = 10 # 内部避碰距离
        s_obs_des = 25
        s_n_des = 10
        # ============ 定义符号变量 ============
        # 加速度 u[t] (2D)，长度控制 l_control[t] (1D)，倾角控制 theta_control[t] (1D)
        u = ca.SX.sym("u", 2, H)  # shape: (H,2)
        l_control = ca.SX.sym("l_control", H)  # shape: (H,)
        theta_control = ca.SX.sym("theta_control", H)  # shape: (H,)

        # 位置 x[t] (2D)，速度 v[t] (2D)，椭圆长轴 l[t]，倾角 theta[t]
        x = ca.SX.sym("x", 2, H + 1)  # 
        v = ca.SX.sym("v", 2, H + 1)  # 
        l = ca.SX.sym("l", H + 1)  # shape: (H+1,)
        theta = ca.SX.sym("theta", H + 1)  # shape: (H+1,)
        # 将外部避障与内部避障约束放在惩罚项中
        s_obs = ca.SX.sym("s_obs", 1) 
        s_n = ca.SX.sym("s_n", 1) 

        # ============ 约束列表，目标函数 ============
        # CasADi 中，约束统一放在列表 g 里
        # 等式约束:  g_i = 0
        # 不等式约束: g_i >= 0
        g = []
        g_types = []  # 用于区分等式/不等式（方便设置 lbg, ubg）

        # ============ 1) 初始状态等式约束 ============
        # x[0] == self.position
        for i in range(2):
            g.append(x[i, 0] - self.position[i])
            g_types.append("eq")
        # v[0] == self.velocity
        for i in range(2):
            g.append(v[i, 0] - self.velocity[i])
            g_types.append("eq")
        # l[0] == self.ellipse_length
        g.append(l[0] - self.ellipse_length)
        g_types.append("eq")
        # theta[0] == self.theta
        g.append(theta[0] - self.theta)
        g_types.append("eq")

        # ============ 2) 动力学等式约束 ============
        # x[t+1] = x[t] + v[t] * dt
        # v[t+1] = v[t] + u[t] * dt
        # l[t+1] = l_control[t]
        # theta[t+1] = theta_control[t]
        for t in range(H):
            for i in range(2):
                # x[t+1, i] - (x[t, i] + v[t, i]*dt) = 0
                g.append(x[i, t + 1] - (x[i, t] + v[i, t] * dt))
                g_types.append("eq")
            for i in range(2):
                # v[t+1, i] - (v[t, i] + u[t, i]*dt) = 0
                g.append(v[i, t + 1] - (v[i, t] + u[i, t] * dt))
                g_types.append("eq")
            # l[t+1] - l_control[t] = 0
            g.append(l[t + 1] - l_control[t])
            g_types.append("eq")
            # theta[t+1] - theta_control[t] = 0
            g.append(theta[t + 1] - theta_control[t])
            g_types.append("eq")
        """Here it is found that if this constraint is added to constraint g, it becomes invalid
        But there is no problem in the constraints of variables
        This could be a bug
        And what I've learned is that for a variable that is directly constrained, it should be placed directly in the variable constraint."""
        # ============ 3) 输入限幅 & 状态边界 (不等式约束) ============
        # # 3.1) 加速度范数: ||u[t]||2 <= self.max_acceleration
        # #     转化为 self.max_acceleration - ||u[t]||2 >= 0
        # for t in range(H):
        #     acc_norm = ca.sqrt(u[t, 0]**2 + u[t, 1]**2)
        #     g.append(self.max_acceleration - acc_norm)
        #     g_types.append("ineq")

        # # 3.2) 长轴上/下限: l_min <= l[t] <= l_max
        # #     l[t] - l_min >= 0,  l_max - l[t] >= 0
        # for t in range(H + 1):
        #     g.append(l[t] - self.min_ellipse_length)  # >= 0
        #     g_types.append("ineq")
        #     g.append(self.max_ellipse_length - l[t])  # >= 0
        #     g_types.append("ineq")

        # # 3.3) 倾角控制上/下限: -max_theta <= theta_control[t] <= max_theta
        # #     theta_control[t] + max_theta >= 0,   max_theta - theta_control[t] >= 0
        # for t in range(H):
        #     g.append(theta_control[t] + self.max_theta)  # >= 0
        #     g_types.append("ineq")
        #     g.append(self.max_theta - theta_control[t])  # >= 0
        #     g_types.append("ineq")

        # ============ 4) 碰撞约束 (不等式) ============
        # 1) 计算方向向量 direction = obs - x[t, :]
        # 2) 计算椭圆在该方向上的边界半径 r
        # 3) 约束：norm_2(direction) - r >= d_min
        #    即  norm_2(direction) - r - d_min >= 0     
        if min_obs is not None:
            for t in range(H + 1):
                    # direction
                    direction = min_obs - x[:, t]  #  (2,) shape in CasADi SX

                    # compute ellipse boundary radius in that direction
                    # a, b, theta 依你而定:
                    a_val = l[t]  # if you define l as the major axis length
                    b_val = (900 / l[t])  # or something if you store area

                    cos_t = ca.cos(theta[t])
                    sin_t = ca.sin(theta[t])

                    x_dir = direction[0] * cos_t + direction[1] * sin_t
                    y_dir = -direction[0] * sin_t + direction[1] * cos_t
                    denominator = (b_val * x_dir)**2 + (a_val * y_dir)**2
                    denominator = ca.fmax(denominator, 1e-8)
                    r = (a_val * b_val) / ca.sqrt(denominator)

                    distance_expr = ca.norm_2(direction) - r - d_obs
                    # g.append(distance_expr)
                    # g_types.append("ineq")
                    g.append(distance_expr - s_obs)  # >= 0
                    g_types.append("ineq")

        #   4.2) 与邻居智能体: ellipses_minimum_distance(...) >= d_min
        for neighbor in visible_neighbors:
            x_j, v_j, l_j, theta_j = neighbor.position, neighbor.velocity, neighbor.ellipse_length, neighbor.theta
            for t in range(H + 1):
                # 预测邻居位置
                x_j_pred = x_j + v_j * t * dt
                # 计算椭圆间最小距离
                distance_expr = ellipses_minimum_distance(
                    x[ :,t],
                    l[t],
                    theta[t],
                    self.ellipse_area,  # 这里你可根据自己需要传入椭圆面积或相关参数
                    x_j_pred,
                    l_j,
                    theta_j,
                    self.ellipse_area,
                ) - d_n
                # g.append(distance_expr)
                # g_types.append("ineq")
                g.append(distance_expr - s_n)  # >= 0
                g_types.append("ineq")

        # ============ 5) 构建目标函数 (Cost) ============
        cost = 0
        # 对每个时间步 t = 0..H-1 的阶段代价
        for t in range(H):
            # 位置误差
            pos_error = x[:,t + 1] - self.target_pos
            # (1,2) shape
            cost += w_pos * ca.norm_2(pos_error)**2
            # 加速度代价
            cost += w_acc * ca.norm_2(u[:,t])**2
            # 椭圆长轴变化
            cost += w_length * (l[t + 1] - l[t])**2
            # 倾角变化
            cost += w_theta * (theta[t + 1] - theta[t])**2
            # 期望保持为圆形
            cost += w_round * (l[t] - (900 / l[t]))**2
        # # 碰撞惩罚
        cost += w_obs * (s_obs - s_obs_des)  
        cost += w_n * (s_n - s_n_des)
        # 末端的额外位置误差
        pos_error_final = x[:,H] - self.target_pos
        cost += w_pos * ca.norm_2(pos_error_final)**2

        # ============ 6) 拼接优化变量 ============
        # 将所有符号变量按顺序堆叠到一个大的向量中
        nlp_x = ca.vertcat(x.reshape((-1, 1)), v.reshape((-1, 1)),
                           l.reshape((-1, 1)), theta.reshape((-1, 1)),
                           u.reshape((-1, 1)), l_control.reshape((-1, 1)),
                           theta_control.reshape((-1, 1)))
        nlp_x = ca.vertcat(nlp_x,s_obs,s_n)

        # ============ 7) 构建NLP问题字典 ============
        nlp = {
            "x": nlp_x,  # 优化变量
            "f": cost,  # 目标函数
            "g": ca.vertcat(*g),  # 所有约束
        }

        # ============ 8) 创建求解器 ============
        solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 0,  # 禁止 IPOPT 输出信息
                "print_time": 0,  # 禁止显示求解时间
                "ipopt.sb": "yes"  # 禁止 IPOPT 显示进度条
            })

        # ============ 9) 设置初始猜测 & 上下界 ============
        # 9.1) 计算各变量段的大小
        num_x = (H + 1) * 2
        num_v = (H + 1) * 2
        num_l = (H + 1)
        num_theta = (H + 1)
        num_u = H * 2
        num_l_control = H
        num_theta_control = H

        total_vars = num_x + num_v + num_l + num_theta + num_u + num_l_control + num_theta_control
        
        total_vars += 2

        # 9.2) 初始猜测
        #     (a) 位置、速度: 重复初始值
        #     (b) l, theta: 设为初始值
        #     (c) u, l_control, theta_control: 全部置0或初始状态
        x0 = np.concatenate([
            np.tile(self.position, H + 1),
            np.tile(self.velocity, H + 1),
            np.full(num_l, self.ellipse_length),
            np.full(num_theta, self.theta),
            np.zeros(num_u),
            np.full(num_l_control, self.ellipse_length),
            np.full(num_theta_control, self.theta),
            np.zeros(2)
        ])

        # 9.3) 优化变量上下界（如果需要的话）
        lbx = np.full(total_vars, -ca.inf)
        ubx = np.full(total_vars, ca.inf)
        #    - 加速度部分
        idx_u = num_x + num_v + num_l + num_theta
        lbx[idx_u:idx_u + num_u] = -self.max_acceleration
        ubx[idx_u:idx_u + num_u] = self.max_acceleration
        #    - 长轴控制
        idx_lctrl = idx_u + num_u
        lbx[idx_lctrl:idx_lctrl + num_l_control] = self.min_ellipse_length
        ubx[idx_lctrl:idx_lctrl + num_l_control] = self.max_ellipse_length
        #    - 倾角控制
        idx_tctrl = idx_lctrl + num_l_control
        lbx[idx_tctrl:idx_tctrl + num_theta_control] = -self.max_theta
        ubx[idx_tctrl:idx_tctrl + num_theta_control] = self.max_theta
        
        idx_s = idx_tctrl + 2

        lbx[idx_s:idx_s + 2] = 5
        ubx[idx_s:idx_s + 2] = ca.inf

        # ============ 10) 设置约束边界 ============
        # 根据 g_types 区分 eq 和 ineq
        # eq -> lbg=0, ubg=0
        # ineq -> lbg=0, ubg=inf
        lbg = []
        ubg = []
        for typ in g_types:
            if typ == "eq":
                lbg.append(0.0)
                ubg.append(0.0)
            else:  # "ineq"
                lbg.append(0.0)
                ubg.append(ca.inf)

        lbg = np.array(lbg, dtype=float)
        ubg = np.array(ubg, dtype=float)

        # ============ 11) 调用求解器 ============
        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        except Exception as e:
            print(f"Solver failed: {e}")
            return None, None, None

        # ============ 12) 提取最优解 ============
        sol_x = sol["x"].full().flatten()

        # 12.1) 把一大串解按对应维度切分出来
        idx = 0
        x_opt = sol_x[idx:idx + num_x].reshape((2,H + 1))
        idx += num_x
        v_opt = sol_x[idx:idx + num_v].reshape((2,H + 1))
        idx += num_v
        l_opt = sol_x[idx:idx + num_l]
        idx += num_l
        theta_opt = sol_x[idx:idx + num_theta]
        idx += num_theta
        u_opt = sol_x[idx:idx + num_u].reshape((2,H))
        idx += num_u
        l_control_opt = sol_x[idx:idx + num_l_control]
        idx += num_l_control
        theta_control_opt = sol_x[idx:idx + num_theta_control]
        idx += num_theta_control

        # ============ 13) 返回第一步控制 ============
        # 传统做法只取第一个时刻的控制作为动作
        self.traditional_acceleration = u_opt[0]
        self.traditional_length = l_control_opt[0]
        self.traditional_theta = theta_control_opt[0]
        # print(sol_x[idx_s:idx_s+2])

        return np.array(
            [u_opt[0][0],u_opt[0][1], l_control_opt[0], theta_control_opt[0]])

    def sense_neighbors_and_obstacles(self, all_agents):
        """
        According to their location, neighbors/obstacles within the obs_radius range are screened out from all_agents and obstacles
        Return:
          visible_neighbors: [neighbor1, ...]
          visible_obstacles: [obs_1, obs_2, ...]
        """
        visible_neighbors = []
        for agent in all_agents:
            if agent.index == self.index:
                continue
            dist = np.linalg.norm(agent.position - self.position)
            if dist <= self.perception_radius:
                visible_neighbors.append(agent)
        visible_obstacles = []
        min_obs = None
        min_dis = float("inf")
        for obs in self.obs_coor:
            if np.linalg.norm(obs - self.position) <= self.perception_radius:
                visible_obstacles.append(obs)
        for obs in visible_obstacles:
            direction = obs - self.position  #  (2,) shape in CasADi SX
            # compute ellipse boundary radius in that direction
            # a, b, theta 依你而定:
            a_val = self.ellipse_length  # if you define l as the major axis length
            b_val = (900 / a_val)  # or something if you store area
            cos_t = math.cos(self.theta)
            sin_t = math.sin(self.theta)
            x_dir = direction[0] * cos_t + direction[1] * sin_t
            y_dir = -direction[0] * sin_t + direction[1] * cos_t
            denominator = (b_val * x_dir)**2 + (a_val * y_dir)**2
            r = (a_val * b_val) / math.sqrt(denominator)
            if  np.linalg.norm(obs - self.position) - r <= min_dis:
                    min_dis = np.linalg.norm(obs - self.position) - r
                    min_obs = obs
        return visible_neighbors, visible_obstacles,min_obs

    def update_state(self, control_input):
        '''
        Parameters: 
        Return: 
        LastEditTime: 20250107
        Description: The long axis increment is no longer exported, but the new long axis is directly exported; Added control of inclination
        '''
        if self.Gotta_Go == False:
            self.acceleration = np.array(control_input[:2], dtype=float)

            if np.linalg.norm(self.acceleration) > self.max_acceleration:
                self.acceleration = self.acceleration * (
                    self.max_acceleration / np.linalg.norm(self.acceleration))

            self.last_pos = self.position
            self.velocity = np.array(self.velocity, dtype=np.float64)
            self.velocity += self.acceleration * self.sampling_time
            if np.linalg.norm(self.velocity) > self.max_speed:
                self.velocity = self.velocity * (self.max_speed /
                                                 np.linalg.norm(self.velocity))
            self.position = np.array(self.position, dtype=np.float64)
            self.position += self.velocity * self.sampling_time

            self.last_ellipse_length = self.ellipse_length
            self.ellipse_length = control_input[2]
            self.ellipse_length = min(self.ellipse_length,
                                      self.max_ellipse_length)
            self.ellipse_length = max(self.ellipse_length,
                                      self.min_ellipse_length)
            self.last_theta = self.theta
            self.theta = control_input[3]
            self.theta = min(self.theta, self.max_theta)
            self.theta = max(self.theta, -self.max_theta)
        else:
            self.velocity = np.array([0, 0], dtype=float)

    def update_relative_state(self, other_agent, obstacle_coor):
        self.target_relative_pos = (self.target_pos -
                                    self.position) / self.communication_radius
        self.relative_pos = (other_agent.position -
                             self.position) / self.communication_radius
        """
            V4.0 Modify modeling
            Modify relative speed
            Modify dis2obstacle and angle2obstacle to relative position forms
        """
        self.relative_vel = self.velocity / self.max_speed
        self.another_relative_ellipse_length = np.array(
            [other_agent.ellipse_length / other_agent.max_ellipse_length],
            dtype=float)  # 确保为一维数组
        self.relative_ellipse_length = np.array(
            [self.ellipse_length / self.max_ellipse_length],
            dtype=float)  # 确保为一维数组

        if np.linalg.norm(self.velocity) == 0:
            velocity_direction = np.array([1.0, 0.0])  # 如果速度为零，使用默认方向
        else:
            velocity_direction = self.velocity
        # 计算短半轴 b
        b = self.ellipse_area / (math.pi * self.ellipse_length)
        # 生成椭圆上的点
        ellipse_points = generate_ellipse_points(self.position[0],
                                                 self.position[1],
                                                 self.ellipse_length, b)
        # 筛选与速度方向夹角小于等于90度的点
        filtered_points = filter_ellipse_points_by_angle(
            ellipse_points, self.position, velocity_direction)
        filtered_obstacles = []
        # 筛选感知范围内的障碍物
        for obs in obstacle_coor:
            distance = np.linalg.norm(self.position - np.array(obs))
            if distance <= self.perception_radius:
                filtered_obstacles.append(obs)

        min_distance = float("inf")
        best_angle = None
        self.last_dis2obstacle = self.dis2obstacle
        for obs in filtered_obstacles:
            for point in filtered_points:
                v2 = obs - point
                v3 = obs - self.position
                distance = np.linalg.norm(v2)
                if distance < min_distance:
                    angle = calculate_angle(velocity_direction, v2)
                    min_distance = distance
                    best_angle = angle
                    best_pos2obs = v3
        if best_angle != None:
            self.angle2obstacle = best_angle / math.pi
            self.dis2obstacle = min_distance / self.perception_radius
            self.pos2obs = best_pos2obs / self.perception_radius
        else:
            self.angle2obstacle = 0
            self.dis2obstacle = 1
            self.pos2obs = np.array([1.0, 0])
        return (
            self.target_relative_pos,
            self.relative_pos,
            self.relative_vel,
            self.pos2obs,
            self.relative_ellipse_length,
            self.another_relative_ellipse_length,
        )

    def get_state(self):
        state = [
            *self.target_relative_pos,
            *self.relative_pos,
            *self.relative_vel,
            *self.pos2obs,
            float(self.relative_ellipse_length),
            float(self.another_relative_ellipse_length),
        ]
        return np.array(state, dtype=float)

    def reset(self, initial_pos):
        self.position = np.array(initial_pos, dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.acceleration = np.array([0, 0], dtype=float)
        self.ellipse_length = 30
        self.last_action = None
        self.Gotta_Go = False


class UAVControlQMIX(UAVControlBase):

    def __init__(
        self,
        method,
        epsilon,  # 添加 epsilon
        index,
        position,
        velocity,
        target_pos,
        obs_coor,  # 添加 obs_coor
        device=torch.device("cpu")):
        # Qmix属性
        super().__init__(method, epsilon, index, position, velocity,
                         target_pos, obs_coor, device)
        ax_values = [-self.max_acceleration, 0, self.max_acceleration]
        ay_values = [-self.max_acceleration, 0, self.max_acceleration]
        L_values = [-1, 0, 1]
        self.action_space = list(
            itertools.product(ax_values, ay_values, L_values))
        self.dis2obstacle = 1.0  # 距最近障碍物的距离
        self.last_dis2obstacle = self.dis2obstacle
        self.pos2obs = np.zeros(2, dtype=float)  # 无人机距障碍的相对归一化位置
        self.angle2obstacle = 0  # 距最近障碍物的夹角 弧度制
        self.another_relative_ellipse_length = 0  # 另一个椭圆的长轴归一化长度
        self.last_action = None  # 用于RNN选择动作

    def select_action(self,
                      state,
                      qmix=None,
                      n_agents=None,
                      n_actions=None,
                      step=None,
                      all_agents=None):
        """
        (ax_values, ay_values, L_values)
        :param state:
        :param qmix:
        :param n_agents:
        :param n_actions:
        :return:
        """
        if self.Gotta_Go == True:
            action = 13
            self.last_action = action
            return action
        '''
        Only Learn 4 me
        '''
        Cheating = True
        if Cheating == True:
            action = 13
            if step <= 250:
                action = 22
            if step > 250 and step < 277:
                if self.index == 0:
                    action = 19
                else:
                    action = 25
            if step >= 277 and step <= 285:
                action = 14
            if step >= 285 and step < 290:
                action = 22
            if step > 290 and step < 295:
                if self.index == 0:
                    action = 17
                else:
                    action = 11
            if step > 295 and step < 300:
                if self.index == 0:
                    action = 19
                else:
                    action = 25
            if step >= 300:
                if step % 2 == 1:
                    action = 19
                else:
                    action = 25
            '''
            700步之后random_sample
            '''
            if step >= 700:
                if np.random.rand() < self.epsilon:  # 以ε的概率选择随机动作
                    action = np.random.randint(0, len(self.action_space))
                    self.last_action = action
                else:
                    state = torch.tensor(state,
                                         dtype=torch.float32,
                                         device=qmix.device).unsqueeze(
                                             0)  # 确保state在正确的设备上
                    # 如果存在上一个动作，拼接动作的 one-hot 编码和智能体编号的 one-hot 编码
                    if self.last_action is not None:
                        agent_id_onehot = torch.zeros(n_agents,
                                                      device=qmix.device)
                        agent_id_onehot[self.index] = 1
                        action_onehot = torch.zeros(n_actions,
                                                    device=qmix.device)
                        action_onehot[self.last_action] = 1
                        combined_onehot = torch.cat(
                            [action_onehot,
                             agent_id_onehot])  # 拼接动作和智能体编号的 one-hot 编码
                    else:
                        # 如果是第一个动作，假设动作为全零的 one-hot 编码
                        agent_id_onehot = torch.zeros(n_agents,
                                                      device=qmix.device)
                        agent_id_onehot[self.index] = 1
                        action_onehot = torch.zeros(n_actions,
                                                    device=qmix.device)
                        combined_onehot = torch.cat(
                            [action_onehot,
                             agent_id_onehot])  # 拼接全零动作和智能体编号的 one-hot 编码

                    # 将 combined_onehot 拼接到 state 张量的最后一个维度
                    state = torch.cat(
                        [state, combined_onehot.unsqueeze(0)], dim=1)
                    ## 将state合并
                    '''
                    eval_hidden 只有在batch learn的时候会被更新  
                    在智能体选择动作的时候  其并没有更新
                    所以在智能体选择动作这里 
                    仍然要使用网络的动作argmax
                    而不能直接利用输出层
                    dammmmn!
                    '''
                    q_values = qmix.get_q_values_single(
                        state, self.index)  # 使用 eval_rnn 网络计算 Q 值  谁TM写的
                    # hidden_state = h.view(self.args.rnn_hidden_dim)
                    action = torch.argmax(q_values).item()  # 选择最大Q值对应的动作
            self.last_action = action
            return action
        else:
            if np.random.rand() < self.epsilon:  # 以ε的概率选择随机动作
                action = np.random.randint(0, len(self.action_space))
                self.last_action = action
            else:
                state = torch.tensor(state,
                                     dtype=torch.float32,
                                     device=qmix.device).unsqueeze(
                                         0)  # 确保state在正确的设备上
                # 如果存在上一个动作，拼接动作的 one-hot 编码和智能体编号的 one-hot 编码
                if self.last_action is not None:
                    agent_id_onehot = torch.zeros(n_agents, device=qmix.device)
                    agent_id_onehot[self.index] = 1
                    action_onehot = torch.zeros(n_actions, device=qmix.device)
                    action_onehot[self.last_action] = 1
                    combined_onehot = torch.cat(
                        [action_onehot,
                         agent_id_onehot])  # 拼接动作和智能体编号的 one-hot 编码
                else:
                    # 如果是第一个动作，假设动作为全零的 one-hot 编码
                    agent_id_onehot = torch.zeros(n_agents, device=qmix.device)
                    agent_id_onehot[self.index] = 1
                    action_onehot = torch.zeros(n_actions, device=qmix.device)
                    combined_onehot = torch.cat(
                        [action_onehot,
                         agent_id_onehot])  # 拼接全零动作和智能体编号的 one-hot 编码

                # 将 combined_onehot 拼接到 state 张量的最后一个维度
                state = torch.cat([state, combined_onehot.unsqueeze(0)], dim=1)
                ## 将state合并
                '''
                eval_hidden 只有在batch learn的时候会被更新  
                在智能体选择动作的时候  其并没有更新
                所以在智能体选择动作这里 
                仍然要使用网络的动作argmax
                而不能直接利用输出层
                dammmmn!
                '''
                q_values = qmix.get_q_values_single(
                    state, self.index)  # 使用 eval_rnn 网络计算 Q 值  谁TM写的
                # hidden_state = h.view(self.args.rnn_hidden_dim)
                action = torch.argmax(q_values).item()  # 选择最大Q值对应的动作
                self.last_action = action
            return action


class UAVControlSAC(UAVControlBase):

    def __init__(
        self,
        method,
        epsilon,  # 添加 epsilon
        index,
        position,
        velocity,
        target_pos,
        obs_coor,  # 添加 obs_coor
        device=torch.device("cpu")):
        self.policy_network = None
        super().__init__(method, epsilon, index, position, velocity,
                         target_pos, obs_coor, device)

    def select_action(self,
                      state,
                      model=None,
                      n_agents=None,
                      n_actions=None,
                      step=None,
                      all_agents=None):
        print('Oops something went wrong!')


class UAVControlIL(UAVControlBase):

    def __init__(
        self,
        method,
        epsilon,  # 添加 epsilon
        index,
        position,
        velocity,
        target_pos,
        obs_coor,  # 添加 obs_coor
        device=torch.device("cpu")):
        self.policy_network = None
        super().__init__(method, epsilon, index, position, velocity,
                         target_pos, obs_coor, device)

    def select_action(self,
                      state,
                      model=None,
                      n_agents=None,
                      n_actions=None,
                      step=None,
                      all_agents=None):
        return self.calculate_action(all_agents)


class UAVControl:

    @staticmethod
    def create_uav(method, **kwargs):
        if method == "qmix":
            return UAVControlQMIX(method, **kwargs)
        elif method == "sac":
            return UAVControlSAC(method, **kwargs)
        elif method == 'None':
            return UAVControlIL(method, **kwargs)
