import numpy as np
class GroundRobot:
    def __init__(self, obstacle_coor, robot_id, position, velocity=np.zeros(2)):
        self.id = robot_id  # 机器人唯一标识符
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.assigned_uav = set()  # 当前分配的无人机
        self.previous_assigned_uav = set()
        self.epsilon = 0.1  # 归一化参数
        self.r = 8  # 期望距离
        self.r_cg = 12  # 通信半径
        self.perception_r = 10  # 感知半径
        self.a = 30.0  # 椭圆长轴
        self.b = 30.0  # 椭圆短轴
        self.r_min = 1  # 最小距离
        self.alpha = 10000  # 形状控制权重
        self.beta = 1.0  # 避碰控制权重
        self.gamma = 1.0  # 编队控制权重
        self.mu = 1.0  # 导航控制权重
        self.tau = 20000000000000.0  # 避障控制权重
        self.max_acc = 5.0  # 最大加速度
        self.max_speed = 5.5  # 最大速度
        self.obstacle = obstacle_coor

    def update_control_input(self, group_robots, x_c, v_c, x_axis, y_axis, theta, dt):
        """更新机器人的速度，根据各控制力的综合作用"""
        # 判断是否有邻居
        has_neighbor = len(group_robots) > 1  # 自己也在 group_robots 中
        # 判断是否超出边界
        x_diff = self.position - x_c
        is_outside = (x_diff[0] / self.a) ** 2 + ( x_diff[1] / self.b) ** 2 > 1  # 椭圆边界判断

        u_p = (
            self.shape_control(x_c, x_axis, y_axis, theta)
            if is_outside
            else np.zeros(2))
        u_Q = self.collision_avoidance(group_robots)
        u_n = self.formation_control(group_robots) if not is_outside else np.zeros(2)
        u_c = self.navigation_control(x_c, v_c)
        u_obs = self.obstacle_avoidance()
        if np.linalg.norm(u_obs) != 0:
            print(f"形状{u_p}，邻居避障{u_Q},编队{u_n},导航{u_c},避障{u_obs}")
        if not has_neighbor and is_outside:
            # 无邻居且超出边界
            u_total = self.alpha * u_p + self.mu * u_c + self.tau * u_obs
        elif has_neighbor and is_outside:
            # 有邻居且超出边界
            u_total = (self.alpha * u_p + self.mu * u_c + self.beta * u_Q + self.tau * u_obs )
        else:
            # 在边界内
            u_total = (self.gamma * u_n + self.mu * u_c + self.beta * u_Q + self.tau * u_obs )

        # 确保 u_total 的范数不超过 self.max_acc
        if np.linalg.norm(u_total) > self.max_acc:
            u_total = u_total * (self.max_acc / np.linalg.norm(u_total))

        # 更新速度并确保速度的范数不超过 self.max_speed
        new_velocity = self.velocity + u_total * dt
        if np.linalg.norm(new_velocity) > self.max_speed:
            new_velocity = new_velocity * (
                self.max_speed / np.linalg.norm(new_velocity)
            )
        self.velocity = new_velocity

    def sigma_norm(self, z):
        """计算 sigma 范数"""
        return (np.sqrt(1 + self.epsilon * np.linalg.norm(z) ** 2) - 1) / self.epsilon

    def rho_h(self, z):
        """冲击函数"""
        h = 0.2
        if z >= 0 and z < h:
            return 1
        elif z >= h and z < 1:
            return 0.5 * (1 + np.cos(3.1415926 * (z - h) / (1 - h)))
        else:
            return 0

    def phi(self, z, a=5, b=5):
        """动作函数"""
        c = abs(a - b) / np.sqrt(4 * a * b)
        return 0.5 * ((a + b) * (z + c) / np.sqrt(1 + (z + c) ** 2) + (a - b))

    def shape_control(self, x_c, x_axis, y_axis, theta):
        """计算形状控制力，保持机器人在椭圆形状的轨迹上"""
        x_diff = self.position - x_c
        # 椭圆坐标变换
        h = (x_diff[0] * np.cos(theta) + x_diff[1] * np.sin(theta)) / x_axis
        g = (-x_diff[0] * np.sin(theta) + x_diff[1] * np.cos(theta)) / y_axis
        f_Ge = h**2 + g**2 - 1
        if f_Ge > 0:
            dh = 2 * h / self.a**2
            dg = 2 * g / self.b**2
            grad_P = np.array(
                [
                    dh * np.cos(theta) - dg * np.sin(theta),
                    dh * np.sin(theta) + dg * np.cos(theta),
                ]
            )
            return -np.maximum(0, f_Ge) * grad_P
        return np.zeros(2)

    def collision_avoidance(self, group_robots):
        """计算避碰控制力"""
        u_Q = np.zeros(2)
        for other_robot in group_robots:
            if other_robot.id != self.id:
                x_diff = other_robot.position - self.position
                dist = np.linalg.norm(x_diff)
                if dist < self.r:
                    u_Q += -np.maximum(0, -np.log(dist**2) + np.log(self.r**2)) * (
                        2 * x_diff / dist**2
                    )
        return u_Q

    def formation_control(self, group_robots):
        """计算编队控制力"""
        c1, c2 = 20.0, 9.0
        force_position = np.zeros(2)
        force_velocity = np.zeros(2)
        for other_robot in group_robots:
            if other_robot.id != self.id:
                x_diff = other_robot.position - self.position
                v_diff = other_robot.velocity - self.velocity
                temp_z = self.sigma_norm(x_diff)
                temp_para = self.rho_h(temp_z / self.sigma_norm(self.r_cg)) * self.phi(
                    temp_z - self.sigma_norm(self.r)
                )
                force_position += (
                    temp_para
                    * x_diff
                    / np.sqrt(1 + self.epsilon * np.linalg.norm(x_diff) ** 2)
                )
                sigma_dist = self.sigma_norm(x_diff)
                a_ij = self.rho_h(sigma_dist / self.sigma_norm(self.r_cg))
                force_velocity += a_ij * v_diff
        return c1 * force_position + c2 * force_velocity

    def navigation_control(self, x_c, v_c):
        """计算导航控制力"""
        c1, c2 = 30.0, 11.0
        return -c1 * (self.position - x_c) / np.sqrt(
            1 + self.epsilon * np.linalg.norm(self.position - x_c) ** 2
        ) - c2 * (self.velocity - v_c)

    def obstacle_avoidance(self):
        """计算避障控制力"""
        # 避障斥力项
        u_obs = np.zeros(2)

        # 感知范围，用于检测附近的障碍物（可以根据需要调整）
        sensing_radius = self.perception_r
        obstacle_avoidance_gain = 1
        # 获取附近的障碍物
        nearby_obstacles = []
        for obstacle_pos in self.obstacle:
            # 计算无人机与障碍物的距离
            distance = np.linalg.norm(self.position - obstacle_pos)
            if distance <= sensing_radius:
                nearby_obstacles.append((obstacle_pos, distance))
        # 计算避障斥力
        for obstacle_pos, distance in nearby_obstacles:
            # 计算斥力方向（从障碍物指向地面机器人）
            direction = self.position - obstacle_pos
            if distance == 0:
                continue  # 避免除零错误
            direction_normalized = direction / distance
            # 计算斥力大小（可以根据需要调整公式）
            force_magnitude = obstacle_avoidance_gain / (distance**2)
            # 限制斥力的最大值
            max_force = self.max_acc  # 或者设置为其他值
            force_magnitude = min(force_magnitude, max_force)
            # 累加斥力
            u_obs += force_magnitude * direction_normalized
        return u_obs

    def update_state(self, dt):
        """
        更新机器人的位置
        :param dt: 时间步长
        """
        self.position += self.velocity * dt
