from datetime import datetime
import json
import math
import random
import os
import time
from typing import Dict
from shapely.geometry import LineString, Polygon, Point, MultiPolygon
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Ellipse
from matplotlib.animation import FFMpegWriter  # 导入 FFMpegWriter
from shapely.ops import substring
import casadi as ca
import numpy as np


def line_substring_by_points(line: LineString, p_start: np.ndarray,
                             p_end: np.ndarray) -> LineString:
    """从线路 line 中提取 p_start 和 p_end 间的子线段。
       p_start, p_end为np数组[x,y]，不要求是节点坐标，但必须在line上。"""
    p_start_pt = Point(p_start)
    p_end_pt = Point(p_end)

    # 获取参数距离
    start_dist = line.project(p_start_pt)
    end_dist = line.project(p_end_pt)

    # 生成子线段
    return substring(line, start_dist, end_dist)


def interpolate_line(line, num_interpolated_points=5):
    # 获取LineString的坐标列表
    coords = list(line.coords)

    # 新的坐标列表，包括原始的坐标和插值后的点
    new_coords = [coords[0]]  # 先将第一个点加入新的坐标列表

    # 遍历所有相邻的点对，进行插值
    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])  # 当前点
        p2 = np.array(coords[i + 1])  # 下一个点

        # 生成5个插值点
        x_vals = np.linspace(p1[0], p2[0],
                             num_interpolated_points + 2)[1:-1]  # 不包括第一个和最后一个点
        y_vals = np.linspace(p1[1], p2[1], num_interpolated_points + 2)[1:-1]

        # 将插值点加入新的坐标列表
        for x, y in zip(x_vals, y_vals):
            new_coords.append((x, y))

        # 最后将原始的下一个点加入
        if i == len(coords) - 2:  # 最后一个点
            new_coords.append(coords[i + 1])

    return new_coords


def lines_to_polygon(lineA, lineB):
    # 提取两条折线的坐标序列
    coordsA = list(lineA.coords)
    coordsB = list(lineB.coords)

    A_start = coordsA[0]
    A_end = coordsA[-1]

    B_start = coordsB[0]
    B_end = coordsB[-1]

    # 构造多边形顶点列表
    # 顺序：A_start -> B_start -> (沿着lineB) -> B_end -> A_end -> (沿着lineA逆序) -> 回到A_start
    polygon_coords = []
    polygon_coords.append(A_start)
    polygon_coords.append(B_start)
    polygon_coords.extend(coordsB)  # 沿着 lineB 的顶点序列
    polygon_coords.append(A_end)
    polygon_coords.extend(
        coordsA[::-1])  # 沿着 lineA 的反向顶点序列（不包括最后一个点，因为 A_start 已有）
    # 最后一个点是 coordsA[0], 即 A_start，会与第一个点重合形成闭合

    # 创建多边形对象
    polygon = Polygon(polygon_coords)

    return polygon


def numpy_to_list(data):
    """将所有 NumPy 数组递归转换为 Python 列表，以便保存为 JSON 格式。"""
    if isinstance(data, dict):
        return {key: numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [numpy_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def get_episode_length(episode_data):
    terminated_list = episode_data['terminated']
    for idx, terminated in enumerate(terminated_list):
        if terminated[0]:  # 这里假设所有智能体的终止状态是同步的，所以检查第一个智能体的状态
            return idx + 1  # 返回实际长度（加1是因为索引从0开始）
    return len(terminated_list)  # 如果没有找到True，返回列表的完整长度


def process_draw_state(draw_state, target_point, communication_radius,
                       max_ellipse_length):
    processed_draw_state = []
    for time_step in draw_state:
        processed_time_step = []
        for agent_id, agent_state in enumerate(time_step):
            # 处理位置数据
            position_x = -((agent_state[0] * communication_radius) -
                           target_point[agent_id][0])
            position_y = -((agent_state[1] * communication_radius) -
                           target_point[agent_id][1])

            # 处理椭圆长轴数据
            ellipse_length = agent_state[8] * max_ellipse_length

            # 只保留位置和椭圆长轴长度
            processed_time_step.append(
                [position_x, position_y, ellipse_length])

        processed_draw_state.append(processed_time_step)
    return processed_draw_state


def plot_environment(processed_draw_state,
                     target_point,
                     obstacle_coor,
                     ellipse_area,
                     WannaRecord,
                     episode_length,
                     video_path=None):
    fig, ax = plt.subplots()
    images = []

    def update_plot(frame):
        ax.clear()

        # 绘制目标点
        ax.scatter(target_point[0][0],
                   target_point[0][1],
                   color='green',
                   s=100,
                   label='Target 1')
        ax.scatter(target_point[1][0],
                   target_point[1][1],
                   color='blue',
                   s=100,
                   label='Target 2')

        # 绘制障碍物
        ax.scatter(obstacle_coor[:, 0],
                   obstacle_coor[:, 1],
                   color='red',
                   s=10,
                   label='Obstacle')

        # 绘制无人机位置及椭圆
        time_step = processed_draw_state[frame]
        for agent in time_step:
            position_x, position_y, ellipse_length = agent
            ellipse = Ellipse(xy=(position_x, position_y),
                              width=2 * ellipse_length,
                              height=2 *
                              (ellipse_area / ellipse_length / np.pi),
                              angle=0,
                              edgecolor='purple',
                              facecolor='none')
            ax.add_patch(ellipse)
            ax.scatter(position_x,
                       position_y,
                       color='purple',
                       s=10,
                       label='UAV Position')

        # 设置图像范围
        ax.set_xlim(0, 1000)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')  # 设置坐标轴比例相等

    # 如果 WannaRecord 为 True，保存为 MP4
    if WannaRecord and video_path is not None:
        # 创建时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"environment_video_{timestamp}.mp4"
        full_video_path = os.path.join(video_path, filename)

        # 逐帧更新图像，并保存为图像数组
        for frame in range(episode_length):
            update_plot(frame)
            fig.canvas.draw()

            # Convert the current plot to an image array and append to images list
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
            images.append(image)

        # 保存为 MP4
        imageio.mimsave(full_video_path, images, fps=10, codec='libx264')
        print(f"Animation saved to {full_video_path}")

    # 如果 WannaRecord 为 False，仅显示图像
    else:
        for index in range(len(processed_draw_state)):
            update_plot(index)
            print(index)
            plt.pause(0.01)  # 控制动画帧显示的时间
        plt.show()


# 示例调用：
# plot_environment(processed_draw_state, target_point, obstacle_coor, ellipse_area, WannaRecord=True, video_path='C:/path/to/your/folder')

    if WannaRecord and video_path is not None:
        # 创建时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"environment_video_{timestamp}.mp4"
        full_video_path = os.path.join(video_path, filename)

        # 创建动画并保存为视频
        ani = animation.FuncAnimation(fig,
                                      update_plot,
                                      frames=len(processed_draw_state),
                                      repeat=False)

        # 使用 FFMpegWriter 保存视频
        writer = FFMpegWriter(fps=10)  # 创建 FFMpegWriter 对象
        ani.save(full_video_path, writer=writer)
        print(f"Video saved to {full_video_path}")
    else:
        # 仅显示图像
        for index in range(len(processed_draw_state)):
            update_plot(index)
            plt.pause(0.1)  # 控制动画帧显示的时间
            print(index)
        plt.show()


def discretize_line(line_points, distance_per_point=1.0):
    """
    将一条折线以固定间隔进行离散化。
    例如 distance_per_point=1，则表示每1米产生一个点。

    参数：
    line_points：numpy array或list of tuples，折线的顶点坐标，如 [(x0, y0), (x1, y1), ...]。
    distance_per_point：float，每两个离散点之间的距离间隔（米或单位长度）。

    返回：
    discretized_points：numpy array，形状为(N, 2)，离散化后的点的坐标。
    """

    # 确保 line_points 为 numpy array
    line_points = np.array(line_points)

    # 计算各段线的长度
    segment_lengths = np.sqrt(np.sum(np.diff(line_points, axis=0)**2, axis=1))
    total_length = np.sum(segment_lengths)

    # 根据 total_length 和 distance_per_point 计算总点数
    # 确保包括终点（加1）
    num_points = int(np.floor(total_length / distance_per_point)) + 1

    # 目标距离数组，从0到total_length，间隔为distance_per_point
    # 注意：这样最后可能会不到终点，如差一点距离到总长度，则最后一个点设为终点
    target_distances = np.linspace(0, total_length, num_points)

    # 对于每个目标距离进行插值求坐标
    discretized_points = []

    # segment_start_length 用来记录每条段的起点在折线中的累计长度
    cumulative_lengths = np.cumsum(segment_lengths)
    cumulative_lengths = np.insert(cumulative_lengths, 0, 0.0)  # 在开头插入0，表示起点

    # cumulative_lengths[i] 表示从第一点到第 i 个点的总长度

    # 函数：给定一个距离d，返回该距离对应的折线上的坐标
    def interpolate_point(d):
        # 找到 d 所处的线段
        # cumulative_lengths: [0, L1, L1+L2, ... , total_length]
        # 使用搜索确定 d 落在哪一段
        seg_index = np.searchsorted(cumulative_lengths, d) - 1
        seg_index = max(0, min(seg_index, len(segment_lengths) - 1))

        # 当前段起点、终点坐标
        p1 = line_points[seg_index]
        p2 = line_points[seg_index + 1]

        # 当前段的起始累计长度
        start_length = cumulative_lengths[seg_index]
        # 当前段的长度
        seg_len = segment_lengths[seg_index]

        # 计算 d 在当前段中的比例
        ratio = (d - start_length) / seg_len

        # 按比例插值
        x = p1[0] + ratio * (p2[0] - p1[0])
        y = p1[1] + ratio * (p2[1] - p1[1])
        return (x, y)

    for d in target_distances:
        discretized_points.append(interpolate_point(d))

    discretized_points = np.array(discretized_points)
    return discretized_points

def discretize_circle(center, radius, distance_per_point=1.0, num_points=None):

    """
    将圆周进行离散化，输出均匀分布在圆上的点。
    
    参数：
    center：tuple，圆心坐标 (cx, cy)
    radius：float，圆的半径
    distance_per_point：float，两个点之间的弧长（可选，默认为1.0）
    num_points：int，圆上总共生成的点数（可选，优先使用该参数）

    返回：
    points：numpy array，形状为(N, 2)，圆周上的离散点坐标
    """
    cx, cy = center

    if num_points is None:
        circumference = 2 * np.pi * radius
        num_points = max(4, int(np.ceil(circumference / distance_per_point)))

    # 角度均匀分布
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # 圆上的点坐标
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)

    points = np.stack((x, y), axis=1)
    return points
def discretize_square(center, side_length, distance_per_point=1.0):
    """
    将正方形边界以固定弧长（线段距离）间隔进行离散化。
    
    参数：
        center: tuple，表示正方形中心坐标 (x, y)
        side_length: float，正方形的边长
        distance_per_point: float，每两个离散点之间的间隔距离（单位长度）
        
    返回：
        discretized_points: numpy array，形状为 (N, 2)，离散化后的正方形边界上的点的坐标
    """
    # 计算正方形的半边长和周长
    half = side_length / 2.0
    perimeter = 4 * side_length
    
    # 根据周长和间隔距离计算需要采样的点数
    num_points = int(np.floor(perimeter / distance_per_point))
    if num_points < 1:
        num_points = 1

    # 生成沿正方形边界的距离参数，均匀分布在 [0, 周长)
    distances = np.linspace(0, perimeter, num_points, endpoint=False)
    points = []

    for d in distances:
        if d < side_length:
            # 下边，从左下角到右下角
            t = d / side_length
            x = center[0] - half + t * side_length
            y = center[1] - half
        elif d < 2 * side_length:
            # 右边，从右下角到右上角
            t = (d - side_length) / side_length
            x = center[0] + half
            y = center[1] - half + t * side_length
        elif d < 3 * side_length:
            # 上边，从右上角到左上角
            t = (d - 2 * side_length) / side_length
            x = center[0] + half - t * side_length
            y = center[1] + half
        else:
            # 左边，从左上角到左下角
            t = (d - 3 * side_length) / side_length
            x = center[0] - half
            y = center[1] + half - t * side_length
        points.append((x, y))

    discretized_points = np.array(points)
    return discretized_points
def linear_decay_epsilon(initial_epsilon: float, min_epsilon: float,
                         decay_steps: int, current_step: int) -> float:
    """
    计算epsilon的线性衰减值。

    参数:
    initial_epsilon (float): epsilon的初始值。
    min_epsilon (float): epsilon的最小值。
    decay_steps (int): epsilon衰减到最小值的步数。
    current_step (int): 当前的训练步数。

    返回:
    float: 当前步的epsilon值。
    """
    epsilon = initial_epsilon - (initial_epsilon -
                                 min_epsilon) * (current_step / decay_steps)
    return max(min_epsilon, epsilon)


def process_batch_to_tensor(batch, device='cpu'):
    """
    将批量数据中的每个元素转换为张量，并放置在指定设备上。

    参数:
    - batch (dict): 包含批量数据的字典。
    - device (str): 目标设备 ('cpu' 或 'cuda')。

    返回:
    - dict: 转换后的批量数据字典，其中所有元素均为张量。
    """
    if 'data' in batch:
        converted_data = {key: [] for key in batch['data'][0].keys()}
        # 遍历每个episode，并将数据转换为新的结构
        for episode in batch['data']:
            for key, value in episode.items():
                converted_data[key].append(value)
        data_dict = converted_data  # 提取 'data' 部分
        for key in data_dict:
            try:
                # 创建一个用于存储处理后数据的临时列表
                processed_data = []
                # 遍历 batch[key] 的每个元素
                for elem in data_dict[key]:
                    if isinstance(elem, list):
                        # 如果是列表，检查每个子元素是否为 tensor 并将其转换为 numpy 数组
                        sub_data = [
                            sub_elem.cpu().numpy() if isinstance(
                                sub_elem, torch.Tensor) else np.array(sub_elem)
                            for sub_elem in elem
                        ]
                        processed_data.append(np.array(sub_data))
                    elif isinstance(elem, torch.Tensor):
                        # 如果元素是 tensor，转换为 numpy 数组
                        processed_data.append(elem.cpu().numpy())
                    else:
                        # 其他类型，尝试直接转换为 numpy 数组
                        processed_data.append(np.array(elem))

                # 将处理后的数据列表转换为 numpy 数组
                batch_array = np.array(processed_data)
                # 将 numpy 数组转换为 tensor 并放置在目标设备上
                data_dict[key] = torch.tensor(batch_array, device=device)

            except ValueError as e:
                print(f"Error converting key: {key}")
                print(f"Expected shape: {np.shape(batch_array)}")
                print(f"Error: {str(e)}")
                raise e

        return data_dict
    else:
        for key in batch:
            try:
                # 创建一个用于存储处理后数据的临时列表
                processed_data = []
                # 遍历 batch[key] 的每个元素
                for elem in batch[key]:
                    if isinstance(elem, list):
                        # 如果是列表，检查每个子元素是否为 tensor 并将其转换为 numpy 数组
                        sub_data = [
                            sub_elem.cpu().numpy() if isinstance(
                                sub_elem, torch.Tensor) else np.array(sub_elem)
                            for sub_elem in elem
                        ]
                        processed_data.append(np.array(sub_data))
                    elif isinstance(elem, torch.Tensor):
                        # 如果元素是 tensor，转换为 numpy 数组
                        processed_data.append(elem.cpu().numpy())
                    else:
                        # 其他类型，尝试直接转换为 numpy 数组
                        processed_data.append(np.array(elem))

            # 将处理后的数据列表转换为 numpy 数组
                batch_array = np.array(processed_data)
                # 将 numpy 数组转换为 tensor 并放置在目标设备上
                batch[key] = torch.tensor(batch_array, device=device)

            except ValueError as e:
                print(f"Error converting key: {key}")
                print(f"Expected shape: {np.shape(batch_array)}")
                print(f"Error: {str(e)}")
                raise e

    return batch


def generate_ellipse_points(center_x, center_y, a, b, num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse_x = center_x + a * np.cos(t)
    ellipse_y = center_y + b * np.sin(t)
    return np.column_stack((ellipse_x, ellipse_y))


def calculate_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def filter_ellipse_points_by_angle(ellipse_points, agent_position,
                                   velocity_direction):
    filtered_points = []
    for point in ellipse_points:
        v = point - agent_position
        angle = calculate_angle(velocity_direction, v)
        if angle <= np.pi / 2:
            filtered_points.append(point)
    return filtered_points


def calculate_obstacle_distances(processed_draw_state, obstacle_coor,
                                 ellipse_area, perception_radius):
    for time_step_index, time_step_data in enumerate(processed_draw_state):
        for agent_index, agent_data in enumerate(time_step_data):
            center_x, center_y, a = agent_data[:3]

            # 如果不是第一个时间步，则计算速度方向
            if time_step_index == 0:
                velocity_direction = np.array([1.0, 0.0])  # 第一个时间步的默认速度方向
            else:
                # 获取上一时间步相同 agent 的位置
                prev_agent_data = processed_draw_state[time_step_index -
                                                       1][agent_index]
                prev_x, prev_y = prev_agent_data[:2]

                # 计算速度方向
                velocity_direction = np.array(
                    [center_x - prev_x, center_y - prev_y])
                if np.linalg.norm(velocity_direction) == 0:
                    velocity_direction = np.array([1.0, 0.0])  # 如果速度为零，使用默认方向

            # 计算短半轴 b
            area = ellipse_area  # 椭圆面积（假设面积已知）
            b = area / (np.pi * a)

            # 生成椭圆上的点
            ellipse_points = generate_ellipse_points(center_x, center_y, a, b)

            # 筛选与速度方向夹角小于等于90度的点
            filtered_points = filter_ellipse_points_by_angle(
                ellipse_points, np.array([center_x, center_y]),
                velocity_direction)
            filtered_obstacles = []
            for obs in obstacle_coor:
                distance = np.linalg.norm(agent_data[:2] - np.array(obs))
                if distance <= perception_radius:
                    filtered_obstacles.append(obs)
            min_distance = float('inf')
            best_angle = None

            for obs in filtered_obstacles:
                for point in filtered_points:
                    v2 = obs - point
                    distance = np.linalg.norm(v2)
                    if distance < min_distance:
                        angle = calculate_angle(velocity_direction, v2)
                        min_distance = distance
                        best_angle = angle

            if best_angle == None:
                best_angle = 0
                min_distance = perception_radius

            # 更新 agent 数据，添加 dis2obstacle 和 angle2obstacle
            agent_data.extend([min_distance, best_angle])

    return processed_draw_state


class ReplayBuffer:

    def __init__(self, buffer_size, manager):
        self.buffer_size = buffer_size
        self.manager = manager
        self.buffer = manager.list()  # 使用 Manager 创建共享列表
        self.lock = manager.Lock()  # 使用 Manager 创建共享锁

    def store(self, experience):
        with self.lock:
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)  # 移除最旧的经验
            self.buffer.append(experience)

    def sample(self, batch_size):
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def get_all_data(self):
        with self.lock:
            return list(self.buffer)

    def size(self):
        with self.lock:
            return len(self.buffer)


def save_replay_buffer(replay_buffer, args):
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # 生成保存文件的名称，包括时间戳和 map 文件名
    filename = f"all_episodes_data_{args.map_filename}_{timestamp}.json"
    # 定义保存路径
    path_on_f_drive = r'F:\RL4ConflictResolution\all_train_data'
    default_save_dir = 'all_train_data'
    if os.path.exists(path_on_f_drive):
        # 如果存在，则保存到 F 盘
        save_path = os.path.join(path_on_f_drive, filename)
        print(f"正在将数据保存到 {save_path}")
    else:
        # 否则，保存到当前目录下的默认文件夹
        save_path = os.path.join(default_save_dir, filename)
        print(f"F 盘路径未找到。正在将数据保存到 {save_path}")
    # 确保路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 从 replay_buffer 获取所有数据
    data = list(replay_buffer)
    data = numpy_to_list(data)  # 确保数据可序列化
    # 将数据保存到指定路径
    with open(save_path, 'w') as f:
        json.dump(data, f)
    print(f"数据已保存到 {save_path}")


def compute_shortest_distance(agent1, agent2):
    if isinstance(agent1, np.ndarray):
        # 提取位置和椭圆参数
        pos1 = agent1[2:4]  # [x_pos, y_pos]
        pos2 = agent2[2:4]
        a1 = agent1[4]      # 长轴
        a2 = agent2[4]
        rot1 = agent1[5]    # 旋转角度（弧度）
        rot2 = agent2[5]
        
        # 计算半短轴
        b1 = 900 / a1
        b2 = 900 / a2
        
        # 计算直线总距离
        total_length = np.linalg.norm(pos2 - pos1)
        if total_length == 0:
            return 0.0
        
        # 计算直线参数：P(t) = pos1 + t*(pos2-pos1), t in [0,1]
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # 计算交点参数 t 值对第一个椭圆（agent1）：
        # 将直线上任一点 P(t) 相对于椭圆中心 pos1: (X, Y) = (x - pos1[0], y - pos1[1])
        # 对点进行逆旋转，旋转角度 = -rot1
        # 得到局部坐标: X' = (X*cos(rot1) + Y*sin(rot1))
        #                   Y' = (-X*sin(rot1) + Y*cos(rot1))
        # 带入椭圆方程 (X'/a1)^2 + (Y'/b1)^2 = 1 得到关于 t 的二次方程
        X1 = pos1[0] - pos1[0]  # =0
        Y1 = pos1[1] - pos1[1]  # =0
        # 实际上，考虑 P(t)= pos1 + t*(dx,dy)，点相对于 pos1 为 (t*dx, t*dy)
        # 逆旋转后的坐标为:
        # X1' = t*dx*cos(rot1) + t*dy*sin(rot1) = t*(dx*cos(rot1) + dy*sin(rot1))
        # Y1' = -t*dx*sin(rot1) + t*dy*cos(rot1) = t*(-dx*sin(rot1) + dy*cos(rot1))
        # 椭圆方程:
        # (t*(dx*cos(rot1)+dy*sin(rot1))/a1)^2 + (t*(-dx*sin(rot1)+dy*cos(rot1))/b1)^2 = 1
        # 整理后得到:
        A1 = ((dx * np.cos(rot1) + dy * np.sin(rot1)) / a1)**2 + ((-dx * np.sin(rot1) + dy * np.cos(rot1)) / b1)**2
        B1 = 0  # 此处 B1=0，因为常数项为0（P(t)相对于中心为0 when t=0）
        C1 = 0 - 1  # -1
        # 因为直线从椭圆中心出发，所以 t=0 点在椭圆内部，
        # 求解 t 满足 A1*t^2 - 1 = 0  => t^2 = 1/A1, t = ±1/sqrt(A1)
        t_values1 = []
        if A1 > 0:
            t_candidate = 1/np.sqrt(A1)
            # t 正负均为交点，取位于 [0,1] 范围内的
            if 0 <= t_candidate <= 1:
                t_values1.append(t_candidate)
            if 0 <= -t_candidate <= 1:
                t_values1.append(-t_candidate)
            t_values1.sort()
        
        # 计算交点参数 t 值对第二个椭圆（agent2）：
        # 直线参数不变，但将坐标转换到以 pos2 为中心、逆旋转 rot2 的局部坐标系中
        # 对于 P(t)= pos1 + t*(dx,dy)，其相对于 pos2 为 (pos1 - pos2 + t*(dx,dy))
        # 令 delta = pos1 - pos2, 则相对于 pos2 的坐标为 (delta_x + t*dx, delta_y + t*dy)
        delta = pos1 - pos2
        # 经过逆旋转 rot2:
        # X2' = (delta_x + t*dx)*cos(rot2) + (delta_y + t*dy)*sin(rot2)
        # Y2' = -(delta_x + t*dx)*sin(rot2) + (delta_y + t*dy)*cos(rot2)
        # 带入椭圆方程: (X2'/a2)^2 + (Y2'/b2)^2 = 1，整理后得到关于 t 的二次方程：
        A2 = ((dx * np.cos(rot2) + dy * np.sin(rot2)) / a2)**2 + ((-dx * np.sin(rot2) + dy * np.cos(rot2)) / b2)**2
        B2 = 2 * (((delta[0]*np.cos(rot2) + delta[1]*np.sin(rot2))*(dx*np.cos(rot2)+dy*np.sin(rot2)))/a2**2 +
                   ((-delta[0]*np.sin(rot2) + delta[1]*np.cos(rot2))*(-dx*np.sin(rot2)+dy*np.cos(rot2)))/b2**2)
        C2 = ((delta[0]*np.cos(rot2) + delta[1]*np.sin(rot2))/a2)**2 + ((-delta[0]*np.sin(rot2) + delta[1]*np.cos(rot2))/b2)**2 - 1
        t_values2 = []
        discriminant2 = B2**2 - 4*A2*C2
        if discriminant2 >= 0 and A2 != 0:
            sqrt_disc2 = np.sqrt(discriminant2)
            t1 = (-B2 + sqrt_disc2) / (2*A2)
            t2 = (-B2 - sqrt_disc2) / (2*A2)
            for t in [t1, t2]:
                if 0 <= t <= 1:
                    t_values2.append(t)
            t_values2.sort()
        
        # 收集线段在每个椭圆内部的参数区间
        intervals = []
        if len(t_values1) == 2:
            intervals.append((t_values1[0], t_values1[1]))
        elif len(t_values1) == 1:
            intervals.append((t_values1[0], t_values1[0]))
        if len(t_values2) == 2:
            intervals.append((t_values2[0], t_values2[1]))
        elif len(t_values2) == 1:
            intervals.append((t_values2[0], t_values2[0]))
        
        # 合并重叠的区间
        intervals.sort(key=lambda x: x[0])
        merged_intervals = []
        for interval in intervals:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                prev = merged_intervals[-1]
                if interval[0] <= prev[1]:
                    merged_intervals[-1] = (prev[0], max(prev[1], interval[1]))
                else:
                    merged_intervals.append(interval)
        
        # 计算直线上在椭圆内部的总长度
        length_inside = sum((interval[1] - interval[0]) * total_length for interval in merged_intervals)
        # 直线上在椭圆外部的长度
        length_outside = total_length - length_inside
        return length_outside
    else:
        # 获取当前时刻的位置和长轴长度
        pos1 = agent1.position
        pos2 = agent2.position
        ellipse_length1 = agent1.ellipse_length
        ellipse_length2 = agent2.ellipse_length

        # 计算当前时刻的半短轴长度
        ellipse_area1 = agent1.ellipse_area
        ellipse_area2 = agent2.ellipse_area
        semi_minor_axis1 = ellipse_area1 / (np.pi * ellipse_length1)
        semi_minor_axis2 = ellipse_area2 / (np.pi * ellipse_length2)

        # 计算两个中心点之间的总距离
        total_length = np.linalg.norm(pos2 - pos1)

        if total_length == 0:
            # 如果两个中心重合，返回长度为0
            return 0.0, 0.0

        # 计算直线与第一个椭圆的交点参数 t_values1
        x0, y0 = pos1
        x1, y1 = pos2
        dx = x1 - x0
        dy = y1 - y0
        h1, k1 = pos1
        a1, b1 = ellipse_length1, semi_minor_axis1
        X1 = x0 - h1
        Y1 = y0 - k1

        A1 = (dx**2) / a1**2 + (dy**2) / b1**2
        B1 = 2 * ((dx * X1) / a1**2 + (dy * Y1) / b1**2)
        C1 = (X1**2) / a1**2 + (Y1**2) / b1**2 - 1

        discriminant1 = B1**2 - 4 * A1 * C1

        t_values1 = []
        if discriminant1 >= 0:
            sqrt_discriminant1 = np.sqrt(discriminant1)
            t_candidates = [(-B1 + sqrt_discriminant1) / (2 * A1),
                            (-B1 - sqrt_discriminant1) / (2 * A1)]
            t_values1 = [t for t in t_candidates if 0 <= t <= 1]
            t_values1.sort()

        # 计算直线与第二个椭圆的交点参数 t_values2
        h2, k2 = pos2
        a2, b2 = ellipse_length2, semi_minor_axis2
        X2 = x0 - h2
        Y2 = y0 - k2

        A2 = (dx**2) / a2**2 + (dy**2) / b2**2
        B2 = 2 * ((dx * X2) / a2**2 + (dy * Y2) / b2**2)
        C2 = (X2**2) / a2**2 + (Y2**2) / b2**2 - 1

        discriminant2 = B2**2 - 4 * A2 * C2

        t_values2 = []
        if discriminant2 >= 0:
            sqrt_discriminant2 = np.sqrt(discriminant2)
            t_candidates = [(-B2 + sqrt_discriminant2) / (2 * A2),
                            (-B2 - sqrt_discriminant2) / (2 * A2)]
            t_values2 = [t for t in t_candidates if 0 <= t <= 1]
            t_values2.sort()

        # 收集线段在每个椭圆内部的参数区间
        intervals = []

        # 处理第一个椭圆的交点
        if len(t_values1) == 2:
            intervals.append((t_values1[0], t_values1[1]))
        elif len(t_values1) == 1:
            intervals.append((t_values1[0], t_values1[0]))

        # 处理第二个椭圆的交点
        if len(t_values2) == 2:
            intervals.append((t_values2[0], t_values2[1]))
        elif len(t_values2) == 1:
            intervals.append((t_values2[0], t_values2[0]))

        # 合并重叠的区间
        intervals.sort(key=lambda x: x[0])
        merged_intervals = []
        for interval in intervals:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                prev = merged_intervals[-1]
                if interval[0] <= prev[1]:
                    merged_intervals[-1] = (prev[0], max(prev[1], interval[1]))
                else:
                    merged_intervals.append(interval)

        # 计算线段在椭圆内部的总长度
        length_inside = sum((interval[1] - interval[0]) * total_length
                            for interval in merged_intervals)

        # 计算线段在椭圆外部的长度
        length_outside = total_length - length_inside

        # 获取上一时刻的位置和长轴长度
        last_pos1 = agent1.last_pos
        last_pos2 = agent2.last_pos
        last_ellipse_length1 = agent1.last_ellipse_length
        last_ellipse_length2 = agent2.last_ellipse_length

        # 计算上一时刻的半短轴长度
        last_semi_minor_axis1 = ellipse_area1 / (np.pi * last_ellipse_length1)
        last_semi_minor_axis2 = ellipse_area2 / (np.pi * last_ellipse_length2)

        # 计算上一时刻两个中心点之间的总距离
        total_length_last = np.linalg.norm(last_pos2 - last_pos1)

        if total_length_last == 0:
            # 如果两个中心重合，返回长度为0
            return length_outside, 0.0

        # 计算上一时刻直线与第一个椭圆的交点参数 t_values1_last
        x0, y0 = last_pos1
        x1, y1 = last_pos2
        dx = x1 - x0
        dy = y1 - y0
        h1, k1 = last_pos1
        a1, b1 = last_ellipse_length1, last_semi_minor_axis1
        X1 = x0 - h1
        Y1 = y0 - k1

        A1 = (dx**2) / a1**2 + (dy**2) / b1**2
        B1 = 2 * ((dx * X1) / a1**2 + (dy * Y1) / b1**2)
        C1 = (X1**2) / a1**2 + (Y1**2) / b1**2 - 1

        discriminant1 = B1**2 - 4 * A1 * C1

        t_values1_last = []
        if discriminant1 >= 0:
            sqrt_discriminant1 = np.sqrt(discriminant1)
            t_candidates = [(-B1 + sqrt_discriminant1) / (2 * A1),
                            (-B1 - sqrt_discriminant1) / (2 * A1)]
            t_values1_last = [t for t in t_candidates if 0 <= t <= 1]
            t_values1_last.sort()

        # 计算上一时刻直线与第二个椭圆的交点参数 t_values2_last
        h2, k2 = last_pos2
        a2, b2 = last_ellipse_length2, last_semi_minor_axis2
        X2 = x0 - h2
        Y2 = y0 - k2

        A2 = (dx**2) / a2**2 + (dy**2) / b2**2
        B2 = 2 * ((dx * X2) / a2**2 + (dy * Y2) / b2**2)
        C2 = (X2**2) / a2**2 + (Y2**2) / b2**2 - 1

        discriminant2 = B2**2 - 4 * A2 * C2

        t_values2_last = []
        if discriminant2 >= 0:
            sqrt_discriminant2 = np.sqrt(discriminant2)
            t_candidates = [(-B2 + sqrt_discriminant2) / (2 * A2),
                            (-B2 - sqrt_discriminant2) / (2 * A2)]
            t_values2_last = [t for t in t_candidates if 0 <= t <= 1]
            t_values2_last.sort()

        # 收集上一时刻线段在每个椭圆内部的参数区间
        intervals_last = []

        # 处理第一个椭圆的交点
        if len(t_values1_last) == 2:
            intervals_last.append((t_values1_last[0], t_values1_last[1]))
        elif len(t_values1_last) == 1:
            intervals_last.append((t_values1_last[0], t_values1_last[0]))

        # 处理第二个椭圆的交点
        if len(t_values2_last) == 2:
            intervals_last.append((t_values2_last[0], t_values2_last[1]))
        elif len(t_values2_last) == 1:
            intervals_last.append((t_values2_last[0], t_values2_last[0]))

        # 合并重叠的区间
        intervals_last.sort(key=lambda x: x[0])
        merged_intervals_last = []
        for interval in intervals_last:
            if not merged_intervals_last:
                merged_intervals_last.append(interval)
            else:
                prev = merged_intervals_last[-1]
                if interval[0] <= prev[1]:
                    merged_intervals_last[-1] = (prev[0],
                                                max(prev[1], interval[1]))
                else:
                    merged_intervals_last.append(interval)

        # 计算上一时刻线段在椭圆内部的总长度
        length_inside_last = sum((interval[1] - interval[0]) * total_length_last
                                for interval in merged_intervals_last)
        # 计算上一时刻线段在椭圆外部的长度
        length_outside_last = total_length_last - length_inside_last
        return length_outside, length_outside_last

def compute_outside_length(agent, point2d):
    # -------------------------
    # 公共函数：判断点是否在椭圆内
    # -------------------------
    def is_point_inside_ellipse(pos, a, b, rot, point):
        # 将点转换到局部坐标系
        dx = point[0] - pos[0]
        dy = point[1] - pos[1]
        x_prime = dx * np.cos(rot) + dy * np.sin(rot)
        y_prime = -dx * np.sin(rot) + dy * np.cos(rot)
        return (x_prime / a)**2 + (y_prime / b)**2 <= 1.0 + 1e-8  # 添加容差

    # -------------------------
    # 处理两种类型的agent
    # -------------------------
    if isinstance(agent, np.ndarray):
        # -------------------------
        # 情况A：agent是NumPy数组（含旋转）
        # -------------------------
        pos = agent[2:4]
        a = agent[4]
        rot = agent[5]
        b = 900.0 / a

        # 计算线段参数
        dx = point2d[0] - pos[0]
        dy = point2d[1] - pos[1]
        total_length = np.hypot(dx, dy)
        
        # 边界情况：线段长度为0
        if total_length < 1e-12:
            return 0.0

        # 计算椭圆方程系数A
        A = ((dx * np.cos(rot) + dy * np.sin(rot)) / a)**2 + \
            ((-dx * np.sin(rot) + dy * np.cos(rot)) / b)**2

        # 处理特殊情况：线段方向完全在椭圆内部
        if A <= 1e-15:
            # 必须验证终点是否在椭圆内
            if is_point_inside_ellipse(pos, a, b, rot, point2d):
                return 0.0
            else:
                return total_length

        # 计算交点参数
        t_pos = 1.0 / np.sqrt(A)

        # 关键逻辑修正：
        if t_pos <= 1.0:
            # 线段部分在外部：从t_pos到1.0的部分
            return (1.0 - t_pos) * total_length
        else:
            # 整个线段在内部
            return 0.0

    else:
        # -------------------------
        # 情况B：自定义对象（无旋转）
        # -------------------------
        pos = agent.position
        a = agent.ellipse_length
        b = agent.ellipse_area / (np.pi * a)

        # 计算线段参数
        dx = point2d[0] - pos[0]
        dy = point2d[1] - pos[1]
        total_length = np.hypot(dx, dy)

        # 边界情况：线段长度为0
        if total_length < 1e-12:
            return 0.0

        # 计算椭圆方程系数A
        A = (dx**2) / a**2 + (dy**2) / b**2

        # 处理特殊情况：线段方向完全在椭圆内部
        if A <= 1.0 + 1e-8:  # 包含容差
            return 0.0
        
        # 计算交点参数
        t_pos = 1.0 / np.sqrt(A)
        
        if t_pos <= 1.0:
            return (1.0 - t_pos) * total_length
        else:
            return 0.0
def find_polygon_containing_or_nearest(multi_polygon: MultiPolygon,
                                       point: Point):
    """
    在 MultiPolygon 中找到包含给定点的 Polygon。
    如果没有 Polygon 包含该点，则返回距离该点最近的 Polygon。

    Parameters:
    - multi_polygon: MultiPolygon 对象
    - point: Point 对象，表示目标点

    Returns:
    - polygon: 包含该点的 Polygon 或距离最近的 Polygon
    """
    # 遍历 MultiPolygon 中的每个 Polygon
    for polygon in multi_polygon.geoms:
        if polygon.contains(point):
            return polygon

    # 如果没有 Polygon 包含该点，则计算距离最近的 Polygon
    nearest_polygon = None
    min_distance = float('inf')
    for polygon in multi_polygon.geoms:
        distance = polygon.distance(point)
        if distance < min_distance:
            min_distance = distance
            nearest_polygon = polygon

    return nearest_polygon


def ellipse_radius_in_direction(a, b, theta, direction):
    """
    计算椭圆在某个方向上的半径（边界点到椭圆中心的距离）。
    Parameters:
        a: 长轴长度
        b: 短轴长度
        theta: 椭圆倾角（弧度）
        direction: 方向向量 (dx, dy)，需要归一化
    Returns:
        椭圆在该方向上的半径长度
    """
    # 将方向向量归一化
    direction = direction / np.linalg.norm(direction)
    dx, dy = direction

    # 椭圆的旋转矩阵
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 椭圆在标准位置的参数化方程
    numerator = (a * cos_t * dx + b * sin_t * dy)**2 + (a * sin_t * dx -
                                                        b * cos_t * dy)**2
    denominator = (a * b)**2
    return np.sqrt(numerator / denominator)


def ellipses_minimum_distance(center1, a1, theta1, area1, center2, a2, theta2,
                              area2):
    """
    计算两个椭圆之间的最短距离。
    
    Parameters:
        center1, center2: 两个椭圆的中心 (x, y)
        a1, a2: 两个椭圆的长轴长度
        theta1, theta2: 两个椭圆的倾斜角（弧度）
        area1, area2: 两个椭圆的面积
    
    Returns:
        两个椭圆之间的最短距离
    """

    # 判断输入是否为 CasADi SX 或 DM 对象
    is_casadi = isinstance(center1, (ca.SX, ca.DM)) or isinstance(
        center2, (ca.SX, ca.DM))

    if is_casadi:
        # 使用 CasADi 的函数进行符号运算

        # 计算短轴长度
        b1 = area1 / (ca.pi * a1)  # 正确使用 CasADi pi
        b2 = area2 / (ca.pi * a2)

        # 计算方向向量
        direction = center2 - center1

        # 计算中心间距的范数
        center_distance = ca.norm_2(direction)

        # 判断中心是否重合
        center_distance_zero = ca.if_else(ca.eq(center_distance, 0), True,
                                          False)

        # 计算椭圆1在方向上的半径
        cos_theta1 = ca.cos(theta1)
        sin_theta1 = ca.sin(theta1)
        x_dir1 = direction[0] * cos_theta1 + direction[1] * sin_theta1
        y_dir1 = -direction[0] * sin_theta1 + direction[1] * cos_theta1
        denominator1 = (b1 * x_dir1)**2 + (a1 * y_dir1)**2
        denominator1 = ca.fmax(denominator1, 1e-8)  # 防止除以零
        r1 = ca.sqrt((a1 * b1)**2 / denominator1)

        # 计算椭圆2在反方向上的半径
        neg_direction = -direction
        cos_theta2 = ca.cos(theta2)
        sin_theta2 = ca.sin(theta2)
        x_dir2 = neg_direction[0] * cos_theta2 + neg_direction[1] * sin_theta2
        y_dir2 = -neg_direction[0] * sin_theta2 + neg_direction[1] * cos_theta2
        denominator2 = (b2 * x_dir2)**2 + (a2 * y_dir2)**2
        denominator2 = ca.fmax(denominator2, 1e-8)  # 防止除以零
        r2 = ca.sqrt((a2 * b2)**2 / denominator2)

        # 计算最小距离
        min_distance = ca.if_else(center_distance_zero, 0,
                                  center_distance - (r1 + r2))

        # 确保最小距离非负
        min_distance = ca.fmax(min_distance, 0)

        return min_distance
    else:
        # 使用 NumPy 进行数值运算

        # 计算短轴长度
        b1 = area1 / (np.pi * a1)
        b2 = area2 / (np.pi * a2)

        # 计算方向向量
        direction = np.array(center2) - np.array(center1)

        # 判断中心是否重合
        norm_direction = np.linalg.norm(direction)
        if norm_direction == 0:
            return 0

        # 计算椭圆1在方向上的半径
        cos_theta1 = np.cos(theta1)
        sin_theta1 = np.sin(theta1)
        x_dir1 = direction[0] * cos_theta1 + direction[1] * sin_theta1
        y_dir1 = -direction[0] * sin_theta1 + direction[1] * cos_theta1
        denominator1 = (b1 * x_dir1)**2 + (a1 * y_dir1)**2
        denominator1 = max(denominator1, 1e-8)  # 防止除以零
        r1 = np.sqrt((a1 * b1)**2 / denominator1)

        # 计算椭圆2在反方向上的半径
        neg_direction = -direction
        cos_theta2 = np.cos(theta2)
        sin_theta2 = np.sin(theta2)
        x_dir2 = neg_direction[0] * cos_theta2 + neg_direction[1] * sin_theta2
        y_dir2 = -neg_direction[0] * sin_theta2 + neg_direction[1] * cos_theta2
        denominator2 = (b2 * x_dir2)**2 + (a2 * y_dir2)**2
        denominator2 = max(denominator2, 1e-8)  # 防止除以零
        r2 = np.sqrt((a2 * b2)**2 / denominator2)

        # 计算最小距离
        min_distance = norm_direction - (r1 + r2)
        min_distance = max(min_distance, 0)

        return min_distance


def initialization_before_each_episode(args, env, episode, network=None):
    '''
    Parameters: 
    Return: 
    LastEditTime: 
    Description: initialization_before_each_episode 
    including: network episode_data epsilon_decay
    '''
    if args.method == 'qmix':
        # 初始化每个 episode 的存储结构
        episode_data = {
            "s": [],
            "s_next": [],
            "o": [],
            "o_next": [],
            "u": [],
            "u_onehot": [],
            "r": [],
            "terminated": [],
            "padded": [],
        }
        # network
        network.init_hidden(episode_num=1)
    elif args.method == 'sac':
        # 初始化每个 episode 的存储结构
        # network
        print('Oops something went wrong!')
    elif args.method == 'None':
        # 初始化每个 episode 的存储结构
        # network
        # 初始化每个 episode 的存储结构
        episode_data = {
            "s": [], #这个s是为了绘图
            "state": [],
            "state_next": [],
            "obs": [],
            "obs_next": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
            "agent_mask": []
        }
        ## Public work
        # epsilon_decay
    if args.epsilon_decay == True and args.method != 'None':
        current_epsilon = linear_decay_epsilon(args.epsilon, args.min_epsilon,
                                               int(args.n_episodes * 0.75),
                                               episode)
        [setattr(agent, "epsilon", current_epsilon) for agent in env.agents]
    env.reset()
    return episode_data


def get_each_agent_data(args, agent_id, action, actions, other_agents_data):
    '''
    Parameters: 
    Return: 
    LastEditTime: 
    Description: initialization_before_each_episode 
    including: network episode_data epsilon_decay
    '''
    if args.method == 'qmix':
        actions.append(action)
        agent_id_onehot = [0] * args.n_agents
        agent_id_onehot[agent_id] = 1
        action_onehot = [0] * args.n_actions
        action_onehot[action] = 1
        combined_onehot = action_onehot + agent_id_onehot
        other_agents_data.append(combined_onehot)

    elif args.method == 'sac':
        # 初始化每个 episode 的存储结构
        actions.append(action)
    elif args.method == 'None':
        """
            TODO
            the action type of tradtional_method is not the index of the action space
            it is the true input
        """
        actions.append(action)
        # network
        print('Oops something went wrong!')
    return actions, other_agents_data

def normalize_true_actions(true_actions):
    """
    将 true_actions（每个元素 shape=(4,)）从 [low,high] 映射到 [0,1]。
    """
    low  = np.array([-4.0, -4.0, 15.0, -math.pi/2], dtype=np.float32)
    high = np.array([ 4.0,  4.0, 60.0,  math.pi/2], dtype=np.float32)
    normalized = []
    for act in true_actions:
        # (act - low) / (high - low) -> [0,1]
        norm = (act - low) / (high - low)
        normalized.append(norm.astype(np.float32))
    return normalized

def step_and_collation_episode_data(args, episode_data, env, actions,
                                    other_agents_data):
    '''
    Parameters: 
    Return: 
    LastEditTime: 
    Description: initialization_before_each_episode 
    including: network episode_data epsilon_decay
    '''
    episode_rewards = 0
    if args.method == 'qmix':
        episode_data["u_onehot"].append(other_agents_data)
        episode_data["s"].append(env.get_global_state())  # 全局状态在CPU上获取
        next_state, rewards, done = env.step(actions,
                                             args.gamma)  # 环境步进在CPU上进行
        episode_rewards += sum(rewards)
        episode_data["s_next"].append(env.get_global_state())
        episode_data["o"].append([agent.get_state() for agent in env.agents])
        episode_data["o_next"].append(
            [next_agent_state for next_agent_state in next_state])
        episode_data["u"].append(actions)
        episode_data["r"].append(rewards)
        episode_data["terminated"].append([done] * args.n_agents)
        episode_data["padded"].append([0] * args.n_agents)
    elif args.method == 'sac':
        next_state, rewards, done = env.step(actions,
                                             args.gamma)  # 环境步进在CPU上进行
        episode_rewards += sum(rewards)
    elif args.method == 'None':
        """
            TODO 在这里获取episode的所有属性数据均在env中完成
            1.修改env.get_global_state()
            2.修改计算rewards
            3.修改agent.get_state() 并通过调用agent.get_state() 更新agent_obs
        """
        episode_data["s"].append(env.get_global_state4draw()) 
        episode_data["state"].append(env.get_global_state())
        episode_data["obs"].append({agent: env.agents_obs[agent] for agent in env.uavs})
        next_state, rewards, done = env.step(actions,
                                             args.gamma)  # 环境步进在CPU上进行
        # TODO 更新env.agents_state[agent]
        env.agents_last_state = env.agents_state.copy()
        env.agents_state = {
            agent:
                # 如果该 agent 处于 success，就把速度置零，否则用真实速度
                np.concatenate((
                    np.array([0, 0]),
                    env.agents[i].position,
                    np.array([env.agents[i].ellipse_length, env.agents[i].theta])
                )) if env.agents_working_state[agent] == 'success'
                else np.concatenate((
                    env.agents[i].velocity,
                    env.agents[i].position,
                    np.array([env.agents[i].ellipse_length, env.agents[i].theta])
                ))
            for i, agent in enumerate(env.uavs)
        }
        env.pay_attention() #更新了obs_state,neighbor_state
        env.agents_obs = {agent: np.concatenate((np.array(normalization(env.agents_state[agent], env.normalization_scale4state, np.array([0,0,-env.target_point[i][0],-env.target_point[i][1],0,0]))),env.neighbor_state[agent],env.obs_state[agent])) for i,agent in enumerate(env.uavs)}
        
        episode_data["state_next"].append(env.get_global_state())
        episode_data["obs_next"].append({agent: env.agents_obs[agent] for agent in env.uavs})
        trans_actions = normalize_true_actions(actions)
        episode_data["actions"].append({agent: trans_actions[i] for (i,agent) in enumerate(env.uavs)})
        episode_data["rewards"].append({agent: rewards[i] for (i,agent) in enumerate(env.uavs)})
        episode_data["terminals"].append( {agent: True if env.agents_working_state[agent] != 'working' else False for agent in env.uavs})
        episode_data["agent_mask"].append({agent: True for agent in env.uavs})
    return episode_data, episode_rewards, done


def preprocessing_episode_data(args, episode_data, episode_rewards):
    '''
    Parameters: 
    Return: 
    LastEditTime: 
    Description: initialization_before_each_episode 
    including: network episode_data epsilon_decay
    '''
    if args.method == 'qmix':
        if len(episode_data["s"]) < args.max_episode_len:
            # 找到 'terminated' 中第一个为 True 的索引
            terminated_index = next(
                (i for i, terminated in enumerate(episode_data["terminated"])
                 if any(terminated)),
                None,
            )
            # terminated_index = terminated_index+1
            for _ in range(args.max_episode_len - len(episode_data["s"])):
                episode_data["s"].append(np.zeros_like(episode_data["s"][0]))
                episode_data["s_next"].append(
                    np.zeros_like(episode_data["s_next"][0]))  # 添加这一行
                episode_data["o"].append(np.zeros_like(episode_data["o"][0]))
                # 使用 'terminated_index' 填补 o_next 数据
                if terminated_index is not None:
                    # 获取第一个为 True 的 'terminated' 时间步的 o_next 数据
                    terminated_obs = episode_data["o_next"][terminated_index]
                    # 如果 terminated_index 时间步只有一个智能体的数据
                    if len(terminated_obs) == 1:
                        # 复制当前智能体的 obs 以填补缺失的 obs
                        terminated_obs.append(
                            episode_data["o_next"][terminated_index - 1][1])
                        episode_data["o_next"][
                            terminated_index] = terminated_obs
                        episode_data["r"][terminated_index].append(0)
                episode_data["o_next"].append(
                    np.zeros_like(episode_data["o_next"][0]))
                episode_data["r"].append([0] * args.n_agents)
                episode_data["u"].append(np.zeros_like(episode_data["u"][0]))
                episode_data["terminated"].append([1] * args.n_agents)
                episode_data["padded"].append([1] * args.n_agents)
                episode_data["u_onehot"].append(
                    np.zeros_like(episode_data["u_onehot"][0]))
            episode_data["rewards"] = episode_rewards

    elif args.method == 'sac':
        # 初始化每个 episode 的存储结构
        # network
        print('Oops something went wrong!')
    elif args.method == 'None':
        """
            TODO
            the action type of tradtional_method is not the index of the action space
            it is the true input
        """
        # network
        print('Oops something went wrong!')
    return episode_data


def save_training_data(path_on_f_drive, replay_buffer, map_filename, method,
                       env):
    """
    保存训练数据到指定目录。如果指定目录不存在，则保存到默认目录。
    
    参数:
    - path_on_f_drive (str): F盘路径,用于保存数据。
    - replay_buffer: 包含所有训练数据的 replay buffer。
    - Args: 包含 map_filename 的对象，用于生成文件名。
    """
    # 获取当前时间戳
    if method != 'None':
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # 生成保存文件的名称，包括时间戳和 map 文件名
        filename = f"all_episodes_data_{map_filename}_{timestamp}.json"
    else:
        filename = f"all_episodes_data_{map_filename}.json"
    filename_obs = f"{map_filename}_obs.json"
    filename_target = f"{map_filename}_target.json"
    # 默认保存目录
    default_save_dir = "all_train_data"

    if os.path.exists(path_on_f_drive):
        # 如果 F 盘路径存在，则保存到 F 盘
        save_path = os.path.join(path_on_f_drive, filename)
        save_path_obs = os.path.join(path_on_f_drive, filename_obs)
        save_path_target = os.path.join(path_on_f_drive, filename_target)
        print(f"Saving data to {save_path} on F drive.")
    else:
        # 如果 F 盘路径不存在，则保存到当前目录
        save_path = os.path.join(default_save_dir, filename)
        save_path_obs = os.path.join(default_save_dir, filename_obs)
        save_path_target = os.path.join(default_save_dir, filename_target)
        print(
            f"{path_on_f_drive} path not found. Saving data to {save_path} in current directory."
        )
    # 确保路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 获取所有经验数据并转换格式
    data = list(replay_buffer.get_all_data())  # 获取所有经验
    data = numpy_to_list(data)  # 假设 numpy_to_list 是用于转换数据格式的函数
    target_data = [numpy_to_list(agent.target_pos) for agent in env.agents]
    obs_data = numpy_to_list(env.obstacle_coor)
    # 将数据保存到指定路径
    with open(save_path, "w") as f:
        json.dump(data, f)
    with open(save_path_obs, "w") as f:
        json.dump(obs_data, f)
    with open(save_path_target, "w") as f:
        json.dump(target_data, f)
    print(f"数据已保存到 {save_path}")
def normalization(original_data, scale, addition):

    """
    对原始数据进行归一化处理，公式为：
      normalized[i] = (addition[i] + original_data[i]) / scale[i]
    
    参数:
      original_data: 一维 NumPy 数组，原始数据
      scale: 一维 NumPy 数组，与 original_data 尺寸一致，用作归一化的分母
      addition: 一维 NumPy 数组，与 original_data 尺寸一致，用作归一化的减法项
      
    返回:
      normalized_data: 一维 NumPy 数组，归一化后的数据
    """
    normalized_data = (addition + original_data) / scale
    return normalized_data
def is_overlap(agent1, agent2, num_samples=200):
    # ——— 提取椭圆参数 ———
    h1, k1, a1, rot1 = agent1[2], agent1[3], agent1[4], agent1[5]
    b1 = 900 / a1
    h2, k2, a2, rot2 = agent2[2], agent2[3], agent2[4], agent2[5]
    b2 = 900 / a2

    # ——— 计算旋转椭圆外接矩形半宽半高 ———
    half_w1 = np.sqrt(a1**2 * np.cos(rot1)**2 + b1**2 * np.sin(rot1)**2)
    half_h1 = np.sqrt(a1**2 * np.sin(rot1)**2 + b1**2 * np.cos(rot1)**2)
    half_w2 = np.sqrt(a2**2 * np.cos(rot2)**2 + b2**2 * np.sin(rot2)**2)
    half_h2 = np.sqrt(a2**2 * np.sin(rot2)**2 + b2**2 * np.cos(rot2)**2)

    # ——— 外接矩形重叠区间 ———
    x_min = max(h1 - half_w1, h2 - half_w2)
    x_max = min(h1 + half_w1, h2 + half_w2)
    y_min = max(k1 - half_h1, k2 - half_h2)
    y_max = min(k1 + half_h1, k2 + half_h2)

    # 如果矩形不重叠，直接返回
    if x_min >= x_max or y_min >= y_max:
        return False, 0.0

    # 划网格采样
    xs = np.linspace(x_min, x_max, num_samples)
    ys = np.linspace(y_min, y_max, num_samples)
    xv, yv = np.meshgrid(xs, ys)
    xv_flat, yv_flat = xv.ravel(), yv.ravel()

    # 椭圆方程左侧计算函数
    def lhs(x, y, h, k, a, b, theta):
        Xp = (x - h) * np.cos(theta) + (y - k) * np.sin(theta)
        Yp = -(x - h) * np.sin(theta) + (y - k) * np.cos(theta)
        return (Xp/a)**2 + (Yp/b)**2

    # 计算每点是否在两椭圆内
    lhs1 = lhs(xv_flat, yv_flat, h1, k1, a1, b1, rot1)
    lhs2 = lhs(xv_flat, yv_flat, h2, k2, a2, b2, rot2)
    mask = (lhs1 <= 1) & (lhs2 <= 1)
    n_overlap = np.count_nonzero(mask)

    # 计算单个网格单元面积
    cell_area = ((x_max - x_min) / (num_samples - 1)) * ((y_max - y_min) / (num_samples - 1))
    intersection_area = n_overlap * cell_area

    # overlap: 是否有交集；intersection_area: 交集面积近似
    overlap = n_overlap > 0
    return overlap, intersection_area
def is_collision(agent1, obstacle_coor):
        cx, cy = agent1[2],agent1[3]
        # 椭圆的长轴半径和短轴半径
        a = agent1[4]  # 长轴半径
        b = (900 / a)  # 根据面积计算短轴半径
        theta = agent1[5]  
        for obstacle in obstacle_coor:
            ox, oy = obstacle
            # 计算障碍物相对于椭圆中心的偏移
            dx = ox - cx
            dy = oy - cy
            # 将全局坐标转换为椭圆局部坐标（逆旋转 theta）
            x_local = dx * np.cos(theta) + dy * np.sin(theta)
            y_local = -dx * np.sin(theta) + dy * np.cos(theta)
            # 判断该点是否在标准椭圆内： (x_local/a)^2 + (y_local/b)^2 <= 1
            if (x_local**2) / (a**2) + (y_local**2) / (b**2) <= 1:
                return True  # 障碍物在椭圆内部
        return False
def fill_offpolicy_buffer(memory, 
                          obs_all: Dict[str, np.ndarray],
                          actions_all: Dict[str, np.ndarray],
                          obs_next_all: Dict[str, np.ndarray],
                          rewards_all: Dict[str, np.ndarray],
                          terminals_all: Dict[str, np.ndarray],
                          agent_mask_all: Dict[str, np.ndarray],
                          state_all: np.ndarray = None,
                          state_next_all: np.ndarray = None,
                          avail_actions_all: Dict[str, np.ndarray] = None,
                          avail_actions_next_all: Dict[str, np.ndarray] = None):
    """
    直接填充 MARL_OffPolicyBuffer.data。

    参数:
      memory: 已初始化的 MARL_OffPolicyBuffer 实例
      obs_all: dict, 每个 agent 的 obs 数组, shape=(n_envs, T, obs_dim)
      actions_all: dict, 每个 agent 的 action 数组, shape=(n_envs, T, act_dim)
      obs_next_all: dict, 每个 agent 的 next_obs 数组, shape=(n_envs, T, obs_dim)
      rewards_all: dict, 每个 agent 的 rewards 数组, shape=(n_envs, T)
      terminals_all: dict, 每个 agent 的 done 数组 (bool), shape=(n_envs, T)
      agent_mask_all: dict, 每个 agent 的 mask 数组 (bool), shape=(n_envs, T)
      state_all: 全局状态数组, shape=(n_envs, T, state_dim) 或 None
      state_next_all: 全局 next_state 数组, shape=(n_envs, T, state_dim) 或 None
      avail_actions_all: dict, 每个 agent 的 avail_actions 数组 (bool), shape=(n_envs, T, act_dim)
      avail_actions_next_all: dict, 每个 agent 的 avail_actions_next 数组 (bool), shape=(n_envs, T, act_dim)
    """
    # 1) 确定要填充多少个时间步 T（不得超过 buffer_size）
    #    这里假设所有输入数组的 T 维都是一致的
    any_agent = next(iter(obs_all))
    n_envs, T, _ = obs_all[any_agent].shape
    assert T <= memory.n_size, "离线数据时间步 T 不得大于 buffer_size"

    # 2) 逐字段写入底层 memory.data
    for k in memory.agent_keys:
        memory.data['obs'][k][:, :T]        = obs_all[k]
        memory.data['actions'][k][:, :T]    = actions_all[k]
        memory.data['obs_next'][k][:, :T]   = obs_next_all[k]
        memory.data['rewards'][k][:, :T]    = rewards_all[k]
        memory.data['terminals'][k][:, :T]  = terminals_all[k]
        memory.data['agent_mask'][k][:, :T] = agent_mask_all[k]

        if memory.use_actions_mask:
            memory.data['avail_actions'][k][:, :T]      = avail_actions_all[k]
            memory.data['avail_actions_next'][k][:, :T] = avail_actions_next_all[k]

    if memory.store_global_state:
        # state_all 和 state_next_all:  shape=(n_envs, T, state_dim)
        memory.data['state']     [:, :T] = state_all
        memory.data['state_next'][:, :T] = state_next_all

    # 3) 更新指针与大小
    memory.size = T
    memory.ptr  = T % memory.n_size
