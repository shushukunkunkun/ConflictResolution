'''
Author: Shukun
Date: 2024-12-30 18:36:04
LastEditors: Shukun
LastEditTime: 2025-04-22 13:15:28
Description: Decentralized MPC + MARL
'''
import os
import sys

import traceback
import torch
import wandb
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 返回上一级目录，即项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
from other_function import initialization_before_each_episode, get_each_agent_data, step_and_collation_episode_data, preprocessing_episode_data, process_batch_to_tensor, numpy_to_list, save_training_data
from basic_class.qmix import QMIX  # 从qmix.py导入QMIX类
from basic_class.multi_agent_env import MultiAgentEnv
from basic_class.Replay_Buffer import ReplayBuffer


class Args:

    def __init__(self, method='None'):
        # 公有基础属性
        self.method = method
        self.n_episodes = 1
        self.max_episode_len = 1200  # 单次训练最多步数
        self.train_step = 0
        self.buffer_size = 20000  # 设置经验回放缓冲区的大小
        self.batch_size = 128  # 设置从缓冲区中抽取的批次大小
        self.n_agents = 4
        self.n_actions = 27
        self.state_shape = 14
        self.obs_shape = 10
        self.lr = 0.001
        self.gamma = 0.9
        self.epsilon = 0.3
        self.min_epsilon = 0.1
        self.epsilon_decay = False  # 是否衰减epsilon
        self.target_update_cycle = 200
        self.grad_norm_clip = 2
        self.save_cycle = 2
        self.load_replay_buffer = False
        self.load_model = False
        self.auto_shutdown = False
        self.ShowMe = True
        self.The_Chosen_One = False
        self.map_filename = "NarrowCorridor_2"
        self.use_wandb = False  # 控制是否上传数据至W&B
        # 检查CUDA是否可用并设置设备
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.path_on_f_drive = fr"C:\Users\pc\Desktop\PythonProject\ConflictResolution\all_train_data\{self.map_filename}_train_data"
        # 根据 method 参数设置特定属性
        if self.method == 'qmix':
            # QMIX特有的属性
            self.rnn_hidden_dim = 128
            self.qmix_hidden_dim = 128
            self.hypernet_hidden_dim = 128
            self.last_action = True  # 是否考虑RNN的上个动作
            self.reuse_network = True  # 是否考虑RNN的智能体编号
            self.two_hyper_layers = False  # 是否使用两层全连接层
            self.rnn_model = "330rnn_net_params_20241016_001915.pkl"
            self.qmix_model = "330qmix_net_params_20241016_001915.pkl"
            self.replay_buffer_dir = r'C:\\Users\\pc\\Desktop\\PythonProject\\RL4ConflictResolution\\all_train_data\\V1TOV2\\Over5000.0_sum_data_V4.0.json'
            self.model_dir = os.path.join(
                r"C:\Users\pc\Desktop\PythonProject\RL4ConflictResolution\models\qmix",
                self.map_filename)
        elif self.method == 'sac':
            self.replay_buffer_dir = r'C:\\Users\\pc\\Desktop\\PythonProject\\RL4ConflictResolution\\all_train_data\\V1TOV2\\Over5000.0_sum_data_V4.0.json'


def main():
    args = Args('None')
    # 环境初始化
    if args.use_wandb:
        wandb.login(key="0a09fce74e57f774b4cd367a096ffe6d7025a427")
        wandb.init(project="ConflictResolution", entity="shushukunkunkun")
    # 网络初始化
    if args.method == 'qmix':
        network = QMIX(args).to(args.device)  # 确保QMIX模型在正确的设备上
    elif args.method == 'sac':
        print('Oops something went wrong!')
    # TODO sac network
    env = MultiAgentEnv(args)  # 初始化环境
    replay_buffer = ReplayBuffer(args)  # 初始化replay_buffer
    """
        The main loop code below
    """
    try:
        for episode in range(args.n_episodes):
            episode_data = initialization_before_each_episode(
                args, env, episode, network if args.method != 'None' else None)
            episode_rewards = 0  # 初始化本次 episode 的总奖励
            this_episode_step = 0
            for step in range(args.max_episode_len):
                actions = []
                other_agents_data = []
                """other_agents_data refer to different network's input
                for example , in qmix method, it refers to the one_hot encoding"""
                for agent_id, agent in enumerate(env.agents):
                    agent_state = agent.get_state()
                    action = agent.select_action(
                        agent_state,
                        network if args.method != 'None' else None,
                        args.n_agents, args.n_actions, step, env.agents
                        if args.method == 'None' else None)  # 确保动作选择在正确的设备上进行
                    actions, other_agents_data = get_each_agent_data(
                        args, agent_id, action, actions, other_agents_data)
                episode_data, episode_rewards, done = step_and_collation_episode_data(
                    args, episode_data, env, actions, other_agents_data)
                print(f"Step {step}")
                if done:
                    this_episode_step = step
                    break
            # 数据预处理
        
            episode_data = preprocessing_episode_data(args, episode_data,
                                                    episode_rewards)
            replay_buffer.store(episode_data)
            # 在经验缓冲区中有足够的样本后，才开始训练
            if replay_buffer.size() >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                batch = process_batch_to_tensor(batch, device=args.device)
                network.learn(batch, args.max_episode_len, args.train_step)
                args.train_step += 1
            # 每隔一定周期保存模型
            if args.method != 'None' and episode % args.save_cycle == 0:
                network.save_model(args.train_step)
                print(f"Episode {episode}: Model saved")
            # 上传数据到W&B
            if args.use_wandb:
                wandb.log({
                    "Episode":
                    episode,
                    "Learning_rate":
                    network.optimizer.param_groups[0]["lr"],
                    "Epsilon":
                    env.agents[0].epsilon,
                    "Whole Step":
                    this_episode_step,
                    "Replay_buffer_size":
                    replay_buffer.size(),
                    "Loss":
                    network.loss,
                    "Gradient_Norm":
                    network.gradient_norm,
                    "Episode Reward":
                    episode_rewards,
                })
            print(f"Episode {episode} completed.")
    except KeyboardInterrupt:
        print("训练被中断，正在保存数据...")
    except Exception as e:
        print("发生了异常")
        # 打印异常的详细信息
        traceback.print_exc()
    finally:
        save_training_data(args.path_on_f_drive, replay_buffer,
                           args.map_filename, args.method, env)
        print("训练完成。")
        if args.use_wandb:
            wandb.finish()
        # 确保模型保存
        args.method != 'None' and network.save_model(args.train_step)


if __name__ == '__main__':
    main()
