'''
Author: Shukun
Date: 2025-01-02 13:43:23
LastEditors: Shukun
LastEditTime: 2025-03-31 16:02:03
Description: 请填写简介
'''
import os
import torch.nn as nn  # 导入pytorch的神经网络模块
import torch.nn.functional as F  # 导入pytorch的功能模块
from torch.optim.lr_scheduler import StepLR
from basic_class.qmix_net import QMixNet
import numpy as np
import torch
from datetime import datetime


class RNN(nn.Module):
    """
    所有智能体共享的RNN网络，input_shape=obs_shape+n_actions+n_agents
    RNN使用One-Hot格式编码
    """

    def __init__(self, data_shape):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(data_shape[0], data_shape[1])
        self.rnn = nn.GRUCell(data_shape[1], data_shape[1])
        self.fc2 = nn.Linear(data_shape[1], data_shape[2])
        self.hidden_dim = data_shape[1]

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMIX(nn.Module):
    """
    QMIX网络，包括评估和目标网络
    """

    def __init__(self, args, device=None):
        super(QMIX, self).__init__()
        dim_rnn_input = args.obs_shape + args.n_agents + args.n_actions
        # 如果未指定 device，则根据 args.cuda 来设置
        if device is None:
            self.device = torch.device("cuda" if args.cuda else "cpu")
        else:
            self.device = torch.device(device)

        self.eval_rnn = RNN(
            [dim_rnn_input, args.rnn_hidden_dim,
             args.n_actions]).to(self.device)
        self.target_rnn = RNN(
            [dim_rnn_input, args.rnn_hidden_dim,
             args.n_actions]).to(self.device)

        self.eval_qmix_net = QMixNet(args).to(self.device)
        self.target_qmix_net = QMixNet(args).to(self.device)
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = args.state_shape
        self.action_dim = args.n_actions

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(
            self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=10,
                                gamma=1)  # 每100步将学习率降低为原来的0.9倍
        self.loss = 0
        self.gradient_norm = 0
        self.eval_hidden = None
        self.target_hidden = None

        self.model_dir = args.model_dir
        if self.args.load_model:
            self.load_model(args)

    def load_model(self, args):
        if args.The_Chosen_One == True:
            if os.path.exists(self.model_dir + '\\' + args.rnn_model):
                # 使用 weights_only=True 加载状态字典，以减少安全风险
                self.eval_rnn.load_state_dict(
                    torch.load(self.model_dir + '\\' + args.rnn_model,
                               map_location=self.device))
                self.eval_qmix_net.load_state_dict(
                    torch.load(self.model_dir + '\\' + args.qmix_model,
                               map_location=self.device))
                # 显式更新目标网络参数
                self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
                self.target_qmix_net.load_state_dict(
                    self.eval_qmix_net.state_dict())
                print('成功加载模型：{}'.format(self.model_dir + args.rnn_model))
            else:
                raise Exception("模型不存在！")
        else:
            if os.path.exists(self.model_dir + '\\' + 'best' + '\\' +
                              args.rnn_model):
                # 使用 weights_only=True 加载状态字典，以减少安全风险
                self.eval_rnn.load_state_dict(
                    torch.load(self.model_dir + '\\' + 'best' + '\\' +
                               args.rnn_model,
                               map_location=self.device))
                self.eval_qmix_net.load_state_dict(
                    torch.load(self.model_dir + '\\' + 'best' + '\\' +
                               args.qmix_model,
                               map_location=self.device))
                # 显式更新目标网络参数
                self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
                self.target_qmix_net.load_state_dict(
                    self.eval_qmix_net.state_dict())
                print('成功加载模型：{}'.format(self.model_dir))
            else:
                raise Exception("模型不存在！")

    def learn(self, batch, max_episode_len, train_step):
        self.train()
        episode_num = len(batch['o'])
        self.init_hidden(episode_num)

        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif key == 'u':
                batch[key] = torch.tensor(batch[key],
                                          dtype=torch.long,
                                          device=self.device)
            else:
                batch[key] = torch.tensor(np.array(batch[key]),
                                          dtype=torch.float32,
                                          device=self.device)

        s, s_next, u, r, terminated = batch['s'], batch['s_next'], batch[
            'u'], batch['r'], batch['terminated']
        mask = 1 - batch["padded"][:, :, :1].float().to(self.device)

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        u = u.unsqueeze(-1)
        u_t = u.long()  # 确保 u 是 int64 类型
        q_evals = torch.gather(q_evals, dim=3, index=u_t).squeeze(3)
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)
        r_summed = r.sum(dim=-1, keepdim=True)
        targets = r_summed + self.args.gamma * q_total_target * (
            1 - terminated[:, :, :1].float())
        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error

        loss = (masked_td_error**2).sum() / mask.sum()
        self.loss = loss
        # print(f'loss{loss}')
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters,
                                       self.args.grad_norm_clip)
        self.optimizer.step()  # 更新参数
        self.scheduler.step()

        for param in self.eval_parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                self.gradient_norm += param_norm.item()**2
        self.gradient_norm = self.gradient_norm**0.5

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(
                self.eval_qmix_net.state_dict())

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim),
            device=self.device)
        self.target_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim),
            device=self.device)

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            q_eval, h = self.eval_rnn(inputs, self.eval_hidden)
            self.eval_hidden = h.view(episode_num, self.n_agents,
                                      self.args.rnn_hidden_dim)

            q_target, h_target = self.target_rnn(inputs_next,
                                                 self.target_hidden)
            self.target_hidden = h_target.view(episode_num, self.n_agents,
                                               self.args.rnn_hidden_dim)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def get_q_values_single(self, state, index):
        """
        单步Q值计算和hidden_state更新，用于智能体选择动作。
        """
        # 使用 eval_rnn 计算 Q 值
        q_eval, hidden_state = self.eval_rnn(state, self.eval_hidden[:,
                                                                     index, :])
        self.eval_hidden[:, index, :] = hidden_state.view(
            1, self.args.rnn_hidden_dim)

        return q_eval

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
            batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(
                    torch.zeros_like(u_onehot[:,
                                              transition_idx]).to(self.device))
            else:
                inputs.append(u_onehot[:, transition_idx - 1].to(self.device))
            inputs_next.append(u_onehot[:, transition_idx].to(self.device))
        """
        我这里的One_Hot已经包含了动作及编号
        """
        # if self.args.reuse_network:
        #     inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).to(self.device))
        #     inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).to(self.device))
        inputs = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs],
            dim=1)
        inputs_next = torch.cat([
            x.reshape(episode_num * self.args.n_agents, -1)
            for x in inputs_next
        ],
                                dim=1)
        # 转换为 float32 类型
        inputs = inputs.float()
        inputs_next = inputs_next.float()
        return inputs, inputs_next

    def save_model(self, train_step):
        # 计算保存文件的编号
        num = str(train_step // self.args.save_cycle)

        # 获取当前日期并格式化为字符串
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 确保模型目录存在
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 保存模型参数到文件，文件名包括编号和日期
        qmix_filename = os.path.join(
            self.model_dir, f'{train_step}qmix_net_params_{date_str}.pkl')
        rnn_filename = os.path.join(
            self.model_dir, f'{train_step}rnn_net_params_{date_str}.pkl')

        torch.save(self.eval_qmix_net.state_dict(), qmix_filename)
        torch.save(self.eval_rnn.state_dict(), rnn_filename)

        print(f'模型已保存：{qmix_filename}')
        print(f'模型已保存：{rnn_filename}')
