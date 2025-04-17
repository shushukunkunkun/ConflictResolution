'''
Author: Shukun
Date: 2025-01-01 15:48:37
LastEditors: Shukun
LastEditTime: 2025-01-01 15:52:11
Description: 请填写简介
'''
import collections
import random
import json  # 用于保存数据


class ReplayBuffer:

    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.buffer = collections.deque(maxlen=args.buffer_size)
        self.position = 0
        # 将replay_buffer的pre_load功能固定在初始化处
        if args.load_replay_buffer == True:
            with open(args.replay_buffer_dir, "r") as file:
                initial_data = json.load(file)
                for experience in initial_data:
                    self.store(experience)

    def store(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.buffer_size

    def get_all_data(self):
        return self.buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 将batch从list转换为dict，合并同一键的所有值
        batch_dict = {
            key: []
            for key in batch[0]
        }  # 初始化dict，keys为experience中的keys
        for experience in batch:
            for key, value in experience.items():
                if key == 'rewards':
                    continue
                batch_dict[key].append(value)

        return batch_dict

    def size(self):
        return len(self.buffer)
