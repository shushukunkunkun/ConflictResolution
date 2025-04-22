'''
Author: Shukun
Date: 2025-04-22 15:20:24
LastEditors: Shukun
LastEditTime: 2025-04-22 15:21:36
Description: 请填写简介
'''

import json
import os
IL_replay_buffer_dir= './scripts/IL_replay_buffer'
IL_data= 'rewards343.3858.json'

json_path = os.path.join(IL_replay_buffer_dir,IL_data )
with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
print(1)