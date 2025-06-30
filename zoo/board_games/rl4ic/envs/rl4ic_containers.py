'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-06-30
FilePath: /RL4IC-LightZero/zoo/board_games/rl4ic/envs/rl4ic_containers.py
Description: 
Copyright (c) 2025 by CAS4ET lab, ShanghaiTech University, All Rights Reserved. 
'''
import numpy as np


def InputGenerator(max_input: int, input_num: int) -> np.ndarray:
    # NOTE: the random number is start from 1 to max_input
    return np.random.randint(1, max_input, input_num)

class Container(object):

    def __init__(self, num_agents: int, num_layers: int, max_input: int, input_seed = None):
        # set 
        self._num_agents = num_agents
        self._num_layers = num_layers
        self._max_input = max_input
        self._input_seed = input_seed

        self._input = np.empty((0))# for pylance type hint

        # reset fifo and input
        self.reset(self._input_seed)

    def reset(self, input_seed): # reset data and reinit input
        # reinit input
        if input_seed is not None:
            np.random.seed(input_seed)
        self._input_num = self._num_layers * self._num_agents + np.random.randint(-self._num_agents, self._max_input)
        self._input = InputGenerator(self._max_input, self._input_num)
        
        # reset fifo
        self.reset_fifo()

    def reset_fifo(self):    
        # reset fifo
        self._fifo =[[] for _ in range(self._num_agents)] 
    
    def get_input(self): # get the input for agents
        if self._input.size < self._num_agents:
            return self._input.tolist() + [0] * (self._num_agents - self._input.size)
        else:
            return self._input[:self._num_agents].tolist()
        
    def observe_fifo(self):
        # return the top layer data and height bias of each channel
        fifo_data = []
        fifo_heights = []
        for i in range(self._num_agents):
            fifo_data.append(self._fifo[i][-1])
            fifo_heights.append(len(self._fifo[i]))
        base_height = min(fifo_heights)
        fifo_heights = [element - base_height for element in fifo_heights]
        return fifo_data, fifo_heights

    def height_check(self) -> list[bool]:
        # return whether each channel of _data is available
        available_list = []
        for i in range(self._num_agents):
            if len(self._fifo[i]) < self._num_layers:
                available_list.append(True)
            else:
                available_list.append(False)
        return available_list

    def is_game_over(self) -> bool:
        # return whether the game is over
        return self._input.size == 0 or (not any(self.height_check()))

    def pop_push(self, action: list) -> bool:
        """
        Pop and push the selected input to the data
        NOTE: sel_i has a bias (+1) because 0 represents do nothing.

        :return bool: whether the game is over
        """
        pop_list = []
        for i, sel_i in enumerate(action):
            if sel_i != 0:
                self._fifo[i].append(self._input[sel_i - 1])
                pop_list.append(sel_i - 1)
        self._input = np.delete(self._input, pop_list)

        return self.is_game_over()







        
