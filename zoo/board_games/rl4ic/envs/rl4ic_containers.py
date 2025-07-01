'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-07-01
FilePath: /RL4IC-LightZero/zoo/board_games/rl4ic/envs/rl4ic_containers.py
Description: 
Copyright (c) 2025 by CAS4ET lab, ShanghaiTech University, All Rights Reserved. 
'''
import numpy as np


def convert_4d_index_to_1d(index: tuple, shape:tuple=(5,5,5,5)) -> int:
    # convert 4d index to 1d index
    i, j, k, l = index
    _, d2, d3, d4 = shape
    return i * (d2 * d3 * d4) + j * (d3 * d4) + k * d4 + l

def convert_1d_index_to_4d(index: int, shape:tuple=(5,5,5,5)) -> tuple:
    # convert 1d index to 4d index
    _, d1, d2, d3 = shape 
    temp = index
    l = temp % d3
    temp = temp // d3
    k = temp % d2
    temp = temp // d2
    j = temp % d1
    temp = temp // d1
    i = temp
    return (i, j, k, l)

def InputGenerator(max_input: int, input_num: int) -> np.ndarray:
    # NOTE: the random number is start from 1 to max_input
    return np.random.randint(1, max_input, input_num)

class Container(object):

    def __init__(self, num_agents: int, num_layers: int, max_input: int):
        # set 
        self._num_agents = num_agents
        self._num_layers = num_layers
        self._max_input = max_input

        self._input = np.empty((0))# for pylance type hint

        # reset fifo and input
        self.reset()

    def find_duplicate_index(self, index_list: list):
        # find duplicate index in the list except 0
        exist_index = set()
        for index in index_list:
            if (index in exist_index) and (index != 0): # allow 0 to be duplicated
                return True
            else:
                exist_index.add(index)
        return False

    def get_static_action_mask(self):
        # get the static action mask for all rounds
        action_shape = (self._num_agents + 1, self._num_agents + 1, self._num_agents + 1, self._num_agents + 1)
        masked_index = []
        # consider same index
        for i,j,k,l in np.ndindex(action_shape):
            if self.find_duplicate_index([i,j,k,l]):
                masked_index.append(convert_4d_index_to_1d((i,j,k,l)))
        # generate action mask
        action_mask = np.ones(np.power(self._num_agents + 1, self._num_agents))
        action_mask[masked_index] = 0
        return action_mask

    def reset(self): # reset data and reinit input
        # reinit input
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
        # TODO: check the condition of the height touch
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
    
    def evaluate(self):
        pass

    def set_new_game(self):
        self.reset()


if __name__ == "__main__":
    # number = convert_4d_index_to_1d((4,4,4,0))
    # print(number)
    # print(convert_1d_index_to_4d(number))
    container = Container(num_agents=4, num_layers=8, max_input=16)
    action_mask = container.get_static_action_mask()
    for i in range(len(action_mask)):
        if action_mask[i] == 0:
            print(convert_1d_index_to_4d(i))
    # print(np.sum(action_mask))


        
