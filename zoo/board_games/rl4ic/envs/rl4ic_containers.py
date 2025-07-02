'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-07-02
FilePath: /RL4IC-LightZero/zoo/board_games/rl4ic/envs/rl4ic_containers.py
Description: 
Copyright (c) 2025 by CAS4ET lab, ShanghaiTech University, All Rights Reserved. 
'''
import numpy as np
import copy
from collections import OrderedDict

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
    return np.sort(np.random.randint(1, max_input, input_num))

class Container(object):

    def __init__(self, num_agents: int, num_layers: int, max_input: int, render:bool=False):
        # set 
        self._num_sub_agents = num_agents
        self._num_layers = num_layers
        self._max_input = max_input
        self._render = render

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

    def find_repeat_number(self, buffer: list):
        count_dict = {}
        for num in buffer:
            if num is not None:
                count_dict[num] = count_dict.get(num, 0) + 1

        duplicates = OrderedDict()
        for num, count in count_dict.items():
            if count > 1:
                duplicates[num] = count

        return duplicates

    def get_static_action_mask(self):
        # get the static action mask for all rounds
        action_shape = (self._num_sub_agents + 1, self._num_sub_agents + 1, self._num_sub_agents + 1, self._num_sub_agents + 1)
        masked_index = []
        # consider same index
        for i,j,k,l in np.ndindex(action_shape):
            if self.find_duplicate_index([i,j,k,l]):
                masked_index.append(convert_4d_index_to_1d((i,j,k,l)))
        # generate action mask
        action_mask = np.ones(np.power(self._num_sub_agents + 1, self._num_sub_agents))
        action_mask[masked_index] = 0
        return action_mask

    def reset(self): # reset data and reinit input
        # reinit input
        self._input_num = self._num_layers * self._num_sub_agents + np.random.randint(-2*self._num_sub_agents, self._max_input) 
        self._input = InputGenerator(self._max_input, self._input_num)
        
        self.reset_fifo() # reset fifo
        self._pop_counter = 0 # reset pop counter

    def reset_fifo(self):    
        # reset fifo
        self._fifo =[[] for _ in range(self._num_sub_agents)] 
    
    def get_input(self): # get the input for agents
        if self._input.size < self._num_sub_agents:
            return self._input.tolist() + [0] * (self._num_sub_agents - self._input.size)
        else:
            return self._input[:self._num_sub_agents].tolist()
        
    def observe_fifo(self):
        # return the top layer data and height bias of each channel
        fifo_data = []
        fifo_heights = []
        for i in range(self._num_sub_agents):
            fifo_data.append(self._fifo[i][-1] if len(self._fifo[i]) > 0 else 0)
            fifo_heights.append(len(self._fifo[i]))
        base_height = min(fifo_heights)
        fifo_heights = [element - base_height for element in fifo_heights]
        return fifo_data, fifo_heights

    def height_check(self) -> list[bool]:
        # return whether each channel of _data is available
        available_list = []
        for i in range(self._num_sub_agents):
            if len(self._fifo[i]) < self._num_layers:
                available_list.append(True)
            else:
                available_list.append(False)
        return available_list

    def is_game_over(self) -> bool:
        # return whether the game is over
        # TODO: check the condition of the height touch
        return self._input.size == 0 or (not any(self.height_check()))

    def pop_push(self, action: int) -> bool:
        """
        Pop and push the selected input to the data
        NOTE: sel_i has a bias (+1) because 0 represents do nothing.

        :return bool: whether the game is over
        """
        action_tuple = convert_1d_index_to_4d(action)
        # print(action_tuple)
        action_list = [action_tuple[i] for i in range(self._num_sub_agents)]
        pop_list = []

        buffer = self.get_input()
        for i, sel_i in enumerate(action_list):
            if sel_i != 0:
                # print(f"action:{action_tuple}, i:{i}, sel_i:{sel_i}, _input:{self._input}, buffer:{buffer}\n", end="")
                self._fifo[i].append(buffer[sel_i - 1])
                if buffer[sel_i - 1] != 0:
                    pop_list.append(sel_i - 1)

        # if len(pop_list) > self._input.size: # cutoff out of bound index
        #     pop_list = pop_list[:self._input.size]
        # print(f"pop_list:{pop_list}, _input:{self._input}")
        
        self._input = np.delete(self._input, pop_list)
        self._pop_counter += len(pop_list)
        return self.is_game_over()
    
    def evaluate(self):
        
        time_counter = 0
        fifo = copy.deepcopy(self._fifo)
        while any([len(fifo[i]) > 0 for i in range(self._num_sub_agents)]):

            # print([len(fifo[i]) for i in range(self._num_sub_agents)]


            # prepare buffer, which is the buttom of fifo
            buffer = np.zeros(self._num_sub_agents)
            buffer.fill(-1)
            for i in range(self._num_sub_agents):
                if len(fifo[i]) > 0:
                    buffer[i] = fifo[i][0]

            #  analysis data 
            repeat_dict = self.find_repeat_number(buffer.tolist())
            max_repeat = 0
            max_repeat_key  = set()
            for keys in repeat_dict.keys():
                if keys == -1:
                    continue
                else:
                    if repeat_dict[keys] > max_repeat:
                        max_repeat = repeat_dict[keys]
                        max_repeat_key = set()
                        max_repeat_key.add(keys)
                    elif repeat_dict[keys] == max_repeat:
                        max_repeat_key.add(keys)

            # print(f"max_repeat:{max_repeat}, max_repeat_key:{max_repeat_key}")

            # selected pop list
            pop_list = [0] * self._num_sub_agents
            if max_repeat in [self._num_sub_agents, self._num_sub_agents - 1]:
                for i in range(self._num_sub_agents):
                    pop_list[i] = (buffer[i] in max_repeat_key)
            elif max_repeat == 2:
                if len(max_repeat_key) == 1:
                    non_repeat_pop_flag = True
                    for i in range(self._num_sub_agents):
                        if buffer[i] in max_repeat_key:
                            pop_list[i] = 1
                        elif non_repeat_pop_flag and (buffer[i] != -1):
                        # elif non_repeat_pop_flag:
                            pop_list[i] = 1
                            non_repeat_pop_flag = False
                        else:
                            pop_list[i] = 0
                else: 
                    pop_list = [1] * self._num_sub_agents
            else:
                non_repeat_pop_flag = 2
                for i in range(self._num_sub_agents):
                    if non_repeat_pop_flag and (buffer[i] != -1):
                    # if non_repeat_pop_flag :
                        pop_list[i] = 1
                        non_repeat_pop_flag -= 1
                    else:
                        pop_list[i] = 0

            # print(f"fifo:{fifo}\nbuffer:{buffer}, max_repeat:{max_repeat}, max_repeat_key:{max_repeat_key}, pop_list:{pop_list}")
            # pop fifo
            # print(max_repeat)
            # print(buffer)
            # print(pop_list)
            for i in range(self._num_sub_agents):
                if pop_list[i] == 1:
                    fifo[i].pop(0)
            time_counter += 1 # counter add

            if self._render:
                self.render_fifo(fifo, time_counter)
        
        ideal_counter = int((self._pop_counter + self._num_sub_agents - 1) / self._num_sub_agents)
        reward = ideal_counter / time_counter
        reward = 9 * reward - 5 # scaling factor
        reward = 1 / (1 + np.exp(-reward))

        if self._render:
            print("********************************")
            print("Pop counter:", self._pop_counter)
            print("Time counter:", time_counter)
            print("Ideal counter:", ideal_counter)
            print("Reward:", reward)
            print("********************************")

        return reward

    def test_fifo(self):
        # fill in fifo with some random data
        for i in range(self._num_sub_agents):
            for j in range(self._num_layers):
                self._fifo[i].append(np.random.randint(1, self._max_input))
                self._pop_counter += 1
            self._fifo[i].sort()
            
            print(self._fifo[i])
    
    def render_fifo(self, fifo, time_counter):
        print("--------------------------------")
        print(f"time_counter: {time_counter}")
        for i in range(self._num_sub_agents):
            print(fifo[i])
        print("--------------------------------")
        

    def set_new_game(self):
        # reset all to start a new game
        self.reset()

    def get_render_msg(self):
        return self._pop_counter,self._input_num


if __name__ == "__main__":
    # number = convert_4d_index_to_1d((4,4,4,0))
    # print(number)
    # print(convert_1d_index_to_4d(number))
    np.random.seed(1)
    container = Container(num_agents=4, num_layers=8, max_input=8, render=True)
    print(container.observe_fifo())

    round = 1
    while True:
        print(container.get_input())
        action = convert_4d_index_to_1d((1,2,3,4))
        game_over_flag = container.pop_push(action)
        container.render_fifo(container._fifo, round)
        round += 1
        if game_over_flag:
            break

    print(container.get_input())


    # action_mask = container.get_static_action_mask()
    # for i in range(len(action_mask)):
    #     if action_mask[i] == 0:
    #         print(convert_1d_index_to_4d(i))
    # print(np.sum(action_mask))
    # container.test_fifo()
    print(container.evaluate())

    container.set_new_game()

    round = 1
    while True:
        print(container.get_input())
        action = convert_4d_index_to_1d((1,2,3,4))
        game_over_flag = container.pop_push(action)
        container.render_fifo(container._fifo, round)
        round += 1
        if game_over_flag:
            break

    print(container.get_input())


    # action_mask = container.get_static_action_mask()
    # for i in range(len(action_mask)):
    #     if action_mask[i] == 0:
    #         print(convert_1d_index_to_4d(i))
    # print(np.sum(action_mask))
    # container.test_fifo()
    print(container.evaluate())
    # print(container._fifo)


        
