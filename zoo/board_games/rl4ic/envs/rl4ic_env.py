'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-07-02
FilePath: /RL4IC-LightZero/zoo/board_games/rl4ic/envs/rl4ic_env.py
Description: Adapt the RL4IC task scheduling environment to the BaseEnv interface.
Copyright (c) 2025 by CAS4ET lab, ShanghaiTech University, All Rights Reserved. 
'''
import os
import sys
import copy
import numpy as np

import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.utils.default_helper import deep_merge_dicts
from easydict import EasyDict
from gymnasium import spaces
# This module should be replaced by the RL4IC environment
# from pettingzoo.classic.go import coords, go
from pettingzoo.utils.agent_selector import AgentSelector
from zoo.board_games.rl4ic.envs.rl4ic_containers import Container


@ENV_REGISTRY.register('RL4IC')
class RL4ICEnv(BaseEnv):

    config = dict(
        # env_id (str): The name of the RL4IC environment.
        env_id="RL4IC-v0",
        # num_agents (int): The number of agents. Which is also the number of containers.
        num_sub_agents=4,
        # num_layers (int): The number of layers in each container.
        num_layers=64,
        # max_input (int): The max number provided by the input generator.
        max_input=64,
        # input_seed (int): The seed of the input generator. it's optional 
        input_seed=None,
        # num_players (int): The number of players.
        num_players=-1, # -1 means there are only one agent player.
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        """
        Return the default configuration of the environment.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = {}):
        """
        Initialize the RL4IC environment.
        """
        self._cfg = deep_merge_dicts(self.config, cfg)
        self._num_sub_agents = cfg['num_sub_agents']
        self._num_layers = cfg['num_layers']
        self._max_input = cfg['max_input']

        # if need to set the seed to debug
        seed = cfg.get('input_seed', None)
        self.seed(seed)
        

        # # multi-player version
        # self.agents = [f'agent_{i}' for i in range(self._num_agents)]
        # self.possible_agents = self.agents[:]
        
        # single-player version
        self.agents = [f'agent']
        self.possible_agents = self.agents[:]

        self._has_reset = False

        # NOTE: this is single-player version, observation is a 3D array
        # NOTE: observe top should be consider twice. what the value "0" means?
        boundary_ndarray = np.array([self._max_input, self._max_input, self._num_layers, 1]).reshape((1,4,1))
        high_ndarray = np.broadcast_to(boundary_ndarray, (4,4,4))
        self._observation_space = spaces.Box(low=0, # [input_data, containers_data, top_layer_bias, agent_mask]
                                            high=high_ndarray.flatten(), 
                                            # shape=(self._num_sub_agents, 4, self._num_sub_agents), 
                                            shape=(self._num_sub_agents * 4 * self._num_sub_agents, ), 
                                            dtype=np.float32)

        # 'action_mask': spaces.Box(low=0, high=1, shape=(np.power(self._num_sub_agents + 1, self._num_sub_agents), ), dtype=bool),

        # merge 2d action space to 1 
        self._action_space = self._convert_to_dict(
            [spaces.Discrete(np.power(self._num_sub_agents + 1, self._num_sub_agents))]
        )

        self._reward_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._agent_selector = AgentSelector(self.agents)

        self.observation_space = self._observation_space
        self.action_space = self._action_space
        self.reward_space = self._reward_space

        #init containers
        self._containers = Container(self._num_sub_agents, self._num_layers, self._max_input)
        self._timestep = 0

        self._game_round = 0

        
        # print(f"---Initializing RL4IC Env---")

        # # NOTE: this is multi-player version, observation is a 2D array
        # self._observation_space = self._convert_to_dict(
        #     [
        #         spaces.Dict(
        #             {
        #                 'observation': spaces.Box(low=0, # [input_data, containers_data, top_layer_bias, agent_mask]
        #                                           high=np.array([self._max_input, self._max_input, self._num_layers, 1]), 
        #                                           shape=(4, self._num_agents), 
        #                                           dtype=np.int8),
        #                 'action_mask': spaces.Box(low=0, high=1, shape=(self._num_agents + 1, ), dtype=np.int8),
        #                 'to_play': self._num_agents
        #             }
        #         ) for _ in range(self._num_agents)
        #     ]
        # )

        # self._action_space = self._convert_to_dict(
        #     [spaces.Discrete(self._num_agents + 1) for _ in range(self.num_agents)]
        # )


    
    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def observe(self):
        # observe the current state of the environment and get the action mask
        # get obs, single player version
        input_data = self._containers.get_input()
        fifo_data, fifo_heights = self._containers.observe_fifo()
        agent_available =  self._containers.height_check()
        agent_mask = [[0] * self._num_sub_agents for _ in range(self._num_sub_agents)]
        for i in range(self._num_sub_agents):
            if agent_available[i]:
                agent_mask[i][i] = 1

        obs = []
        for i in range(self._num_sub_agents):
            obs.append(np.array([input_data, fifo_data, fifo_heights, agent_mask[i]]))
        obs = np.array(obs).flatten()

        return obs.astype(np.float32)
        
    def reset(self):
        self._has_reset = True
        
        try:

            # initialize agents selection, it looks useless
            self.agents = self.possible_agents[:]
            self._agent_selector.reinit(self.agents)
            self.agent_selection = self._agent_selector.reset()

            # reset rewards for agents
            self._cumulative_rewards = self._convert_to_dict(np.array([0.0]))
            self.rewards = self._convert_to_dict(np.array([0.0]))
            self.dones = self._convert_to_dict([False])
            self.infos = self._convert_to_dict([{}])
            self.infos[self.agents[0]]['eval_episode_return'] = 0

            for agent, reward in self.rewards.items():
                self._cumulative_rewards[agent] += reward

            # the fisrt time Observation
            obs = self.observe()
            self._action_mask = self._get_action_mask()
            self._timestep = 0

            print(f"Reset. Start round-{self._game_round}.")

            return {'observation': obs, 'action_mask': self._action_mask, 'to_play': -1, 'timestep': self._timestep}
        
        except Exception as e:
            import traceback
            print(f"[ENV ERROR] Reset failed: {e}\n{traceback.format_exc()}")

    def step(self, action):
        # assume action is filtered and legal
        game_over_flag = self._containers.pop_push(action)
        # This is much different with go_envs because of the single-player setting

        # check if the game is over
        if game_over_flag:
            # print(f"We are in game over stage!")
            self.dones = self._convert_to_dict([True])
            self.rewards = self._convert_to_dict([self._encode_rewards()])
            self.render(self.rewards[self.agents[0]])
            self._new_game()
        else:
            self.dones = self._convert_to_dict([False])
            self.rewards = self._convert_to_dict([0.0])

        # self._accumulate_rewards()
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward
            self.infos[agent]['eval_episode_return'] += self._cumulative_rewards[agent]

        self._timestep += 1

        # print(f"Runing round-{self._game_round}-timestep-{self._timestep}.")
        
        obs = self.observe()
        observation = {'observation': obs, 'action_mask': self._action_mask, 'to_play': -1, 'timestep': self._timestep}

        # self.infos is useless
        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    def _get_action_mask(self):
        # using fixed action mask to satisfy the hardware constraint
        return self._containers.get_static_action_mask()

    def _new_game(self):
        # set a new game
        self._game_round += 1
        self._cumulative_rewards[self.agents[0]] = 0

        print(f"Start new round-{self._game_round}.")

        return self._containers.set_new_game()

    def _encode_rewards(self):
        # evaluate the current game
        # print("Next stage is evaluation!")
        return self._containers.evaluate()
    

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the random seed for the environment.

        Args:
            seed (int): The seed value.
            dynamic_seed (bool): Whether to use dynamic seed generation.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self):
        pass

    def __repr__(self):
        return f"RL4ICEnv(num_sub_agents={self._num_sub_agents}, num_layers={self._num_layers}, max_input={self._max_input})"

    def render(self, reward, mode='human'):
        pass
        agent = self.agents[0]
        accumulate_reward = self._cumulative_rewards[agent]
        game_round = self._game_round
        input_num, pop_num = self._containers.get_render_msg()
        print(f"------------Game Round: {game_round}------------\nInput Amount: {input_num}\nPop Amount: {pop_num}\nReward: {reward}\nAccumulate Reward: {accumulate_reward}\n\n", end="")
        print("")