'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-06-30
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
        num_agents=4,
        # num_layers (int): The number of layers in each container.
        num_layers=16,
        # max_input (int): The max number provided by the input generator.
        max_input=64,
        # input_seed (int): The seed of the input generator.
        input_seed=0,
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
        cfg.input_seed = np.random.randint(0, 20250630)
        return cfg

    def __init__(self, cfg: dict = {}):
        """
        Initialize the RL4IC environment.
        """
        self._cfg = deep_merge_dicts(self.config, cfg)
        self._num_agents = cfg['num_agents']
        self._num_layers = cfg['num_layers']
        self._max_input = cfg['max_input']
        self._input_seed = cfg.get('input_seed', np.random.randint(0, 20250630))

        # # multi-player version
        # self.agents = [f'agent_{i}' for i in range(self._num_agents)]
        # self.possible_agents = self.agents[:]
        
        # single-player version
        self.agents = [f'agent']
        self.possible_agents = self.agents[:]

        self._has_reset = False

        # NOTE: this is single-player version, observation is a 3D array
        self._observation_space = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        'observation': spaces.Box(low=0, # [input_data, containers_data, top_layer_bias, agent_mask]
                                                  high=np.array([self._max_input, self._max_input, self._num_layers, 1]).reshape((1,4,1)), 
                                                  shape=(self._num_agents, 4, self._num_agents), 
                                                  dtype=np.int8),
                        'action_mask': spaces.Box(low=0, high=1, shape=(self._num_agents + 1, ), dtype=np.int8),
                        'to_play': self._num_agents
                    }
                ) 
            ]
        )

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

        self._action_space = self._convert_to_dict(
            [spaces.Discrete(self._num_agents + 1) for _ in range(self.num_agents)]
        )
        self._agent_selector = AgentSelector(self.agents)
        

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def observe(self):
        # observe the current state of the environment and get the action mask
        # get obs, single player version
        input_data = self._containers.get_input()
        fifo_data, fifo_heights = self._containers.observe_fifo()
        agent_available =  self._containers.height_check()
        agent_mask = [[0] * self._num_agents for _ in range(self._num_agents)]
        for i in range(self._num_agents):
            if agent_available[i]:
                agent_mask[i][i] = 1

        obs = []
        for i in range(self._num_agents):
            obs.append(np.array([input_data, fifo_data, fifo_heights, agent_mask[i]]))
        obs = np.array(obs)

        # get action mask
        action_mask = [1] # select "empty" is always legal
        for i in input_data:
            if i != 0:
                action_mask.append(1)
            else:
                action_mask.append(0)
        action_mask = np.array(action_mask)

        return {'observation': obs, 'action_mask': action_mask}

    def reset(self):
        self._has_reset = True
        
        #init containers
        self._containers = Container(self._num_agents, self._num_layers, self._max_input, self._input_seed)

        # initialize agents selection, it looks useless
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # reset rewards for agents
        self._cumulative_rewards = self._convert_to_dict(np.array([0.0 for _ in range(self._num_agents)]))
        self.rewards = self._convert_to_dict(np.array([0.0 for _ in range(self._num_agents)]))
        self.dones = self._convert_to_dict([False for _ in range(self._num_agents)])
        self.infos = self._convert_to_dict([{} for _ in range(self._num_agents)])

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        # the fisrt time Observation
        obs = self.observe()
        return obs

    def step(self, action):
        # assume action is filtered and legal
        self._containers.pop_push(action)
        # This is much different with go_envs because of the single-player setting

        # check if the game is over
        if self._containers.is_game_over():
            self.dones = self._convert_to_dict([True for _ in range(self._num_agents)])
            self.rewards = self._convert_to_dict(self._encode_rewards())
            self._new_round()        

        # self._accumulate_rewards()
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward
        
        observation = self.observe()

        # self.infos is useless
        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])


    def _new_game(self):
        # TODO: reset the containers or start a new game
        pass

    def _encode_rewards(self):
        # TODO: encode the rewards, it should be return a list?
        pass
