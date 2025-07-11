'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-07-11
FilePath: /RL4IC-LightZero/zoo/board_games/rl4ic/envs/rl4ic_env_torch.py
Description: PyTorch optimized RL4IC environment for GPU parallel training
Copyright (c) 2025 by CAS4ET lab, ShanghaiTech University, All Rights Reserved. 
'''
import os
import sys
import copy
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math

# import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.utils.default_helper import deep_merge_dicts
from easydict import EasyDict
from gymnasium import spaces
from pettingzoo.utils.agent_selector import AgentSelector
from zoo.board_games.rl4ic.envs.rl4ic_containers_torch import ContainerTorch

@ENV_REGISTRY.register('RL4IC_TORCH')
class RL4ICEnvTorch(BaseEnv):
    """
    PyTorch optimized RL4IC environment for GPU parallel training
    """

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
        # device (str): PyTorch device for computation
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # batch_mode (bool): Whether to use batch processing for multiple environments
        batch_mode=False,
        # batch_size (int): Batch size for parallel environments
        batch_size=1,
        # allow place empty elements
        allow_place_empty=False,
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
        Initialize the PyTorch optimized RL4IC environment.
        """
        self._cfg = deep_merge_dicts(self.config, cfg)
        self._num_sub_agents = cfg['num_sub_agents']
        self._num_layers = cfg['num_layers']
        self._max_input = cfg['max_input']
        self._device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self._batch_mode = cfg.get('batch_mode', False)
        self._batch_size = cfg.get('batch_size', 1)
        self._env_use_wandb = cfg.get('env_use_wandb', False)
        self._allow_place_empty = cfg.get('allow_place_empty', False)

        # Set seed for reproducibility
        seed = cfg.get('input_seed', None)
        self.seed(seed)
        # Initialize reward space
        self._reward_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Initialize containers
        self._containers = ContainerTorch(
            num_agents=self._num_sub_agents,
            num_layers=self._num_layers,
            max_input=self._max_input,
            device=self._device,
            allow_place_empty=self._allow_place_empty
        )
        self._timestep = 0
        self._game_round = 0

        # Initialize observation space
        self._setup_observation_space()
        
        # Initialize action space
        self._setup_action_space()

        # for muzero, action mask is required
        self._muzero_action_mask = torch.ones(self._action_space_size).bool()

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space
    
    @property
    def action_space(self) -> spaces.Space:
        return self._action_space
    
    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def _setup_observation_space(self):
        """Setup observation space with PyTorch tensor support"""
        # Calculate observation shape
        # one hot encoding * sub_agent_num * 2 + sub_agent_num
        # input date (onehot) + fifo date (onehot) + fifo height bias (int)
        obs_shape = ((self._max_input * self._num_sub_agents) * 2 + self._num_sub_agents, )
        
        # Define high values for each component
        boundary_values = [1] * (self._max_input * self._num_sub_agents * 2) + [self._max_input] * self._num_sub_agents
        # boundary_values = np.array([self._max_input, self._max_input, self._num_layers]).reshape(3, 1)  
        high_values = np.array(boundary_values, dtype=np.int32)
        
        self._observation_space = spaces.Box(
            low=0,
            high=high_values,
            shape=obs_shape,
            dtype=np.int32
        )

    def _setup_action_space(self):
        """Setup action space"""
        self._action_space_size = self._containers.get_optimized_action_space_size()
        self._action_space = spaces.Discrete(self._action_space_size)

    def observe(self):
        """Observe the current state of the environment"""
        return self._observe_single()

    def get_onehot(self, input_data: int) -> torch.Tensor:
        "change input date to one hot encoding"
        onehot_data = torch.zeros(self._max_input, device=self._device)
        onehot_data[input_data] = 1
        return onehot_data

    def _observe_single(self):
        """Observe single environment"""
        input_data = self._containers.get_input()
        fifo_data, fifo_heights_bias = self._containers.observe_fifo()
        
        input_onehot = torch.zeros((self._num_sub_agents, self._max_input), device=self._device)
        fifo_onehot = torch.zeros((self._num_sub_agents, self._max_input), device=self._device)
        # print(f"input_data: {input_data}")
        # print(f"fifo_data: {fifo_data}")
        # print(f"fifo_heights_bias: {fifo_heights_bias}")
        # change data to one hot encoding
        for i in range(self._num_sub_agents):
            input_onehot[i] = self.get_onehot(input_data[i])
            fifo_onehot[i] = self.get_onehot(fifo_data[i])
        
        obs_tensor = torch.cat([
            input_onehot.flatten(),
            fifo_onehot.flatten(),
            fifo_heights_bias.flatten(),
        ])
        # print("input_onehot: ", input_onehot)
        
        obs = obs_tensor
        return obs.cpu().numpy().flatten()


    def reset(self):
        """Reset the environment"""
        self._has_reset = True

        # Reset rewards for agents
        self._cumulative_rewards = 0.0
        self.rewards = 0.0
        self.dones = False
        self.infos = {}
        self.infos['eval_episode_return'] = 0

        self._containers.reset()

        # Get first observation
        obs = self.observe()
        # self._action_mask = self._get_action_mask()
        self._timestep = 0

        return {
            'observation': obs, 
            'action_mask': self._muzero_action_mask,
            'to_play': -1, 
            'timestep': self._timestep
        }
    

    def step(self, action):
        """Execute step for single environment"""      
        # Execute action
        game_over_flag = self._containers.pop_push(action)
        # Check if game is over
        if game_over_flag:
            self.dones = True
            self.rewards = self._encode_rewards()
            self.render(self.rewards)
            self._new_game()
        else:
            self.dones = False
            self.rewards = 0.0

        # Update cumulative rewards
        self._cumulative_rewards += self.rewards
        self.infos['eval_episode_return'] += self._cumulative_rewards

        self._timestep += 1
        
        if self._cumulative_rewards is None:
            self._cumulative_rewards = 0.0

        # Get new observation
        obs = self.observe()
        observation = {
            'observation': obs, 
            'action_mask': self._muzero_action_mask,
            'to_play': -1, 
            'timestep': self._timestep
        }

        return BaseEnvTimestep(observation, self._cumulative_rewards, 
                              self.dones, self.infos)

    # def _get_action_mask(self):
    #     """Get action mask"""
    #     # For single environment
    #     mask = self._containers.get_static_action_mask()
    #     return mask.cpu().numpy()

    def _new_game(self):
        """Start a new game"""
        self._game_round += 1
        self._cumulative_rewards = 0
        # print(f"Start new round-{self._game_round}.")

        return self._containers.set_new_game()

    def _encode_rewards(self):
        """Evaluate the current game and return reward"""
        return self._containers.evaluate()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """Set the random seed for the environment"""
        if seed is None:
            seed = np.random.randint(0, 2023231050)
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
        np.random.seed(self._seed)

    def close(self):
        """Close the environment"""
        pass

    def to(self, device: torch.device):
        """Move environment to specified device"""
        self._device = device
        self._containers = self._containers.to(device)
        return self

    def __repr__(self):
        return f"RL4ICEnvTorch(num_sub_agents={self._num_sub_agents}, num_layers={self._num_layers}, max_input={self._max_input}, device={self._device}, batch_mode={self._batch_mode})"

    def render(self, reward):
        """Render the environment"""
         # accumulate_reward = self._cumulative_rewards[agent]
        game_round = self._game_round
        pop_num, input_num, = self._containers.get_render_msg()
        print(f"------------Game Round: {game_round}------------\nInput Amount: {input_num}\nPop Amount: {pop_num}\nReward: {reward}\n\n", end="")
        print("")

if __name__ == "__main__":
    # Test single environment
    config = {
        'num_sub_agents': 4,
        'num_layers': 16,
        'max_input': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    env = RL4ICEnvTorch(config)
    print(f"Environment initialized: {env}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs['observation'].shape}")
    print(f"Observation shape: {env.observation_space}")
    print(f"Action shape: {env.action_space}")
        
    # Test step
    action = 120  # Example action
    timestep = env.step(action)
    print(f"Step result: reward={timestep.reward}, done={timestep.done}, obs={timestep.obs['observation'].shape}")
    
    # # Test batch environment
    # batch_env = BatchRL4ICEnv(num_envs=4, config=config)
    # print(f"Batch environment initialized: {batch_env}")

    # # Test batch reset
    # batch_obs = batch_env.reset()
    # print(f"Batch observation shape: {batch_obs['observation'].shape}")
    
    # # Test batch step
    # batch_actions = torch.randint(0, 625, (4,))  # Random actions for 4 environments
    # batch_timestep = batch_env.step(batch_actions)
    # print(f"Batch step result: rewards={batch_timestep.reward}, dones={batch_timestep.done}")