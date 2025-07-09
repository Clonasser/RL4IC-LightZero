'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-07-09
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

import pygame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.utils.default_helper import deep_merge_dicts
from easydict import EasyDict
from gymnasium import spaces
from pettingzoo.utils.agent_selector import AgentSelector
from zoo.board_games.rl4ic.envs.rl4ic_containers_torch import ContainerTorch, BatchContainerTorch


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

        # Set seed for reproducibility
        seed = cfg.get('input_seed', None)
        self.seed(seed)
        
        # Single-player version
        self.agents = [f'agent']
        self.possible_agents = self.agents[:]
        self._has_reset = False

        # Initialize observation space
        self._setup_observation_space()
        
        # Initialize action space
        self._setup_action_space()
        
        # Initialize reward space
        self._reward_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._agent_selector = AgentSelector(self.agents)
        # self.observation_space = self._observation_space
        # self.action_space = self._action_space
        # self.reward_space = self._reward_space

        # Initialize containers
        if self._batch_mode:
            self._containers = BatchContainerTorch(
                batch_size=self._batch_size,
                num_agents=self._num_sub_agents,
                num_layers=self._num_layers,
                max_input=self._max_input,
                device=self._device
            )
        else:
            self._containers = ContainerTorch(
                num_agents=self._num_sub_agents,
                num_layers=self._num_layers,
                max_input=self._max_input,
                device=self._device
            )
        
        self._timestep = 0
        self._game_round = 0

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space
    
    @property
    def action_space(self) -> spaces.Space:
        return self._action_space
    
    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    # def _setup_observation_space(self):
    #     """Setup observation space with PyTorch tensor support"""
    #     # Calculate observation shape - ensure it's (4, 4, 4)
    #     obs_shape = (4, 4, 4)
        
    #     # For batch mode, add batch dimension
    #     if self._batch_mode:
    #         obs_shape = (self._batch_size,) + obs_shape
        
    #     # Define high values for each component
    #     boundary_values = np.array([self._max_input, self._max_input, self._num_layers, 1])
    #     high_values = np.broadcast_to(boundary_values, (4, 4, 4))
        
    #     self._observation_space = spaces.Box(
    #         low=0,
    #         high=high_values,
    #         shape=obs_shape,
    #         dtype=np.float32
    #     )

    def _setup_observation_space(self):
        """Setup observation space with PyTorch tensor support"""
        # Calculate observation shape - ensure it's (4, 4, 4)
        obs_shape = (3 * 4, )
        
        # For batch mode, add batch dimension
        if self._batch_mode:
            obs_shape = (self._batch_size,) + obs_shape
        
        # Define high values for each component
        boundary_values = np.array([self._max_input, self._max_input, self._num_layers]).reshape(3, 1)  
        high_values = np.broadcast_to(boundary_values, (3, 4)).flatten()
        
        self._observation_space = spaces.Box(
            low=0,
            high=high_values,
            shape=obs_shape,
            dtype=np.float32
        )


    def _setup_action_space(self):
        """Setup action space"""
        action_size = np.power(self._num_sub_agents + 1, self._num_sub_agents)
        
        if self._batch_mode:
            self._action_space = self._convert_to_dict(
                [spaces.MultiDiscrete([action_size] * self._batch_size)]
            )
        else:
            self._action_space = self._convert_to_dict(
                [spaces.Discrete(action_size)]
            )

    def _convert_to_dict(self, list_of_list):
        """Convert list to dictionary for agent actions"""
        return dict(zip(self.possible_agents, list_of_list))

    def observe(self):
        """Observe the current state of the environment"""
        # if self._batch_mode:
        #     return self._observe_batch()
        # else:
        return self._observe_single()

    def _observe_single(self):
        """Observe single environment"""
        input_data = self._containers.get_input()
        fifo_data, fifo_heights = self._containers.observe_fifo()
        agent_available = self._containers.height_check()
        
        # Create agent mask
        agent_mask = torch.zeros(self._num_sub_agents, self._num_sub_agents, device=self._device)
        for i in range(self._num_sub_agents):
            if agent_available[i]:
                agent_mask[i, i] = 1

        # Build observation tensor - ensure consistent shape (4, 4, 4)
        obs_list = []
        # for i in range(self._num_sub_agents):
        #     obs_tensor = torch.stack([
        #         input_data.float(),
        #         fifo_data.float(),
        #         fifo_heights.float(),
        #         agent_mask[i].float()
        #     ])
        #     obs_list.append(obs_tensor)
        
        # obs = torch.stack(obs_list)

        
        obs_tensor = torch.stack([
            input_data.float(),
            fifo_data.float(),
            fifo_heights.float(),
        ])
        
        obs = obs_tensor
        
        # Ensure the observation has the correct shape (4, 4, 4)
        # if obs.shape != (4, 4, 4):
        #     # Reshape if necessary
        #     obs = obs.view(4, 4, 4)
        
        # Debug: print observation shape
        # if hasattr(self, '_debug_count') and self._debug_count < 5:
        #     print(f"DEBUG: Observation shape: {obs.shape}, type: {type(obs)}")
        #     self._debug_count = getattr(self, '_debug_count', 0) + 1
        
        return obs.cpu().numpy().astype(np.float32).flatten()

    def _observe_batch(self):
        """Observe batch of environments"""
        input_data = self._containers.get_input()
        fifo_data, fifo_heights = self._containers.observe_fifo()
        agent_available = self._containers.height_check()
        
        # Create agent mask for all environments
        agent_mask = torch.zeros(self._batch_size, self._num_sub_agents, self._num_sub_agents, device=self._device)
        for b in range(self._batch_size):
            for i in range(self._num_sub_agents):
                if agent_available[b, i]:
                    agent_mask[b, i, i] = 1

        # Build observation tensor for all environments
        obs_batch = []
        for b in range(self._batch_size):
            obs_env = []
            for i in range(self._num_sub_agents):
                obs_tensor = torch.stack([
                    input_data[b].float(),
                    fifo_data[b].float(),
                    fifo_heights[b].float(),
                    agent_mask[b, i].float()
                ])
                obs_env.append(obs_tensor)
            
            # Stack and ensure correct shape (4, 4, 4)
            obs_env_tensor = torch.stack(obs_env)
            if obs_env_tensor.shape != (4, 4, 4):
                obs_env_tensor = obs_env_tensor.view(4, 4, 4)
            obs_batch.append(obs_env_tensor)
        
        obs = torch.stack(obs_batch)
        return obs.cpu().numpy().astype(np.float32)

    def reset(self):
        """Reset the environment"""
        self._has_reset = True
        
        try:
            # Initialize agents selection
            self.agents = self.possible_agents[:]
            self._agent_selector.reinit(self.agents)
            self.agent_selection = self._agent_selector.reset()

            # Reset rewards for agents
            self._cumulative_rewards = self._convert_to_dict(np.array([0.0]))
            self.rewards = self._convert_to_dict(np.array([0.0]))
            self.dones = self._convert_to_dict([False])
            self.infos = self._convert_to_dict([{}])
            self.infos[self.agents[0]]['eval_episode_return'] = 0

            for agent, reward in self.rewards.items():
                self._cumulative_rewards[agent] += reward

            # Reset containers
            if self._batch_mode:
                self._containers.reset()
            else:
                self._containers.reset()

            # Get first observation
            obs = self.observe()
            self._action_mask = self._get_action_mask()
            self._timestep = 0

            # print(f"Reset.")

            return {
                'observation': obs, 
                'action_mask': self._action_mask, 
                'to_play': -1, 
                'timestep': self._timestep
            }
        
        except Exception as e:
            import traceback
            print(f"[ENV ERROR] Reset failed: {e}\n{traceback.format_exc()}")

    def step(self, action):
        """Execute one step in the environment"""
        if self._batch_mode:
            return self._step_batch(action)
        else:
            return self._step_single(action)

    def _step_single(self, action):
        """Execute step for single environment"""
        # Execute action
        game_over_flag = self._containers.pop_push(action)

        # Check if game is over
        if game_over_flag:
            self.dones = self._convert_to_dict([True])
            self.rewards = self._convert_to_dict([self._encode_rewards()])
            self.render(self.rewards[self.agents[0]])
            self._new_game()
        else:
            self.dones = self._convert_to_dict([False])
            self.rewards = self._convert_to_dict([0.0])

        # Update cumulative rewards
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward
            self.infos[agent]['eval_episode_return'] += self._cumulative_rewards[agent]

        self._timestep += 1
        
        if self._cumulative_rewards[self.agents[0]] is None:
            self._cumulative_rewards[self.agents[0]] = 0.0

        # Get new observation
        obs = self.observe()
        observation = {
            'observation': obs, 
            'action_mask': self._action_mask, 
            'to_play': -1, 
            'timestep': self._timestep
        }

        return BaseEnvTimestep(observation, self._cumulative_rewards[self.agents[0]], 
                              self.dones[self.agents[0]], self.infos[self.agents[0]])

    def _step_batch(self, actions):
        """Execute step for batch of environments"""
        # Handle batch actions
        if isinstance(actions, (list, np.ndarray)):
            actions = torch.tensor(actions, device=self._device)
        
        # Execute actions for all environments
        game_over_flags = []
        for i, action in enumerate(actions):
            game_over_flag = self._containers.pop_push(action.item())
            game_over_flags.append(game_over_flag)

        # Update environment states
        game_over_tensor = torch.tensor(game_over_flags, device=self._device)
        
        # Calculate rewards
        rewards = torch.zeros(self._batch_size, device=self._device)
        for i, game_over in enumerate(game_over_flags):
            if game_over:
                rewards[i] = self._encode_rewards_batch(i)
                self._new_game_batch(i)
            else:
                rewards[i] = 0.0

        # Update timestep
        self._timestep += 1
        
        # Get new observations
        obs = self.observe()
        observation = {
            'observation': obs, 
            'action_mask': self._action_mask, 
            'to_play': -1, 
            'timestep': self._timestep
        }

        assert obs.shape == (4, 4, 4), f"Observation shape should be (4,4,4) but got {tuple(obs.shape)}, the obs is {obs}."
        # Return batch timestep
        return BaseEnvTimestep(observation, rewards.cpu().numpy(), 
                              game_over_tensor.cpu().numpy(), {})

    def _get_action_mask(self):
        """Get action mask"""
        if self._batch_mode:
            # For batch mode, return mask for all environments
            mask = self._containers.get_static_action_mask()
            return mask.cpu().numpy()
        else:
            # For single environment
            mask = self._containers.get_static_action_mask()
            return mask.cpu().numpy()

    def _new_game(self):
        """Start a new game"""
        self._game_round += 1
        self._cumulative_rewards[self.agents[0]] = 0
        # print(f"Start new round-{self._game_round}.")
        
        if self._batch_mode:
            # Reset all environments in batch
            self._containers.reset()
        else:
            return self._containers.set_new_game()

    def _new_game_batch(self, env_id: int):
        """Start new game for specific environment in batch"""
        # Reset specific environment
        env_ids = torch.tensor([env_id], device=self._device)
        self._containers.reset(env_ids)

    def _encode_rewards(self):
        """Evaluate the current game and return reward"""
        return self._containers.evaluate()

    def _encode_rewards_batch(self, env_id: int):
        """Evaluate specific environment in batch"""
        # This would need to be implemented in BatchContainerTorch
        # For now, return a placeholder
        return 0.0

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

    def render(self, reward, mode='human'):
        """Render the environment state"""
        pass
        agent = self.agents[0]
        # accumulate_reward = self._cumulative_rewards[agent]
        game_round = self._game_round
        input_num, pop_num = self._containers.get_render_msg()
        print(f"------------Game Round: {game_round}------------\nInput Amount: {input_num}\nPop Amount: {pop_num}\nReward: {reward}\n\n", end="")
        print("")


# Batch environment wrapper for parallel training
class BatchRL4ICEnv:
    """
    Batch wrapper for multiple RL4IC environments
    """
    
    def __init__(self, num_envs: int, config: dict):
        """
        Initialize batch environment
        
        Args:
            num_envs: Number of parallel environments
            config: Environment configuration
        """
        self.num_envs = num_envs
        self.config = config.copy()
        self.config['batch_mode'] = True
        self.config['batch_size'] = num_envs
        
        # Create batch environment
        self.env = RL4ICEnvTorch(self.config)
        
    def reset(self):
        """Reset all environments"""
        return self.env.reset()
    
    def step(self, actions):
        """Step all environments"""
        return self.env.step(actions)
    
    def observe(self):
        """Observe all environments"""
        return self.env.observe()
    
    def seed(self, seed: int):
        """Set seed for all environments"""
        self.env.seed(seed)
    
    def close(self):
        """Close all environments"""
        self.env.close()
    
    def to(self, device: torch.device):
        """Move to device"""
        self.env.to(device)
        return self


if __name__ == "__main__":
    # Test single environment
    config = {
        'num_sub_agents': 4,
        'num_layers': 8,
        'max_input': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    env = RL4ICEnvTorch(config)
    print(f"Environment initialized: {env}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs['observation'].shape}")
    

    # obs = env.observe()
    # print(f"Observation shape: {obs.shape}")
    
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