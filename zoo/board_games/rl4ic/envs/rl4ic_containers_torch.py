'''
Author: wangyt32023@shanghaitech.edu.cn
Date: 2025-06-30
LastEditors: wangyt32023@shanghaitech.edu.cn
LastEditTime: 2025-07-10
FilePath: /RL4IC-LightZero/zoo/board_games/rl4ic/envs/rl4ic_containers_torch.py
Description: PyTorch optimized version for GPU parallel computation
Copyright (c) 2025 by CAS4ET lab, ShanghaiTech University, All Rights Reserved. 
'''
import torch
import copy
from typing import Tuple, List, Optional
import math
import wandb

# Utility functions
def convert_4d_index_to_1d(index: Tuple[int, int, int, int], shape: Tuple[int, int, int, int] = (5, 5, 5, 5)) -> int:
    """Convert 4D index to 1D index using PyTorch operations"""
    i, j, k, l = index
    _, d2, d3, d4 = shape
    return i * (d2 * d3 * d4) + j * (d3 * d4) + k * d4 + l

def convert_1d_index_to_4d(index: int, shape: Tuple[int, int, int, int] = (5, 5, 5, 5)) -> Tuple[int, int, int, int]:
    """Convert 1D index to 4D index using PyTorch operations"""
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

def input_generator_torch(max_input: int, input_num: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Generate sorted random input using PyTorch"""
    # Generate random numbers from 1 to max_input
    random_nums = torch.randint(1, max_input, (input_num,), device=device)
    # Sort the numbers
    return torch.sort(random_nums)[0]

# Main class
class ContainerTorch:
    """
    PyTorch optimized Container class for GPU parallel computation
    """

    def __init__(self, num_agents: int, num_layers: int, max_input: int, 
                 device: torch.device = torch.device('cpu'), render: bool = False, use_wandb: bool = False):
        """
        Initialize the container with PyTorch tensors
        
        Args:
            num_agents: Number of sub-agents
            num_layers: Number of layers in each container
            max_input: Maximum input value
            device: PyTorch device (cpu/cuda)
            render: Whether to enable rendering
        """
        self._num_sub_agents = num_agents
        self._num_layers = num_layers
        self._max_input = max_input
        self._device = device
        self._render = render
        self._use_wandb = use_wandb
        # Initialize tensors
        self._input = torch.empty(0, device=device, dtype=torch.long)
        self._fifo = None  # Will be initialized in reset_fifo
        self._pop_counter = 0
        self._input_num = 0
        
        # Pre-compute action mask for efficiency
        self._action_mask = self._precompute_action_mask()

        # Pre-compute optimized action space
        self._opt_action_space_size = self.get_optimized_action_space_size()
        
        # Reset the container
        self.reset()

    def reset(self):
        """Reset data and reinitialize input"""
        # Generate new input size
        base_size = self._num_layers * self._num_sub_agents
        random_offset = torch.randint(-2 * self._num_sub_agents, self._max_input, (1,), device=self._device).item()
        self._input_num = base_size + random_offset
        
        # Generate new input
        self._input = input_generator_torch(self._max_input, self._input_num, self._device)
        
        # Reset FIFO and counters
        self.reset_fifo()
        self._pop_counter = 0

    def reset_fifo(self):
        """Reset FIFO queues"""
        # Use tensor-based FIFO for better GPU performance
        self._fifo = torch.full((self._num_sub_agents, self._num_layers), 
                               -1, device=self._device, dtype=torch.long)
        self._fifo_lengths = torch.zeros(self._num_sub_agents, device=self._device, dtype=torch.long)

    def get_input(self) -> torch.Tensor:
        """Get the current input buffer for agents"""
        if self._input.size(0) < self._num_sub_agents:
            # Pad with zeros if input is smaller than num_sub_agents
            padding_size = self._num_sub_agents - self._input.size(0)
            padding = torch.zeros(padding_size, device=self._device, dtype=torch.long)
            return torch.cat([self._input, padding])
        else:
            return self._input[:self._num_sub_agents]

    def observe_fifo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the top layer data and height bias of each channel"""
        # Get top layer data (last non-negative value in each FIFO)
        fifo_data = torch.full((self._num_sub_agents, ), -1, device=self._device, dtype=torch.long)
        for i in range(self._num_sub_agents):
            if self._fifo_lengths[i] > 0:
                fifo_data[i] = self._fifo[i, self._fifo_lengths[i] - 1]
        
        # Convert -1 to 0 for consistency
        fifo_data = torch.where(fifo_data == -1, torch.tensor(0, device=self._device), fifo_data)
        
        # Calculate height bias
        fifo_heights = self._fifo_lengths.clone()
        base_height = torch.min(fifo_heights)
        fifo_heights_bias = fifo_heights - base_height
        
        return fifo_data, fifo_heights_bias

    def height_check(self) -> torch.Tensor:
        """Return whether each channel is available (not full)"""
        return self._fifo_lengths < self._num_layers

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        # which is not same with the original version
        return (self._input.size(0) == 0) or (not torch.all(self.height_check()).item())

    def pop_push(self, action: int) -> bool:
        """
        Pop and push the selected input to the FIFO
        
        Args:
            action: 1D action index
            
        Returns:
            bool: whether the game is over
        """
        
        
        action = self._convert_reduced_to_original_index(action) # conver optimized index to original
        action_tuple = convert_1d_index_to_4d(action)
        print("Pop_push_action: ", action_tuple)
        action_list = list(action_tuple)
        pop_indices = []
        buffer = self.get_input()
        
        # Process each sub-agent's action
        for i, sel_i in enumerate(action_list):
            if sel_i != 0 and sel_i <= buffer.size(0):
                # Push to FIFO
                if self._fifo_lengths[i] < self._num_layers:
                    self._fifo[i, self._fifo_lengths[i]] = buffer[sel_i - 1]
                    self._fifo_lengths[i] += 1
                    
                    # Mark for removal from input
                    if buffer[sel_i - 1] != 0:
                        pop_indices.append(sel_i - 1)

        # Remove used items from input
        if pop_indices:
            # Sort indices in descending order to avoid index shifting
            pop_indices.sort(reverse=True)
            for idx in pop_indices:
                if idx < self._input.size(0):
                    self._input = torch.cat([self._input[:idx], self._input[idx+1:]])
            
            self._pop_counter += len(pop_indices)

        return self.is_game_over()

    def evaluate(self) -> float:
        """
        Evaluate the current game state and return reward
        Optimized version using PyTorch operations
        """
        # Create a copy of FIFO for evaluation
        fifo_copy = self._fifo.clone()
        fifo_lengths_copy = self._fifo_lengths.clone()
        
        time_counter = 0
        
        # Continue until all FIFOs are empty
        while torch.any(fifo_lengths_copy > 0):
            # use suipervisor to adviod an element wait too long.\
            pop_supervisor = torch.zeros(self._num_sub_agents, device=self._device, dtype=torch.int)
            supervisor_mapping = {0: 1, 1: 0, 2: 3, 3: 2}

            # Prepare buffer (bottom of each FIFO)
            buffer = torch.full((self._num_sub_agents, ), -1, device=self._device, dtype=torch.long)
            for i in range(self._num_sub_agents):
                if fifo_lengths_copy[i] > 0:
                    buffer[i] = fifo_copy[i, 0]

            # Find duplicates efficiently using PyTorch operations
            pop_list = self._compute_pop_list(buffer)
            
            # pop the element, gurantee the lonely element is poped
            # print(f"Original pop list: {pop_list.tolist()}")
            if torch.sum(pop_supervisor) == 1:
                for i in range(self._num_sub_agents):
                    if (pop_supervisor[i] == 1) and (pop_list[i] == False):
                        pop_list[i] = True # pop it
                        pop_list[supervisor_mapping[i]] = False # adviod pop another element
                    pop_supervisor[i] = 0 # reset the supervisor
            # record the lonely element
            if torch.sum(pop_list) == 3:
                for i in range(self._num_sub_agents):
                    if (pop_list[i] == False) and fifo_lengths_copy[i] > 0: # if the channel is empty, ignore it
                        pop_supervisor[i] = 1
                    else:
                        pop_supervisor[i] = 0
                    
            # print(f"Supervised pop list: {pop_list.tolist()}")
            # Remove items from FIFO
            for i in range(self._num_sub_agents):
                if pop_list[i] and fifo_lengths_copy[i] > 0:
                    # Shift remaining elements
                    fifo_copy[i, :-1] = fifo_copy.clone()[i, 1:]
                    fifo_lengths_copy[i] -= 1
            
            time_counter += 1

            if self._render:
                self._render_fifo_torch(fifo_copy, fifo_lengths_copy, time_counter, pop_list)

        # Calculate reward
        ideal_counter = math.ceil((self._pop_counter + self._num_sub_agents - 1) / self._num_sub_agents)
        reward = ideal_counter / time_counter if time_counter > 0 else 0.0
        reward = reward * (self._pop_counter / (self._num_sub_agents  * self._num_layers))
        reward = 9 * reward - 5  # scaling factor
        reward = 1 / (1 + math.exp(-reward))  # sigmoid

        if self._render:
            print("********************************")
            print(f"Pop counter: {self._pop_counter}")
            print(f"Time counter: {time_counter}")
            print(f"Ideal counter: {ideal_counter}")
            print(f"Reward: {reward}")
            print("********************************")

        return reward

    def _compute_pop_list(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Compute which items to pop based on buffer contents
        Optimized version using PyTorch operations
        """
        pop_list = torch.zeros(self._num_sub_agents, device=self._device, dtype=torch.bool)
        
        # Find unique values and their counts
        valid_mask = buffer != -1
        if not torch.any(valid_mask):
            return pop_list # TODO: If pop a empty fifo, what should we do?
            
        valid_values = buffer[valid_mask]
        unique_values, counts = torch.unique(valid_values, return_counts=True)
        
        # Find maximum repeat count
        max_repeat = torch.max(counts).item()
        max_repeat_values = unique_values[counts == max_repeat]
        
        # Apply pop logic based on repeat count
        # NOTE: this logic is only for 4 sub-agents
        if max_repeat in [self._num_sub_agents, self._num_sub_agents - 1]:
            # Pop all matching values
            for val in max_repeat_values:
                pop_list = pop_list | (buffer == val)
        elif max_repeat == 2:
            if len(max_repeat_values) == 1:
                # Pop matching values and one non-matching
                val = max_repeat_values[0]
                pop_list = buffer == val
                # Add one non-matching value
                non_match_buffer = torch.where(buffer != val, buffer, -1)
                selected_idx = self.find_min_two_indices(non_match_buffer)
                if len(selected_idx):
                    pop_list[selected_idx[0]] = True
                # non_matching = (buffer != val) & (buffer != -1)
                # if torch.any(non_matching):
                #     first_non_matching = torch.where(non_matching)[0][0]
                #     pop_list[first_non_matching] = True
            else:
                # Pop all
                pop_list = valid_mask
        else:
            # Pop first two non-negative values
            selected_idx = self.find_min_two_indices(buffer)
            non_negative = (buffer != -1)
            for idx in selected_idx:
                pop_list[idx] = True
            # if torch.sum(non_negative) >= 2:
            #     indices = torch.where(non_negative)[0][:2]
            #     pop_list[indices] = True
            # elif torch.sum(non_negative) == 1:
            #     indices = torch.where(non_negative)[0][:1]
            #     pop_list[indices] = True
                
        return pop_list

    def find_min_two_indices(self, lst):
        non_negative = []
        for index, num in enumerate(lst):
            if num >= 0: 
                non_negative.append((num, index))

        n = len(non_negative)

        if n == 0:
            return []  
        elif n == 1:
            return [non_negative[0][1]]  
        else:
            non_negative_sorted = sorted(non_negative, key=lambda x: x[0])
            first_min_index = non_negative_sorted[0][1]
            second_min_index = non_negative_sorted[1][1]
            return [first_min_index, second_min_index]

    def get_static_action_mask(self) -> torch.Tensor:
        """Get the pre-computed static action mask"""
        return self._action_mask

    def get_optimized_action_space_size(self) -> int:
        """Get the pre-computed optimized action space"""
        return torch.sum(self._action_mask).item()

    def _precompute_action_mask(self) -> torch.Tensor:
        """Pre-compute static action mask for all rounds"""
        action_shape = (self._num_sub_agents + 1, self._num_sub_agents + 1, 
                       self._num_sub_agents + 1, self._num_sub_agents + 1)
        total_actions = (self._num_sub_agents + 1) ** self._num_sub_agents
        
        # Create all possible action combinations
        indices = torch.arange(total_actions, device=self._device)
        action_mask = torch.ones(total_actions, device=self._device, dtype=torch.bool)
        
        # Convert 1D indices to 4D and check for duplicates
        for idx in indices:
            action_tuple = convert_1d_index_to_4d(idx.item(), action_shape)
            if self._has_duplicate_index(action_tuple):
                action_mask[idx] = False
                
        return action_mask

    def _has_duplicate_index(self, action_tuple: Tuple[int, int, int, int]) -> bool:
        """Check for duplicate indices"""
        seen = set()
        for idx in action_tuple:
            if idx != 0 and idx in seen:
                return True
            seen.add(idx)
        return False

    def _convert_reduced_to_original_index(self, reduced_index: int) -> int:
        """
        convert reduced space index to original space index
        
        Args:
            reduced_index: reduced space index
            
        Returns:
            int: original space index
        """
        # get all valid actions
        valid_indices = torch.where(self._action_mask)[0]
        
        if reduced_index >= len(valid_indices):
            raise ValueError(f"Reduced index {reduced_index} is out of bounds (max: {len(valid_indices)-1})")
        
        # return the corresponding original index
        return valid_indices[reduced_index].item()

    
    def convert_original_to_reduced_index(self, original_index: int) -> int:
        """
        convert original space index to reduced space index
        
        Args:
            original_index: original space index
            action_mask: boolean tensor, True for valid actions, False for masked actions
            
        Returns:
            int: reduced space index, -1 for invalid actions
        """
        # if the index is masked, return -1
        if not self._action_mask[original_index]:
            return -1
        
        # calculate the position of the index in the reduced space
        # that is, calculate how many valid actions are before original_index
        valid_count = torch.sum(self._action_mask[:original_index]).item()
        return valid_count

    def set_new_game(self):
        """Reset all to start a new game"""
        self.reset()

    def get_render_msg(self) -> Tuple[int, int]:
        """Get render message: [pop counter, input counter]"""
        return self._pop_counter, self._input_num

    def _render_fifo_torch(self, fifo: torch.Tensor, fifo_lengths: torch.Tensor, time_counter: int, pop_list: torch.Tensor=None):
        """Render FIFO state for debugging"""
        
        print(f"time_counter: {time_counter}")
        # if pop_list is not None:
        #     print(f"This step pop: {pop_list.tolist()}")
        for i in range(self._num_sub_agents):
            if fifo_lengths[i] > 0:
                print(f"FIFO {i}: {fifo[i, :fifo_lengths[i]].tolist()}")
            else:
                print(f"FIFO {i}: []")

        print("--------------------------------")

    def to(self, device: torch.device):
        """Move container to specified device"""
        self._device = device
        self._input = self._input.to(device)
        self._fifo = self._fifo.to(device)
        self._fifo_lengths = self._fifo_lengths.to(device)
        self._action_mask = self._action_mask.to(device)
        return self

    def __repr__(self):
        return f"ContainerTorch(num_sub_agents={self._num_sub_agents}, num_layers={self._num_layers}, max_input={self._max_input}, device={self._device})"


# # Batch version for parallel processing
# class BatchContainerTorch:
#     """
#     Batch version of ContainerTorch for parallel processing of multiple environments
#     """
    
#     def __init__(self, batch_size: int, num_agents: int, num_layers: int, max_input: int,
#                  device: torch.device = torch.device('cpu')):
#         """
#         Initialize batch container
        
#         Args:
#             batch_size: Number of parallel environments
#             num_agents: Number of sub-agents per environment
#             num_layers: Number of layers per environment
#             max_input: Maximum input value
#             device: PyTorch device
#         """
#         self.batch_size = batch_size
#         self.num_agents = num_agents
#         self.num_layers = num_layers
#         self.max_input = max_input
#         self.device = device
        
#         # Initialize batch tensors
#         self._input = torch.empty(batch_size, 0, device=device, dtype=torch.long)
#         self._fifo = torch.full((batch_size, num_agents, num_layers), 
#                                -1, device=device, dtype=torch.long)
#         self._fifo_lengths = torch.zeros(batch_size, num_agents, device=device, dtype=torch.long)
#         self._pop_counters = torch.zeros(batch_size, device=device, dtype=torch.long)
#         self._input_nums = torch.zeros(batch_size, device=device, dtype=torch.long)
        
#         # Pre-compute action mask
#         self._action_mask = self._precompute_action_mask()
        
#         self.reset()

#     def _precompute_action_mask(self) -> torch.Tensor:
#         """Pre-compute static action mask"""
#         action_shape = (self.num_agents + 1, self.num_agents + 1, 
#                        self.num_agents + 1, self.num_agents + 1)
#         total_actions = (self.num_agents + 1) ** self.num_agents
        
#         indices = torch.arange(total_actions, device=self.device)
#         action_mask = torch.ones(total_actions, device=self.device, dtype=torch.bool)
        
#         for idx in indices:
#             action_tuple = convert_1d_index_to_4d(idx.item(), action_shape)
#             if self._has_duplicate_index(action_tuple):
#                 action_mask[idx] = False
                
#         return action_mask



#     def reset(self, env_ids: Optional[torch.Tensor] = None):
#         """Reset specified environments or all if env_ids is None"""
#         if env_ids is None:
#             env_ids = torch.arange(self.batch_size, device=self.device)
        
#         # Generate new input sizes
#         base_size = self.num_layers * self.num_agents
#         random_offsets = torch.randint(-2 * self.num_agents, self.max_input, 
#                                      (len(env_ids),), device=self.device)
#         self._input_nums[env_ids] = base_size + random_offsets
        
#         # Generate new inputs for specified environments
#         for i, env_id in enumerate(env_ids):
#             input_num = self._input_nums[env_id].item()
#             self._input = input_generator_torch(self.max_input, input_num, self.device)
        
#         # Reset FIFO and counters for specified environments
#         self._fifo[env_ids] = -1
#         self._fifo_lengths[env_ids] = 0
#         self._pop_counters[env_ids] = 0

#     def get_input(self) -> torch.Tensor:
#         """Get input buffer for all environments"""
#         # This is a simplified version - in practice you'd need to handle variable input sizes
#         max_input_size = torch.max(self._input_nums).item()
#         inputs = torch.zeros(self.batch_size, self.num_agents, device=self.device, dtype=torch.long)
        
#         # For now, return a placeholder - this needs more sophisticated handling
#         return inputs

#     def observe_fifo(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Observe FIFO state for all environments"""
#         # Get top layer data
#         fifo_data = torch.full((self.batch_size, ), self.num_agents, -1, device=self.device, dtype=torch.long)
#         for b in range(self.batch_size):
#             for i in range(self.num_agents):
#                 if self._fifo_lengths[b, i] > 0:
#                     fifo_data[b, i] = self._fifo[b, i, self._fifo_lengths[b, i] - 1]
        
#         # Convert -1 to 0
#         fifo_data = torch.where(fifo_data == -1, torch.tensor(0, device=self.device), fifo_data)
        
#         # Calculate height bias
#         fifo_heights = self._fifo_lengths.clone()
#         base_heights = torch.min(fifo_heights, dim=1, keepdim=True)[0]
#         fifo_heights = fifo_heights - base_heights
        
#         return fifo_data, fifo_heights

#     def height_check(self) -> torch.Tensor:
#         """Check height availability for all environments"""
#         return self._fifo_lengths < self.num_layers

#     def is_game_over(self) -> torch.Tensor:
#         """Check game over status for all environments"""
#         no_input = self._input.size(1) == 0 if self._input.size(0) > 0 else True
#         no_available = ~torch.any(self.height_check(), dim=1)
#         return torch.tensor([no_input] * self.batch_size, device=self.device) | no_available

#     def get_static_action_mask(self) -> torch.Tensor:
#         """Get action mask"""
#         return self._action_mask


if __name__ == "__main__":

    seed = 11
    torch.manual_seed(seed)

    # Test the PyTorch version
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test single container
    container = ContainerTorch(num_agents=4, num_layers=32, max_input=32, device=device, render=True)
    print(container.observe_fifo())

    round_num = 1
    while True:
        print(container.get_input())
        # action = convert_4d_index_to_1d((1, 3, 2, 4))
        # action = container.convert_original_to_reduced_index(action)
        # print("Converted action:", action)
        action = 71
        game_over_flag = container.pop_push(action)
        container._render_fifo_torch(container._fifo, container._fifo_lengths, round_num)
        round_num += 1
        if game_over_flag:
            break
        
    print(container.get_input())
    print(container.evaluate())
    print("find pop amount and input amount:", container.get_render_msg())

    # Test batch container
    # batch_container = BatchContainerTorch(batch_size=4, num_agents=4, num_layers=8, max_input=8, device=device)
    # print(f"Batch container initialized: {batch_container}")


        
