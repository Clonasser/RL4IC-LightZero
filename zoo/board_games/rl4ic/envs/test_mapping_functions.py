#!/usr/bin/env python3
"""
Test script for mapping functions
"""

import torch
import numpy as np

def test_mapping_functions():
    """Test the mapping functions"""
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create original space (1D tensor)
    original_size = 10
    original_space = torch.arange(original_size, device=device)
    
    # Create mask (some actions are invalid)
    mask = torch.tensor([True, False, True, True, False, True, False, True, True, False], 
                       device=device, dtype=torch.bool)
    
    print(f"Original space: {original_space}")
    print(f"Mask: {mask}")
    
    # Test compute_reduced_space_shape
    from zoo.board_games.rl4ic.envs.rl4ic_env_torch import RL4ICEnvTorch
    
    reduced_size = RL4ICEnvTorch.compute_reduced_space_shape(original_space, mask)
    print(f"Reduced space size: {reduced_size}")
    
    # Test create_reduced_to_original_mapping
    reduced_to_original = RL4ICEnvTorch.create_reduced_to_original_mapping(original_space, mask)
    print(f"Reduced to original mapping: {reduced_to_original}")
    
    # Test create_original_to_reduced_mapping
    original_to_reduced = RL4ICEnvTorch.create_original_to_reduced_mapping(original_space, mask)
    print(f"Original to reduced mapping: {original_to_reduced}")
    
    # Test conversion functions
    print("\nTesting conversion functions:")
    
    # Test reduced to original conversion
    for i in range(reduced_size):
        original_idx = RL4ICEnvTorch.convert_reduced_index_to_original(i, reduced_to_original)
        print(f"Reduced {i} -> Original {original_idx}")
    
    # Test original to reduced conversion
    for i in range(original_size):
        reduced_idx = RL4ICEnvTorch.convert_original_index_to_reduced(i, original_to_reduced)
        if reduced_idx != -1:
            print(f"Original {i} -> Reduced {reduced_idx}")
        else:
            print(f"Original {i} -> Invalid (masked out)")
    
    # Test round-trip conversion
    print("\nTesting round-trip conversion:")
    for i in range(reduced_size):
        original_idx = RL4ICEnvTorch.convert_reduced_index_to_original(i, reduced_to_original)
        recovered_reduced = RL4ICEnvTorch.convert_original_index_to_reduced(original_idx, original_to_reduced)
        print(f"Reduced {i} -> Original {original_idx} -> Recovered reduced {recovered_reduced}")
        assert recovered_reduced == i, f"Round-trip conversion failed for reduced index {i}"
    
    print("\nAll tests passed!")

def test_with_rl4ic_action_space():
    """Test with actual RL4IC action space"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create original action space (5^4 = 625 actions)
    num_sub_agents = 4
    original_size = (num_sub_agents + 1) ** num_sub_agents  # 5^4 = 625
    original_space = torch.arange(original_size, device=device)
    
    # Create a mask that removes some actions (simulate static mask)
    # For demonstration, let's mask out actions that have duplicate indices when converted to 4D
    from zoo.board_games.rl4ic.envs.rl4ic_containers_torch import convert_1d_index_to_4d
    
    mask = torch.ones(original_size, device=device, dtype=torch.bool)
    
    # Apply the same logic as in _precompute_action_mask
    for idx in range(original_size):
        action_tuple = convert_1d_index_to_4d(idx, (5, 5, 5, 5))
        # Check for duplicates (except 0)
        seen = set()
        for val in action_tuple:
            if val != 0 and val in seen:
                mask[idx] = False
                break
            seen.add(val)
    
    print(f"RL4IC action space test:")
    print(f"Original size: {original_size}")
    print(f"Valid actions: {torch.sum(mask).item()}")
    print(f"Invalid actions: {torch.sum(~mask).item()}")
    print(f"Reduction ratio: {torch.sum(mask).item()/original_size:.2%}")
    
    # Test mapping functions
    from zoo.board_games.rl4ic.envs.rl4ic_env_torch import RL4ICEnvTorch
    
    reduced_size = RL4ICEnvTorch.compute_reduced_space_shape(original_space, mask)
    reduced_to_original = RL4ICEnvTorch.create_reduced_to_original_mapping(original_space, mask)
    original_to_reduced = RL4ICEnvTorch.create_original_to_reduced_mapping(original_space, mask)
    
    print(f"Reduced size: {reduced_size}")
    
    # Test a few conversions
    test_reduced_indices = [0, 10, 50, 100, reduced_size-1]
    for reduced_idx in test_reduced_indices:
        if reduced_idx < reduced_size:
            original_idx = RL4ICEnvTorch.convert_reduced_index_to_original(reduced_idx, reduced_to_original)
            recovered_reduced = RL4ICEnvTorch.convert_original_index_to_reduced(original_idx, original_to_reduced)
            print(f"Reduced {reduced_idx} -> Original {original_idx} -> Recovered {recovered_reduced}")
            assert recovered_reduced == reduced_idx
    
    print("RL4IC action space test completed successfully!")

if __name__ == "__main__":
    print("Running mapping function tests...")
    
    try:
        test_mapping_functions()
        print("\n" + "="*50 + "\n")
        test_with_rl4ic_action_space()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 