from easydict import EasyDict
from datetime import datetime
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 64  # Increased for better parallelization
n_episode = 64
evaluator_env_num = 8
num_simulations = 64
update_per_collect = 8
batch_size = 512
max_env_step = int(1e6)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

global_seed = 33

current_time = datetime.now()

rl4ic_muzero_torch_config = dict(
    exp_name=f'RL4IC/data_muzero_torch/rl4ic_muzero_torch_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed{global_seed}',
    wandb_name=f"LightZero-RL4IC-{current_time.month}-{current_time.day}",
    env=dict(
        env_id='RL4IC_TORCH-v0',  # Use PyTorch optimized environment
        stop_value=200,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(
            shared_memory=False,
            # Enable GPU parallel processing
            gpu_parallel=True,
            # Batch processing for better GPU utilization
            batch_mode=True,
            batch_size=64,  # Process 64 environments in parallel
        ),
        # PyTorch specific configurations
        device='cuda',  # Use GPU
        num_sub_agents=4,
        num_layers=32,
        max_input=32,
    ),
    policy=dict(
        use_wandb=True,
        model=dict(
            observation_shape=12,  # 3D observation shape
            action_space_size=625,
            # image_channel=4,
            # num_res_blocks=1,
            # num_channels=32,
            # support_scale=10,
            # reward_support_size=21,
            # value_support_size=21,
            # Ensure model can handle 3D observations
            model_type='mlp',  # Use convolutional model for 3D observations
            # model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            # self_supervised_learning_loss=False,  # Disable SSL for vector observations
            discrete_action_encoding_type='one_hot',
            # norm_type='BN', 
            # # PyTorch optimizations
            use_torch_optimizations=True,
            # mixed_precision=True,  # Enable mixed precision training
        ),
        wandb_logger=dict(
            gradient_logger=True, video_logger=False, plot_logger=True, action_logger=True, return_logger=True
        ),
        # Model path and device
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=0,  # Disable SSL loss for vector observations
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(100),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # GPU memory optimizations
        mini_infer_size=32,  # Process inference in smaller batches
        # Parallel processing optimizations
        parallel_collect=True,
        parallel_eval=True,
        # MCTS optimizations
        mcts_use_parallel=True,
        mcts_num_parallel=8,  # Number of parallel MCTS simulations
    ),
)

rl4ic_muzero_torch_config = EasyDict(rl4ic_muzero_torch_config)
main_config = rl4ic_muzero_torch_config

rl4ic_muzero_torch_create_config = dict(
    env=dict(
        type='RL4IC_TORCH',  # Use PyTorch optimized environment
        import_names=['zoo.board_games.rl4ic.envs.rl4ic_env_torch'],
    ),
    env_manager=dict(
        type='subprocess',
        context='spawn',
        # Enable GPU parallel processing
        gpu_parallel=True,
        # Batch processing
        batch_mode=True,
        batch_size=64,
    ),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
rl4ic_muzero_torch_create_config = EasyDict(rl4ic_muzero_torch_create_config)
create_config = rl4ic_muzero_torch_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        """
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
        """
        from lzero.entry import train_muzero_with_gym_env as train_muzero

    train_muzero([main_config, create_config], seed=global_seed, model_path=main_config.policy.model_path, max_env_step=max_env_step) 