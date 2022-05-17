default_args = {
  # Environment
  "env_name":{
  },

  # Player: PPO 
  "player_1":{
    "agent_name": "PPO",
    "type": "on_policy",
    "agent":{
      "beta": 1.0,
      "policy_loss_type": "clipped_surrogate",
      "entropy_coeff": 0.0,  
      "c1": 1.0,
      "c2": 1.0,
      "clip_range": 0.2,
      "target_kl": 0.01,
      "adaptive_kl_coeff": False,
      "train_policy_iters": 80,
      "train_v_iters": 80,

      "v_network":{
        "hidden_dims": [64,64],
        "optimizer_class": "Adam",
        "learning_rate":0.001,
        "act_fn": "tanh",
        "out_act_fn": "identity"
      },

      "policy_network":{
        "hidden_dims": [64, 64],
        "optimizer_class": "Adam",
        "deterministic": False,
        "learning_rate":0.0003,
        "act_fn": "tanh",
        "out_act_fn": "identity",
        "re_parameterize": False,
        "fix_std": True,
        "paramterized_std": True,
        "stablize_log_prob": False
      },
    },
    "buffer":{
      "max_trajectory_length": 1000,
      "advantage_type": "gae",
      "size": 4000,
      "gamma": 0.99,
      "normalize_advantage": True,
      "gae_lambda": 0.97
    },
  },

  # Player 2: SAC
  "player_2":{
    "agent_name": "SAC",
    "type_name": "off_policy",
    "agent":{
      "gamma": 0.99,
      "update_target_network_interval": 1,
      "target_smoothing_tau": 0.005,
      "alpha": 0.2,
      "reward_scale": 1.0,
      "q_network":{
        "hidden_dims": [256,256],
        "optimizer_class": "Adam",
        "learning_rate":0.0003,
        "act_fn": "relu",
        "out_act_fn": "identity"
      },
      "opponent_agent":{
        "type":"random"
      },
      "policy_network":{
        "hidden_dims": [256,256],
        "optimizer_class": "Adam",
        "learning_rate":0.0003,
        "act_fn": "relu",
        "out_act_fn": "identity",
        "re_parameterize": True, 
        "log_var_min": -20, 
        "log_var_max": 2
      },
      "entropy":{
        "automatic_tuning": True,
        "learning_rate": 0.0003,
        "optimizer_class": "Adam"
      },
    },
    "buffer":{
      "max_buffer_size": 1000000
    }, 
  },   

  # TwoAgentGameTrainer
  "trainer":{
    "max_env_steps": 3000000,
    "num_env_steps_per_epoch": 4000,
    "batch_size": 64,
    "eval_interval": 10,
    "num_eval_trajectories": 5,
    "snapshot_interval": 100,
    "start_timestep": 0,
    "save_video_demo_interval": 300,
    "log_interval": 1
  },
}
