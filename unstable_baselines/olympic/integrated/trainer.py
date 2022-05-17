import os
from random import sample
from operator import itemgetter

import torch
from tqdm import trange



class Player:

    def __init__(self, agent, buffer, type, index):

        self.agent = agent
        self.buffer = buffer
        self.type = type
        self.index = index

        

class TwoAgentGame:

    def __init__(self,
                 player_1,
                 player_2,
                #  agent_1,
                #  buffer_1,

                #  train_env,
                #  eval_env,
                #  buffer,
                #  batch_size,
                #  max_env_steps,
                #  start_timestep,
                #  random_policy_timestep,
                #  load_dir="",
                 **kwargs):

        self.player_1 = player_1
        self.player_2 = player_2

        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep

        self.train_env = train_env
        self.eval_env = eval_env
        self.max_trajectory_length = max_trajectory_length
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_trajectories = num_eval_trajectories
        self.save_video_demo_interval = save_video_demo_interval
        self.snapshot_interval = snapshot_interval




        # hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def get_player_action(self, player, obs, tot_env_steps):
        """ 得到智能体应该输出的动作
        """
        if player.type == 'on_policy':
            action, log_prob = itemgetter('action', 'log_prob')(player.agent.select_action(obs))
        elif player.type == 'off_policy':
            log_prob = None
            if tot_env_steps < self.random_policy_timestep:
                action = self.train_env.action_space.sample()
            else:
                action = player.agent.select_action(obs)
        else:
            raise ValueError(f"Wrong {player.type} type for player. The valid type includes"
                             f"'on_policy' and 'off_policy.")
        return action, log_prob

    def store_player_buffer(self, player, obs, action, reward, next_obs, done):
        """
        """
        if player.type == 'on_policy':
            value = player.agent.estimate_value(obs)[0]
            player.buffer.add_transition(obs, action, reward, value, log_prob)
        elif player.buffer.type == 'off policy':
            player.buffer.add_transition(obs, action, next_obs, reward, done)
        else:
            raise ValueError(f"Wrong {player.type} type for buffer. The valid type includes"
                             f"'on_policy' and 'off_policy.")

    def calculate_gae_when_traj_end(self, player, obs, timeout, epoch_ended):
        """
        """
        if player.type == 'on_policy':
            if timeout or epoch_ended:
                # bootstrap
                last_v = player.agent.estimate_value(obs)
            else:
                last_v = 0
            player.buffer.finish_path(last_v)
        elif player.type == 'off_policy':
            return None
        else:
            raise ValueError(f"Wrong {player.type} type for player. The valid type includes"
                             f"'on_policy' and 'off_policy.")
        

    def train(self):
        train_traj_returns = [0]
        train_traj_returns_player_1 = [0]
        train_traj_returns_player_2 = [0]
        trian_traj_lengths = [0]
        tot_env_steps = 0
        traj_return_player_1 = 0
        traj_return_player_2 = 0

        traj_length = 0
        done = False
        obs = self.train_env.reset()    # TODO(mimeku): 处理switch_game

        for epoch_id in trange(self.max_epoch):
            self.pre_iter()
            log_infos = {}

            for env_step in trange(self.num_env_steps_per_epoch):

                # get action
                action_1, log_prob_1 = self.get_player_action(self.player_1, obs[self.player_1.index], tot_env_steps)
                action_2, log_prob_2 = self.get_player_action(self.player_2, obs[self.player_2.index], tot_env_steps)
                action = [action_1, action_2] if self.player_1.index == 0 else [action_2, action_2]

                next_obs, joint_reward, done, info_before, info = self.train_env.step(action)   # 处理switch_game

                traj_return_player_1 += joint_reward[self.player_1.index]
                traj_return_player_2 += joint_reward[self.player_2.index]
                traj_length += 1

                timeout = traj_length == self.max_trajectory_length
                terminal = done or timeout
                epoch_ended = env_step == self.num_env_steps_per_epoch - 1

                # process done
                if timeout or epoch_ended:
                    done = False

                # store buffer
                self.store_player_buffer(self.player_1, 
                                         obs[self.player_1.index], 
                                         action_1, 
                                         joint_reward[self.player_1.index], 
                                         next_obs, 
                                         done)
                self.store_player_buffer(self.player_2, 
                                         obs[self.player_2.index], 
                                         action_2, 
                                         joint_reward[self.player_2.index], 
                                         next_obs,
                                         done)

                # transitate after storing transition
                obs = next_obs

                # calculate the GAE value for OnlineBuffer used by on-policy agent
                self.calculate_gae_when_traj_end(self.player_1, obs[self.player_1.index], timeout, epoch_ended)
                self.calculate_gae_when_traj_end(self.player_2, obs[self.player_2.index], timeout, epoch_ended)

                # process training logic
                if terminal or epoch_ended:
                    # TODO(mimeku): wether record the agent winning this epoch
                    winned = 1 if joint_reward[self.player_1.index] > joint_reward[self.player_2.index] else 2
                    match_results.append(winned)
                    obs = self.train_env.reset()
                    train_traj_returns_player_1.append(traj_return_player_1)
                    train_traj_returns_player_2.append(traj_return_player_2)
                    traj_length = 0
                    traj_return_player_1 = 0
                    traj_return_player_2 = 0
                
                log_infos['performance/train_return_player_1'] = train_traj_returns_player_1[-1]
                log_infos['performance/train_return_player_2'] = train_traj_returns_player_2[-1]
                tot_env_steps += 1

                # train agent
                if tot_env_steps < self.start_timestep:
                    continue

    @torch.no_grad()
    def evaluate(self):
        pass

    def save_video_demo(self, ite, width=256, height=256, fps=30):
        pass