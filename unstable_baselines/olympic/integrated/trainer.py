import os
from time import time
from random import sample
from operator import itemgetter
from tensorboard import summary

import torch
from tqdm import trange
import numpy as np
import cv2


from unstable_baselines.common import util


class Player:

    def __init__(self, name, agent, buffer, agent_type, index,
                 start_timestep,
                 random_policy_timestep,
                 batch_size,
                 load_path=None,
                 ):
        self.name = name
        self.agent = agent
        self.buffer = buffer
        self.type = agent_type
        self.index = index
        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep
        self.batch_size = batch_size

        self.freeze_model = False
        if load_path is not None:
            self.load_model()
            self.freeze_model = True

    def save_model(self, timestamp):
        self.agent.snapshot(timestamp)    
    
    def load_model(self, load_dir):
        self.agent.load_snapshot(load_dir)


class TwoAgentGame:

    def __init__(self,
                 player_1,
                 player_2,
                 train_env,
                 eval_env,

                 max_env_steps,
                 max_trajectory_length,
                 num_env_steps_per_epoch,
                 
                 eval_interval,
                 num_eval_trajectories,
                 snapshot_interval,
                 save_video_demo_interval,
                 log_interval,
                 **kwargs):

        self.player_1 = player_1
        self.player_2 = player_2
        self.train_env = train_env
        self.eval_env = eval_env
        
        self.max_env_steps = max_env_steps
        self.max_trajectory_length = max_trajectory_length
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.max_epoch = self.max_env_steps // self.num_env_steps_per_epoch

        # hyperparameters
        self.eval_interval = eval_interval
        self.num_eval_trajectories = num_eval_trajectories
        self.snapshot_interval = snapshot_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.log_interval = log_interval
        

    def get_player_action(self, player, obs, tot_env_steps):
        """
        """
        if player.type == 'on_policy':
            action, log_prob = itemgetter('action', 'log_prob')(player.agent.select_action(obs))
        elif player.type == 'off_policy':
            if tot_env_steps < player.random_policy_timestep:
                action = self.train_env.action_space.sample()
                log_prob = None
            else:
                action, log_prob = itemgetter('action', 'log_prob')(player.agent.select_action(obs))
        else:
            raise ValueError(f"Wrong {player.type} type for player. The valid type includes"
                             f"'on_policy' and 'off_policy.")
        return action, log_prob

    def _wrap_action(self, action):
        """ warp action for olympics-integrated envrionment
        """
        if len(action.shape) > 1:
            action = action.squeeze()
        # return [[action[0]], [action[1]]]
        return action

    def store_player_buffer(self, player, obs, action, reward, next_obs, done, log_prob):
        """
        """
        if player.freeze_model:
            return None
        if player.type == 'on_policy':
            value = player.agent.estimate_value(obs)[0]
            player.buffer.add_transition(obs, action, reward, value, log_prob)
        elif player.type == 'off_policy':
            player.buffer.add_transition(obs, action, next_obs, reward, done)
        else:
            raise ValueError(f"Wrong {player.type} type for buffer. The valid type includes"
                             f"['on_policy', 'off_policy', 'model'].")

    def calculate_gae_when_traj_end(self, player, obs, timeout, epoch_ended):
        """
        """
        if player.freeze_model:
            return None
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

    def update_agent(self, player, epoch_ended, tot_env_steps):
        """
        """
        if player.freeze_model:
            return None
        if tot_env_steps < player.start_timestep:   # update after start_timestep
            return None
        if player.type == 'on_policy':
            if epoch_ended:
                start_time = time()
                data_batch = player.buffer.get()
                loss_dict = player.agent.update(data_batch)
                train_on_policy_time = time() - start_time
                loss_dict[f"times/{player.name}"] = train_on_policy_time
                return loss_dict
            else:
                return None
        elif player.type == 'off_policy':
            start_time = time()
            data_batch = player.buffer.sample(player.batch_size)
            loss_dict = player.agent.update(data_batch)
            train_off_policy_time = time() - start_time
            loss_dict[f"times/{player.name}"] = train_off_policy_time
            return loss_dict
        else:
            raise ValueError(f"Wrong {player.type} type for buffer. The valid type includes"
                             f"'on_policy' and 'off_policy.")
        
    def train(self):
        # TODO(mimeku): 核实流程
        train_traj_returns_player_1 = [0]
        train_traj_returns_player_2 = [0]
        trian_traj_lengths = [0]
        tot_env_steps = 0
        traj_return_player_1 = 0
        traj_return_player_2 = 0

        traj_length = 0
        done = False
        obs, switch_game = self.train_env.reset()    # TODO(mimeku): 处理switch_game

        for epoch_id in trange(self.max_epoch):
            self.pre_iter()
            match_count = 0
            win_times_1 = 0

            for epoch_step in trange(self.num_env_steps_per_epoch):
                # get action
                action_1, log_prob_1 = self.get_player_action(self.player_1, obs[self.player_1.index], tot_env_steps)
                action_2, log_prob_2 = self.get_player_action(self.player_2, obs[self.player_2.index], tot_env_steps)
                wrap_action_1 = self._wrap_action(action_1)
                wrap_action_2 = self._wrap_action(action_2)
                action = [wrap_action_1, wrap_action_2] if self.player_1.index == 0 else [wrap_action_2, wrap_action_1]

                next_obs, joint_reward, done, info_before, info = self.train_env.step(action)
                next_obs, switch_game = next_obs

                traj_return_player_1 += joint_reward[self.player_1.index]
                traj_return_player_2 += joint_reward[self.player_2.index]
                traj_length += 1

                timeout = traj_length == self.max_trajectory_length
                terminal = done or timeout
                epoch_ended = epoch_step == self.num_env_steps_per_epoch - 1

                # process done
                if timeout or epoch_ended:
                    done = False

                # store buffer
                self.store_player_buffer(self.player_1, 
                                         obs[self.player_1.index], 
                                         action_1, 
                                         joint_reward[self.player_1.index], 
                                         next_obs[self.player_1.index], 
                                         done,
                                         log_prob_1)
                self.store_player_buffer(self.player_2, 
                                         obs[self.player_2.index], 
                                         action_2, 
                                         joint_reward[self.player_2.index], 
                                         next_obs[self.player_2.index],
                                         done,
                                         log_prob_2)

                # transitate after storing transition
                obs = next_obs

                # calculate the GAE value for OnlineBuffer used by on-policy agent
                self.calculate_gae_when_traj_end(self.player_1, obs[self.player_1.index], timeout, epoch_ended)
                self.calculate_gae_when_traj_end(self.player_2, obs[self.player_2.index], timeout, epoch_ended)

                # train agent
                loss_dict_player_1 = self.update_agent(self.player_1, epoch_ended, tot_env_steps)
                loss_dict_player_2 = self.update_agent(self.player_2, epoch_ended, tot_env_steps)

                # process training logic
                tot_env_steps += 1
                if terminal or epoch_ended:
                    match_count += 1
                    win_times_1 = win_times_1 + 1 if traj_return_player_1 > traj_return_player_2 else win_times_1
                    
                    obs, switch_game = self.train_env.reset()
                    train_traj_returns_player_1.append(traj_return_player_1)
                    train_traj_returns_player_2.append(traj_return_player_2)
                    trian_traj_lengths.append(traj_length)
                    traj_length = 0
                    traj_return_player_1 = 0
                    traj_return_player_2 = 0

            log_infos = {}
            log_infos.update(loss_dict_player_1)
            log_infos.update(loss_dict_player_2)
            
            self.post_iter(log_infos, tot_env_steps)

    @torch.no_grad()
    def evaluate(self):
        """
        """
        traj_lengths = []
        win_times_1 = 0
        for traj_id in range(self.num_eval_trajectories):
            traj_return_1 = 0
            traj_return_2 = 0
            traj_length = 0
            obs, switch_game = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action_1, log_prob_1 = self.get_player_action(self.player_1, obs[self.player_1.index], self.player_1.random_policy_timestep+1)
                action_2, log_prob_2 = self.get_player_action(self.player_2, obs[self.player_2.index], self.player_2.random_policy_timestep+1)
                wrap_action_1 = self._wrap_action(action_1)
                wrap_action_2 = self._wrap_action(action_2)
                action = [wrap_action_1, wrap_action_2] if self.player_1.index == 0 else [wrap_action_2, wrap_action_1]
                next_obs, joint_reward, done, _, _ = self.eval_env.step(action)
                next_obs, switch_game = next_obs
                # record
                traj_return_1 += joint_reward[self.player_1.index]
                traj_return_2 += joint_reward[self.player_2.index]
                traj_length += 1
                if done:
                    break
            win_times_1 = win_times_1 + 1 if traj_return_1 > traj_return_2 else win_times_1
            traj_lengths.append(traj_length)
        win_rate_1 = win_times_1 / self.num_eval_trajectories
        win_rate_2 = 1 - win_rate_1
        return {
            "performance/eval_length": np.mean(traj_lengths),
            "performance/eval_win_rate_1": win_rate_1,
            "performance/eval_win_rate_2": win_rate_2,
        }

    def pre_iter(self):
        self.ite_start_time = time()

    def post_iter(self, log_info_dict, timestamp):
        """
        """
        # record loss
        if timestamp % self.log_interval == 0:
            for loss_name in log_info_dict:
                util.logger.log_var(loss_name, log_info_dict[loss_name], timestamp)
        # record evaluation result
        if timestamp % self.eval_interval == 0:
            eval_start_time = time()
            log_dict = self.evaluate()
            eval_used_time = time() - eval_start_time
            for log_key in log_dict:
                util.logger.log_var(log_key, log_dict[log_key], timestamp)
            util.logger.log_var("times/evaluate", eval_used_time, timestamp)
            win_rate_1 = log_dict['performance/eval_win_rate_1']
            summary_str = f"Timestamp:{timestamp}\t" + \
                    f"Win rate({self.player_1.name} vs. {self.player_2.name}): {win_rate_1} vs. {1 - win_rate_1}"
            util.logger.log_str(summary_str)
        # save model
        if timestamp % self.snapshot_interval == 0:
            self.player_1.save_model()
            self.player_2.save_model()
        # save video demo
        if self.save_video_demo_interval > 0 and timestamp % self.save_video_demo_interval == 0:
            self.save_video_demo(timestamp)

    def save_video_demo(self, ite, width=256, height=256, fps=30):
        """
        """
        video_demo_dir = os.path.join(util.logger.log_dir, "demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        video_size = (height, width)
        video_save_path = os.path.join(video_demo_dir, f"ite_{ite}.mp4")

        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

        # rollout to generate pictures and write video
        obs, switch_game = self.eval_env.reset()
        img = self.eval_env.render(mode="rgb_array", width=width, height=height)
        video_writer.write(img)
        for step in range(self.max_trajectory_length):
            action_1, log_prob_1 = self.get_player_action(self.player_1, obs[self.player_1.index], self.player_1.random_policy_timestep+1)
            action_2, log_prob_2 = self.get_player_action(self.player_2, obs[self.player_2.index], self.player_2.random_policy_timestep+1)
            wrap_action_1 = self._wrap_action(action_1)
            wrap_action_2 = self._wrap_action(action_2)
            action = [wrap_action_1, wrap_action_2] if self.player_1.index == 0 else [wrap_action_2, wrap_action_1]
            next_obs, joint_reward, done, _, _ = self.eval_env.step(action)
            next_obs, switch_game = next_obs
            img = self.eval_env.render(mode="rgb_array", width=width, height=height)
            video_writer.write(img)
            if done:
                break
        video_writer.release()