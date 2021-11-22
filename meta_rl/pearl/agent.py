import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from common.agents import BaseAgent
from common.networks import MLPNetwork, PolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
import numpy as np
from common import util 

class SACAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        update_target_network_interval=50, 
        target_smoothing_tau=0.1,
        alpha=0.2,
        **kwargs):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(SACAgent, self).__init__()
        #save parameters
        self.args = kwargs

        #initilze networks
        self.latent_dim = kwargs['latent_dim']
        self.q1_network = MLPNetwork(obs_dim + action_dim + self.latent_dim, 1, **kwargs['q_network'])
        self.q2_network = MLPNetwork(obs_dim + action_dim + self.latent_dim, 1,**kwargs['q_network'])
        self.policy_network = PolicyNetwork(obs_dim, action_space,  ** kwargs['policy_network'])
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        if self.use_next_obs_in_context:
            context_encoder_input_dim = 2 * obs_dim + action_dim + 1
        else:
            context_encoder_input_dim =  obs_dim + action_dim + 1
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        if self.use_information_bottleneck:
            context_encoder_output_dim = kwargs['context_encoder_network']['latent_dim'] * 2
        else:
            context_encoder_output_dim = kwargs['context_encoder_network']['latent_dim']
        self.context_encoder_network = MLPNetwork(context_encoder_input_dim, context_encoder_output_dim, **kwargs['context_encoder_network'])

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.context_encoder_network = self.context_encoder_network.to(util.device)

        #register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            'policy_network': self.policy_network,
            'context_encoder_network': self.context_encoder_network
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        self.context_encoder_optimizer = get_optimizer(kwargs['context_encoder_network']['optimizer_class'], self.policy_network, kwargs['context_encoder_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau

    def update(self, context_batch, data_batch):
        num_tasks = len(context_batch)

        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = data_batch
        
        # infer z from context
        self.infer_z_posterior(context_batch, do_sampling=True)
        task_z_batch = self.sample_z()

        # expand z to concatenate with obs, action batches
        num_tasks, batch_size, obs_dim = obs_batch.shape
        #flatten obs
        obs_batch = obs_batch.view(num_tasks * batch_size, -1)
        #expand z to match obs batch
        task_z_batch = [task_z.repeat(batch_size, 1 ) for task_z in task_z_batch]
        task_z_batch = torch.cat(task_z_batch, dim=0)

        # get policy output
        policy_input = torch.cat([obs_batch, task_z_batch.detach()], dim=1)
        action_mean, action_log_std = self.policy_network(policy_input)





        curr_state_q1_value = self.q1_network(obs_batch, action_batch)
        curr_state_q2_value = self.q2_network(obs_batch, action_batch)
        new_curr_state_action, new_curr_state_log_pi, _, _ = self.policy_network.sample(obs_batch)
        next_state_action, next_state_log_pi, _, _ = self.policy_network.sample(next_obs_batch)

        new_curr_state_q1_value = self.q1_network(obs_batch, new_curr_state_action)
        new_curr_state_q2_value = self.q2_network(obs_batch, new_curr_state_action)

        next_state_q1_value = self.target_q1_network(next_obs_batch, next_state_action)
        next_state_q2_value = self.target_q2_network(next_obs_batch, next_state_action)
        next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
        target_q = (next_state_min_q - self.alpha * next_state_log_pi)
        target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)

        #compute q loss
        
        q1_loss = F.mse_loss(curr_state_q1_value, target_q.detach())
        q2_loss = F.mse_loss(curr_state_q2_value, target_q.detach())

        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()

        #compute policy loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        #compute entropy loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().numpy()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.detach().cpu().numpy()
        else:
            alpha_loss_value = 0.
            alpha_value = self.alpha
        self.tot_update_count += 1

        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.policy_optimizer.step()

        self.try_update_target_network()
        
        return {
            "loss/q1": q1_loss_value, 
            "loss/q2": q2_loss_value, 
            "loss/policy": policy_loss_value, 
            "loss/entropy": alpha_loss_value, 
            "others/entropy_alpha": alpha_value
        }
        
    def clear_z(self, num_tasks):
        pass

    def infer_z_posterior(self, context_batch):
        pass

    def sample_z_from_posterior(self):
        pass



    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
            util.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            
    def select_action(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        action, log_prob, mean, std = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0], log_prob
        else:
            return action.detach().cpu().numpy()[0], log_prob