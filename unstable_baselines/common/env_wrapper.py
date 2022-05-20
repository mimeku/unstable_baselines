import gym
import numpy as np
from gym import spaces

from typing import Dict, List

from unstable_baselines.envs.olympics_engine.generator import create_scenario
from unstable_baselines.envs.olympics_engine.scenario.running import Running
from unstable_baselines.envs.olympics_engine.scenario.running_competition import Running_competition
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True) 


MUJOCO_SINGLE_ENVS = [
    'Ant-v2', 'Ant-v3',
    'HalfCheetah-v2', 'HalfCheetah-v3',
    'Hopper-v2', 'Hopper-v3',
    'Humanoid-v2', 'Humanoid-v3',
    'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2',
    'Swimmer-v2', 'Swimmer-v3',
    'Walker2d-v2', 'Walker2d-v3',
    'Pusher-v2',
    'Reacher-v2',
    'Striker-v2',
    'Thrower-v2',
    'CartPole-v1',
    'MountainCar-v0'
    ]

MUJOCO_META_ENVS = [
    'point-robot', 'sparse-point-robot', 'walker-rand-params', 
    'humanoid-dir', 'hopper-rand-params', 'ant-dir', 
    'cheetah-vel', 'cheetah-dir', 'ant-goal']

JIDIAI = ['olympics-integrated', 'running-competition', 'table-hockey', 'football', 'wrestling']

METAWORLD_ENVS = ['MetaWorld']

MBPO_ENVS = [
    'AntTruncatedObs-v2',
    'HumanoidTruncatedObs-v2',
    ]
ATARI_ENVS = ['']


def get_env(env_name, **kwargs):
    if env_name in MUJOCO_SINGLE_ENVS:
        return gym.make(env_name, **kwargs)
    elif env_name in MUJOCO_META_ENVS:
        from unstable_baselines.envs.mujoco_meta.rlkit_envs import ENVS as MUJOCO_META_ENV_LIB
        return MUJOCO_META_ENV_LIB[env_name](**kwargs)
    elif env_name in METAWORLD_ENVS:
        raise NotImplementedError
    elif env_name in MBPO_ENVS:
        from unstable_baselines.envs.mbpo import register_mbpo_environments
        register_mbpo_environments()
        env = gym.make(env_name)
        return env
    elif env_name in JIDIAI:
        env = make_jidi_env(env_name)
        return env
    else:
        print("Env {} not supported".format(env_name))
        exit(0)


def make_jidi_env(env_name):
    """ create envrionment in JIDI
    """
    if env_name == 'olympics-integrated':
        from unstable_baselines.envs.olympics_integrated.chooseenv import make as make_olympics_integrated_env
        return make_olympics_integrated_env(env_name)
    # single environment
    elif env_name == 'running-competition':
        from unstable_baselines.envs.olympics_engine.scenario import Running_competition
        env = JidiRunningEnv()
        return JidiSubEnvWrapper(env)
    elif env_name == 'table-hockey':
        from unstable_baselines.envs.olympics_engine.generator import create_scenario
        from unstable_baselines.envs.olympics_engine.scenario import table_hockey
        Gamemap = create_scenario(env_name)
        env = table_hockey(Gamemap)
        return JidiSubEnvWrapper(env)
    elif env_name == 'football':
        from unstable_baselines.envs.olympics_engine.generator import create_scenario
        from unstable_baselines.envs.olympics_engine.scenario import football
        Gamemap = create_scenario(env_name)
        env = football(Gamemap)
        return JidiSubEnvWrapper(env)
    elif env_name == 'wrestling':
        from unstable_baselines.envs.olympics_engine.generator import create_scenario
        from unstable_baselines.envs.olympics_engine.scenario import wrestling
        Gamemap = create_scenario(env_name)
        env = wrestling(Gamemap)
        return JidiSubEnvWrapper(env)


class JidiRunningEnv:
    """ Make the `running-competition` environment's map change when calling reset
    """
    def __init__(self, map_id_list=[1, 2, 3, 4]):
        self.map_id_list = map_id_list
        self.reset()
        
    @property
    def max_step(self):
        return self.env.max_step
    
    @max_step.setter
    def max_step(self, value):
        self.env.max_step = value

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return self._obs2dict(next_obs), reward, done, info

    def _obs2dict(self, obs):
        dict_obs_list = []
        for sub_obs in obs:
            dict_obs_list.append({'agent_obs': sub_obs})
        return dict_obs_list

    def reset(self):
        map_id = np.random.choice(self.map_id_list)
        Gamemap = create_scenario('running-competition')
        self.env = Running_competition(meta_map=Gamemap, 
                                  map_id=map_id, 
                                  vis = 200, 
                                  vis_clear=5, 
                                  agent1_color = 'light red',
                                  agent2_color = 'blue')
        obs = self.env.reset()
        return self._obs2dict(obs)
    
    def render(self, mode='human', width=256, height=256):
        if mode == 'human':
            return self.env.render()
        elif mode == 'rgb_array':
            return self.env.render(mode='rgb_array', width=width, height=height)
   

class BaseEnvWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(BaseEnvWrapper, self).__init__(env)
        self.reward_scale = 1.0
        return


class ScaleRewardWrapper(BaseEnvWrapper):
    def __init__(self, env, **kwargs):
        super(ScaleRewardWrapper, self).__init__(env)
        self.reward_scale = kwargs['reward_scale']

    def step(self, action):
        try:
            s, r, d, info = self.env.step(action)
        except:
            print(action)
            assert 0
        scaled_reward = r * self.reward_scale
        return s, scaled_reward, d, info


import inspect
import sys


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out


class ProxyEnv(Serializable, gym.Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = gym.spaces.Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._reward_scale = d["_reward_scale"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


class JidiSubEnvWrapper(BaseEnvWrapper):
    """ Uniform the step out
    """
    def __init__(self, env):
        super(JidiSubEnvWrapper, self).__init__(env)
        self.energy = self.env.max_step

    def _uniform_obs(self, obs):
        """ sub env step:
        [{
            'agent_obs': numpy.ndarray, (40, 40),
            'id': Optional['team_0', 'team_1'],
         },
         { // another agent's observation}
        ]
        """
        joint_obs = []
        for id, agent_obs in enumerate(obs):
            agent_obs['game_mode'] = ''
            agent_obs['energy'] = self.energy
            joint_obs.append({'obs': agent_obs, 'controlled_player_index': id})
        return joint_obs
 
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self._uniform_obs(next_obs)
        return next_obs, reward, done, None, info

    def reset(self):
        self.energy = self.env.max_step
        return self._uniform_obs(self.env.reset())

    def render(self, mode='human', width=256, height=256):
        if mode == 'human':
            return self.env.render()
        elif mode == 'rgb_array':
            return self.env.render(mode='rgb_array', width=width, height=height)


class JidiFlattenEnvWrapper(BaseEnvWrapper):
    """ 
    """
    def __init__(self, env):
        super(JidiFlattenEnvWrapper, self).__init__(env)

        # Primal action_space from `self.env.get_single_action_space(0)`
        # [Box([-100.], [200.], (1,), float32), Box([-30.], [30.], (1,), float32)]
        self.action_space = spaces.Box(low=np.array([-100, -30]), high=np.array([200, 30]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(40*40 + 1, ), dtype=np.uint8)

    def _process_obs(self, pri_obs: Dict):
        """ 将原始环境返回的obs打平，并

        Args
        ----
        obs: Primal observation of the `olympics-integrated` envrionment
            [{
                'obs': {
                    'agent_obs': numpy.ndarray, (40, 40),
                    'id': Optional['team_0', 'team_1'],
                    'game_mode': Optional['NEW GAME', ''],
                    'energy': int,
                }
                'controlled_player_index': Optional[0, 1],
             },
             { // another agent's observation}
            ]
        """
        # print('\033[33m FlattenEnvWrapper football obs\033[0m', pri_obs)


        flatten_obs = pri_obs['obs']['agent_obs'].flatten()
        energy = np.array((pri_obs['obs']['energy'], ))
        # concatenate energy
        return np.concatenate((flatten_obs, energy))

    def return_obs(self, obs: List):
        """ 返回两个智能体的观测

        Return
        ------

        """
        switch_game = True if obs[0]['obs']['game_mode'] == 'NEW GAME' else False
        return [self._process_obs(obs_per_agent) for obs_per_agent in obs], switch_game
        
    def step(self, action):
        next_obs, reward, done, info_before, info = self.env.step(action)
        next_obs = self.return_obs(next_obs)
        return next_obs, reward, done, info_before, info

    def reset(self):
        obs = self.env.reset()
        return self.return_obs(obs)

    def render(self, mode='human', width=256, height=256):
        if mode == 'human':
            return self.env.render()
        elif mode == 'rgb_array':
            return self.env.render(mode='rgb_array', width=width, height=height)
