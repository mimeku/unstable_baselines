from email.policy import default
import os
import click

from unstable_baselines.baselines.ppo.agent import PPOAgent
from unstable_baselines.baselines.sac.agent import SACAgent
from unstable_baselines.baselines.ddpg.agent import DDPGAgent

from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common.buffer import OnlineBuffer

from unstable_baselines.common.logger import Logger
from unstable_baselines.common.env_wrapper import get_env, JidiFlattenEnv
from unstable_baselines.olympic_baselines.integrated.trainer import Player, TwoAgentGame
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed


AGENT = {
    'PPO': PPOAgent,
    'SAC': SACAgent,
    'DDPG': DDPGAgent
}
BUFFER = {
    'on_policy': OnlineBuffer,
    'off_policy': ReplayBuffer
}


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.argument("config-path", type=str, required=True)
@click.option("--log-dir", type=str, default=os.path.join("logs", "integrated"))
@click.option("--gpu", type=int, default=-1)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=59)
@click.option("--info", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, seed, info, args):
    print(args)
    args = load_config(config_path, args)

    # set global seed
    set_global_seed(seed)

    # initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, env_name, prefix=info, print_to_terminal=print_log)

    # set device and logger
    set_device_and_logger(gpu, logger)

    # save args
    logger.log_str_object("parameters", log_dict=args)

    # initialize environment
    logger.log_str("Initializing Environment")
    train_env = get_env(env_name)
    eval_env = get_env(env_name)
    env_mode = args['env_mode']
    if env_mode == 'flatten':
        train_env = JidiFlattenEnv(train_env)
        eval_env = JidiFlattenEnv(eval_env)

    state_space = train_env.observation_space
    action_space = train_env.action_space
    # print(state_space, action_space)

    # initialize Player 1
    agent1_name = args['player_1']['agent_name']
    logger.log_str(f"Initializing No.1 Player {agent1_name}")
    logger.log_str(f"....initialzing agent")
    agent_1 = AGENT[agent1_name](state_space, action_space, **args['player_1']['agent'])
    logger.log_str(f"....initialzing buffer")
    buffer_1 = BUFFER[args['player_1']['type']](state_space, action_space, **args['player_1']['buffer'])
    player_1 = Player(agent_1, buffer_1, args['player_1']['type'], 0)

    # initialize Player 2
    agent2_name = args['player_2']['agent_name']
    logger.log_str(f"Initializing No.2 Player {agent2_name}")
    logger.log_str(f"....initialzing agent")
    agent_2 = AGENT[agent2_name](state_space, action_space, **args['player_2']['agent'])
    logger.log_str(f"....initialzing buffer")
    buffer_2 = BUFFER[args['player_2']['type']](state_space, action_space, **args['player_2']['buffer'])
    player_2 = Player(agent_2, buffer_2, args['player_2']['type'], 1)

    # initialize trainer
    logger.log_str("Initializing Trainer")
    trainer = TwoAgentGame(
        player_1,
        player_2,

        **args['trainer']
    )

    logger.log_str("Started training")
    # trainer.train()


if __name__ == '__main__':
    main()