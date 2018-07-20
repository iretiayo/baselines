#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines.common import tf_util as U
from baselines import logger
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
import mujoco_py
import os
import time


def train(env_id, num_timesteps, seed, model_path=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    mujoco_py.ignore_mujoco_warnings().__enter__()
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = make_robotics_env(env_id, workerseed, rank=rank)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=256, num_hid_layers=3)

    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=256,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    if model_path:
        U.save_state(model_path)
    return pi


def main():

    parser = robotics_arg_parser()
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--model-path', default=None)
    args = parser.parse_args()

    args.log_dir = os.path.join('log-files', args.env, time.strftime("%b-%d_%H:%M:%S"))
    logger.configure(dir=args.log_dir)

    if not args.play:
        # train the model
        args.model_path = os.path.join(args.log_dir, 'policy')
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(args.env, num_timesteps=1, seed=args.seed)
        U.load_state(args.model_path)
        env = make_robotics_env(args.env, seed=0)

        ob = env.reset()
        while True:
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ = env.step(action)
            env.render()
            time.sleep(0.033)
            if done:
                ob = env.reset()


if __name__ == '__main__':
    main()
