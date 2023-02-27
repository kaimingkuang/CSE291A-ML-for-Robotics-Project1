# Import required packages
import argparse
import os
import os.path as osp

import gym
import mani_skill2.envs
import numpy as np
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder

from utils import ContinuousTaskWrapper, SuccessInfoWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Config name.")
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = OmegaConf.load(f"configs/{args.cfg}.yaml")
    if args.name != "":
        cfg.trial_name = args.name

    log_dir = f"{cfg.log_dir}/{cfg.trial_name}"
    os.makedirs(log_dir, exist_ok=True)

    if "env_seed" in cfg.env:
        set_random_seed(cfg.env.seed)

    def make_env(
        env_id: str,
        max_episode_steps: int = None,
        record_dir: str = None,
    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            env = gym.make(
                env_id,
                obs_mode=cfg.env.obs_mode,
                reward_mode="dense",
                control_mode=cfg.env.act_mode
            )
            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env, max_episode_steps)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(
                    env, record_dir, info_on_video=True, render_mode="cameras"
                )
            return env

        return _init

    # create eval environment
    record_dir = osp.join(log_dir, "videos/eval")
    eval_env = SubprocVecEnv(
        [make_env(cfg.env.name, record_dir=record_dir) for _ in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(cfg.env.seed)
    eval_env.reset()

    model = eval(cfg.model_name)(
        "MlpPolicy",
        eval_env,
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        learning_rate=cfg.train.lr,
        tensorboard_log=log_dir,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        **cfg.model_kwargs
    )

    # load model
    model_path = args.model_path
    model = model.load(model_path, eval_env)

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=100,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)


if __name__ == "__main__":
    main()
