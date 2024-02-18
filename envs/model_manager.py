#!/usr/bin/env python3

from building_blocks_env import BuildingBlocksEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import numpy as np

"""训练、保存、加载、评估和使用模型."""


def custom_evaluate(env: BuildingBlocksEnv, model: PPO, episodes=10):
    all_score = 0
    end_error = 0
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        score = 0
        step_cnt = 0
        while not done:
            action, _ = model.predict(obs)
            last_obs = obs.copy()
            obs, reward, done, _, info = env.step(action)
            print(last_obs, action, reward)
            score += reward
            all_score += score
            step_cnt += 1
            if step_cnt != 1:
                if done == True:
                    env.reset()
                    done = False
        end_error += np.linalg.norm(obs)
        print("Episode:{} Score:{}".format(episode, score))
    print("Average score:{}".format(all_score / episodes))
    print("Average end_error:{}".format(end_error / episodes))
    env.close()


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser("Train, save, load and evaluate the model.")
    parser.add_argument(
        "-msp",
        "--model_save_path",
        type=str,
        default="saved_models",
        help="The path to save the model.",
    )
    parser.add_argument(
        "-rsp",
        "--record_save_path",
        type=str,
        default="recorded_data",
        help="The path to save the record data.",
    )
    parser.add_argument(
        "-eid", "--experiment_id", type=int, default=0, help="The id of the experiment."
    )
    parser.add_argument(
        "-mts",
        "--max_train_steps",
        type=int,
        default=int(2e5),
        help="The total train steps of the experiment.",
    )
    parser.add_argument(
        "-mrs",
        "--max_record_steps",
        type=int,
        default=0,
        help="The total record steps of the experiment.",
    )
    parser.add_argument(
        "-olt",
        "--only_train",
        action="store_true",
        help="Whether to only train the model.",
    )
    parser.add_argument(
        "-ole",
        "--only_evaluate",
        action="store_true",
        help="Whether to only evaluate the model.",
    )
    parser.add_argument(
        "-env",
        "--env_config",
        type=str,
        default="isaac",
        help="The path to the environment configuration file or the preset name.",
    )
    parser.add_argument(
        "-mes",
        "--max_enjoy_steps",
        type=int,
        default=0,
        help="The total enjoy steps of the experiment.",
    )
    args, unknown = parser.parse_known_args()

    model_save_path = args.model_save_path + f"_{args.experiment_id}"
    record_save_path = args.record_save_path
    experiment_id = args.experiment_id
    max_train_steps = args.max_train_steps
    max_record_steps = args.max_record_steps
    only_train = args.only_train
    only_evaluate = args.only_evaluate
    env_config = args.env_config
    max_enjoy_steps = args.max_enjoy_steps
    assert max_train_steps > 0, "max_train_steps should be greater than 0."
    assert not (
        only_train and only_evaluate and max_enjoy_steps <= 0
    ), "Only one of only_train and only_evaluate should be True unless enjoy is true."

    # Create environment and Instantiate the agent
    if env_config == "isaac":
        env_config_path = "./pick_place_configs_isaac_new.json"
    elif env_config == "real":
        env_config_path = "./pick_place_configs_real.json"
    env = BuildingBlocksEnv(env_config_path)
    env.record_start(record_save_path, experiment_id, max_record_steps)

    model = None
    if not only_evaluate:
        # verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
        model = PPO("MlpPolicy", env, verbose=1)
        # Train the agent and display a progress bar
        model.learn(total_timesteps=int(max_train_steps), progress_bar=True)
        # Save the agent
        model.save(model_save_path)
        del model  # delete trained model to demonstrate loading

    if not only_train:
        # Load the trained agent
        # NOTE: if you have loading issue, you can pass `print_system_info=True`
        # to compare the system on which the model was trained vs the current one
        # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
        model = PPO.load(model_save_path, env=env)

        # Evaluate the agent
        # NOTE: If you use wrappers with your environment that modify rewards,
        #       this will be reflected here. To evaluate with original rewards,
        #       wrap environment in a "Monitor" wrapper before other wrappers.
        mean_reward, std_reward = evaluate_policy(
            model, model.get_env(), n_eval_episodes=10
        )

    if max_enjoy_steps > 0:
        # Enjoy trained agent
        model = PPO.load(model_save_path, env=env) if model is None else model
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
