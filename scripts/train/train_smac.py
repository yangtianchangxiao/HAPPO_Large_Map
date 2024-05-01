#!/usr/bin/env python
import sys
import os
sys.path.append(r"/home/cx/happo/happo")
sys.path.append(r"/home/cx/happo")

import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
# from envs.starcraft2.StarCraft2_Env import StarCraft2Env
# from envs.starcraft2.smac_maps import get_map_params
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from envs.env_wrappers_multiprocess import MultiDummyVecEnv

from runners.separated.smac_runner import SMACRunner as Runner
from envs.env_discrete import DiscreteActionEnv
import pickle
# save_count = 13 # 1 for resnet_attention and trajectory state
               # 0 for mixer 2 for attention_resnet_120  
               # 3 for resnet_attention and 3 drones and trajectory state;
               # 11 for resnet_attention and 4 drones and trajectory state;
               # 12 for resnet_attention and 5 drones and trajectory state and no collision
               
                # 4 for happo_mixer but no curriculum learning
                # 5 for happo_mixer but erosion and obstacle
                # 6 for happo_mixer no curiculum learning but erosion and obstacle
                
                # 7 also happo_mxier no curriculum learning， allow collision
                # 8 aslo happo_mixer no curriculum learning but no collision
                # 9 for resnet_attention and use trajectories
                # 10 specific for gazebo
                
                
            
            
save_count = 1 # for sparse reward using np.sum
# save_count = 2 # for sparse reward using last sum - current sum

cuda_device = "cuda:1"
# cuda_device = "cpu"
"""Train script for SMAC."""

def make_train_env(all_args, map_set, map_num):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                # env = StarCraft2Env(all_args)
                pass
            elif all_args.env_name == "SearchGrid":
                # print("create env")
                env = DiscreteActionEnv(map_set, map_num)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return MultiDummyVecEnv([get_env_fn(0)])
    else:
        # return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

        return MultiDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")
    parser.add_argument('--lr_path', type=str, default="/home/cx/happo/configs/learning_rate"+str(save_count)+".txt") # 13用的是不带数字的
    parser.add_argument('--critic_lr_path', type=str, default="/home/cx/happo/configs/critic_learning_rate"+str(save_count) +".txt") # 同上

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True 
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.running_id
    else:
        all_args.seed=np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(cuda_device)
        print("device", device)
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")

        torch.set_num_threads(all_args.n_training_threads)


    log_dir_address = '/home/cx/happo/mappo_result/sparse_'+ str(save_count)
    content_after_last_slash = log_dir_address.split('/')[-1]

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str("mapset") + "-" + str(all_args.experiment_name) + "@" + str(
        content_after_last_slash))
    if not os.path.exists(log_dir_address):
        print("不存在该路径，正在创建")
        # print(log_dir_address)
        os.makedirs(log_dir_address)
    print(log_dir_address)
    print("Start")
    # setproctitle.setproctitle(
    #     str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
    #         all_args.user_name))

    # seed
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)

    # env
    train_path = os.path.join('/home/cx/HARL', 'assets', 'train_data.pickle')
    # test_path = os.path.join('D:', '\code', 'resize_scale_120', 'test_data.pickle')
    with open(train_path, 'rb') as tp:
        data = pickle.load(tp)
    
    # record the index of map where all the targets are found during training
    with open ("/home/cx/happo/envs/EnvDrone/classic_control/map_index.txt","w") as w:
        w.truncate(0)

    
    map_num = len(data)

    envs = make_train_env(all_args, map_set=data, map_num=map_num)
    print("checkpoint")
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    # num_agents = get_map_params(all_args.map_name)["n_agents"]
    num_agents = all_args.num_agents
    save_dir = "/home/cx/happo/mappo_model/happo_57_" +str(save_count) + "/"
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": log_dir_address,
        "save_dir": save_dir
    }
    # run experiments
    runner = Runner(config, save_dir)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    # runner.writter.close()


if __name__ == "__main__":
    
    main(sys.argv[1:])

    