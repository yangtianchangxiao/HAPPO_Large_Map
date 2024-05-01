import time
import numpy as np
from functools import reduce
import torch
from runners.separated.base_runner import Runner
from tensorboardX import SummaryWriter

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config, log_dir_address):
        super(SMACRunner, self).__init__(config, log_dir_address)
        self.log_dir_address = log_dir_address
        self.step_counter = 0

    def run(self):
        print("start run")

        with open ("/home/cx/happo/envs/EnvDrone/classic_control/happo_sparse_reward.txt","w") as f:
        # with open ("/home/cx/happo/envs/EnvDrone/classic_control/happo_sparse_reward_2.txt","w") as f:
            pass
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        print("episodes are", episodes)
        last_average_reward = -1000
        best_episode = 0
        # 记录环境步数
        env_run_time = 5 # 不加载模型时
        # env_run_time = 105 # 加载模型时
        target_find_list = []
        with SummaryWriter(log_dir=self.log_dir_address, comment='Reward per episode') as w:
            last_episode = 0
            indices = np.arange(self.num_agents)
            not_raise_time = 0
            for episode in range(episodes):
                self.reset_count = 0
                self.reset_count2 = 0
                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)
                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    # Obser reward and next obs
                    # action shape is (16, 2, 1) action 是正常的
                    obs, rewards, dones, infos, share_obs, rescue_masks, available_actions = self.envs.step(actions)

                    data = obs, share_obs, rewards, dones, infos, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic, rescue_masks, available_actions
                    # print("shape of share obs", share_obs.shape)
                    # insert data into buffer
                    self.insert(data)
                    # print()
                    # Extract the first column
                    first_column = dones[:, 0]
                    second_column = dones[:, 1]
                    # Count the number of True values in the first column
                    count_true = np.count_nonzero(first_column)
                    count_true2 = np.count_nonzero(second_column)
                    self.reset_count = self.reset_count + count_true
                    self.reset_count2 = self.reset_count2 + count_true2
                    for i, row in enumerate(dones):
                        if True in row:
                            if infos[i] >= 0:
                                target_find_list.append(infos[i])
                # compute return and update network
                self.compute()
                # print("start training")
                train_infos = self.train()

                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                average_total_reward, average_reward_log = self.average_rewards()
                # for i in range(len(average_reward_log)):
                #     w.add_scalar('average reward', average_reward_log[i], i+last_episode)

                # save model
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    # print("average reward", average_total_reward)
                    # print("last_average reawrd", last_average_reward)
                    if average_total_reward > last_average_reward:
                        print("save the model in episode", episode, "env.run_time is", env_run_time)
                        best_episode = episode
                        self.save()
                        last_average_reward = average_total_reward
                        not_raise_time = 0
                        print("self.env run time", env_run_time)
                    else:
                        print("Not better than episode", best_episode, ". And the the highest reward is",
                              last_average_reward, 'env.run_time is', env_run_time)
                        not_raise_time = not_raise_time + 1
                        self.save_copy()
                        print("self.env run time", env_run_time)
                       
                        # 课程学习部分：当连续100次没有提升时，增加环境难度
                        if not_raise_time > 35:
                            self.envs.raise_difficulty()
                            env_run_time += 5
                            # env_run_time = 300 if env_run_time > 300 else env_run_time
                            not_raise_time = 0
                            with open ("/home/cx/happo/envs/EnvDrone/classic_control/happo_sparse_reward.txt","a") as f:
                            # with open ("/home/cx/happo/envs/EnvDrone/classic_control/happo_sparse_reward_2.txt","a") as f:
                                    f.write(str(last_average_reward)+"\n")
                            last_average_reward = -10
                            
                        # if env_run_time > 100: # 不加载模型时，当环境最大步数达到200时，停止训练
                        if env_run_time > 250: # 加载模型时，当环境最大步数达到200时，停止训练
                            break
                        


                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}. Reward is{}. Reset time1 is{}, reset time2 is {}\n"
                            .format(self.all_args.map_name,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    total_num_steps,
                                    self.num_env_steps,
                                    int(total_num_steps / (end - start)),
                                    average_total_reward,
                                    self.reset_count,
                                    self.reset_count2))
                    self.reset_count = 0
                    self.reset_count2 = 0
                    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("time is", now)

                    # print("average episode extrinsic reward is {}".format(train_infos["average_episode_rewards"]))
                    # print("average episode intrinsic reward is {}".format(average_total_reward - train_infos["average_episode_rewards"]))
                    print("average episode whole reward is {}".format(average_total_reward))
                    # modified

                    for agent_id in range(self.num_agents):
                        train_infos[agent_id]['dead_ratio'] = 1 - self.buffer[agent_id].active_masks.sum() /(self.num_agents* reduce(lambda x, y: x*y, list(self.buffer[agent_id].active_masks.shape)))

                # self.log_train(train_infos, total_num_steps)
                # eval
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)

                # 写入数据
                # for i in range(len(target_find_list)):
                #     w.add_scalar('Targets found during training', target_find_list[i], i+last_episode)
                last_episode = last_episode + len(target_find_list)
                target_find_list = []
                print("已经保存",self.log_dir_address)
        w.close()
        print("self.    ", self.log_dir_address)

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            # print("share_obs shape is", share_obs.shape)
            # print("self.buffer[agent_id].share_obs[0] shape is ", self.buffer[agent_id].share_obs[0].shape)
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            # print("obs shape", obs.shape)
            self.buffer[agent_id].obs[0] = obs[:,agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]
        soft_prob_collector=[]
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            # print("self.buffer[agent_id].share_obs shape", self.buffer[0].share_obs.shape)
            value, action, action_log_prob, rnn_state, rnn_state_critic, soft_prob \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step],)
                                                # available_actions=self.buffer[agent_id].available_actions[step],)
            # print("share_obs[step].shape", self.buffer[agent_id].share_obs[step].shape)
            # print("obs[step] shape", self.buffer[agent_id].obs[step].shape)
            #
            # print("self.buffer[agent_id].masks[step]",self.buffer[agent_id].masks[step].shape)
            # print("self.buffer[agent_id].rnn_states[step]", self.buffer[agent_id].rnn_states[step].shape)
            # print("self.buffer[agent_id].rnn_states_critic[step]", self.buffer[agent_id].rnn_states_critic[step].shape)
            # print("actions", action)
            # print("")
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            soft_prob_collector.append(_t2n(soft_prob))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        # print("action_collector are",action_collector)
        # print("!!!!!!!!!!!!!!! shape of values", values.shape)
        actions = np.array(action_collector).transpose(1, 0, 2)
        # print("!!!!!!!!!!!!!!! shape of actions", actions.shape)
        # print("actions are", action_collector)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, rescue_mask, available_actions = data
        # print("actions is", actions)
        dones_env = np.array(dones)
        # dones_env = np.all(dones, axis=1)

        # rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # print("dones env .shape", dones_env.shape)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs
        # print("Insert share obs shape", share_obs.shape)
        # print("obs shape", obs.shape)
        # print("num_agents", self.num_agents)
        for agent_id in range(self.num_agents):
            # 打印每个数组的shape
            # rescue_mask = rescue_mask[:, np.newaxis]
            # print(f"share_obs.shape: {share_obs.shape}")
            # print(f"obs[:,{agent_id}].shape: {obs[:,agent_id].shape}")
            # print(f"rnn_states[:,{agent_id}].shape: {rnn_states[:,agent_id].shape}")
            # print(f"rnn_states_critic[:,{agent_id}].shape: {rnn_states_critic[:,agent_id].shape}")
            # print(f"actions[:,{agent_id}].shape: {actions[:,agent_id].shape}")
            # print(f"action_log_probs[:,{agent_id}].shape: {action_log_probs[:,agent_id].shape}")
            # print(f"values[:,{agent_id}].shape: {values[:,agent_id].shape}")
            # print(f"rewards[:,{agent_id}].shape: {rewards[:,agent_id].shape}")
            # print(f"masks[:,{agent_id}].shape: {masks[:,agent_id].shape}")
            # print(f"rescue_mask[:,{agent_id}].shape: {rescue_mask[:,agent_id].shape}")

            self.buffer[agent_id].insert(share_obs[:], obs[:,agent_id], rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id], actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id], rescue_mask[:,agent_id])

                    # available_actions=available_actions[:,agent_id])

    def average_rewards(self):
        total_rewards = 0

        for agent_id in range(self.num_agents):
            total_rewards += sum(self.buffer[agent_id].rewards)
        total_rewards = np.sum(total_rewards)
        
        average_reward = total_rewards / self.num_agents/self.n_rollout_threads/self.episode_length
        
        # 记录环境的平均奖励
        mean_rewards_per_agent = [np.mean(self.buffer[agent_id].rewards, axis=1) for agent_id in range(self.num_agents)]
        # 把每个 agent 对应位置的奖励加起来除以 agent 的数量
        average_rewards = np.mean(mean_rewards_per_agent, axis=0)
        # 使用 self.writer 记录每个时间步的平均奖励
        average_rewards = np.array(average_rewards)
        print("return average reward", np.mean(average_rewards))
        return np.mean(average_rewards),average_rewards.flatten()
    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector=[]
            eval_rnn_states_collector=[]
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                            eval_rnn_states[:,agent_id],
                                            eval_masks[:,agent_id],
                                            eval_available_actions[:,agent_id],
                                            deterministic=True)
                eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1,0,2)

            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
