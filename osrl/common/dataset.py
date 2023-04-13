from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass
import random
import copy
import heapq

import numpy as np
import oapackage
from collections import Counter
from scipy.optimize import minimize
from torch.nn import functional as F  # noqa
from torch.utils.data import IterableDataset
from tqdm.auto import trange  # noqa


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def process_bc_dataset(dataset: dict, cost_limit: float, gamma: float, bc_mode: str, 
                       frontier_fn=None, frontier_range=None):

    done_mask = (dataset["terminals"] == 1) | (dataset["timeouts"] == 1)
    n_transitions = dataset["observations"].shape[0]
    idx = np.arange(n_transitions)
    done_idx = idx[done_mask]
    selected_transition = np.zeros((n_transitions,), dtype=int)
    dataset["cost_returns"] = np.zeros_like(dataset["costs"])
    
    for i in range(done_idx.shape[0]-1):
        
        start = 0 if i == 0 else done_idx[i] + 1
        end = done_idx[i] + 1 if i == 0 else done_idx[i+1] + 1
        
        cost_returns = discounted_cumsum(dataset["costs"][start:end], gamma=gamma)
        reward_returns = discounted_cumsum(dataset["rewards"][start:end], gamma=gamma)
        dataset["cost_returns"][start:end] = cost_returns[0]

        if bc_mode == "all" or bc_mode == "multi-task":
            selected_transition[start:end] = 1
        elif bc_mode == "safe":
            # safe trajectories
            if cost_returns[0] <= cost_limit:
                selected_transition[start:end] = 1
        elif bc_mode == "risky":
            # high cost trajectories
            if cost_returns[0] >= 2 * cost_limit:
                selected_transition[start:end] = 1
        elif bc_mode == "frontier":
            # trajectories that are near the Pareto frontier
            if frontier_fn(cost_returns[0]) - frontier_range <= reward_returns[0] and \
               reward_returns[0] <= frontier_fn(cost_returns[0]) + frontier_range:
                selected_transition[start:end] = 1
        elif bc_mode == "boundary":
            # trajectories that are near the cost limit
            if 0.5 * cost_limit < cost_returns[0] and cost_returns[0] <= 1.5 * cost_limit:
                selected_transition[start:end] = 1

    for k, v in dataset.items():
        dataset[k] = v[selected_transition == 1]
    if bc_mode == "multi-task":
        dataset["observations"] = np.hstack((dataset["observations"], dataset["cost_returns"].reshape(-1, 1)))

    print(f"original size = {n_transitions}, cost limit = {cost_limit}, filtered size = {np.sum(selected_transition == 1)}")


def process_sequence_dataset(dataset: dict, cost_reverse: bool = False):
    '''
    Process the d4rl format dataset to the Sequence dataset.
    '''
    traj, traj_len = [], []
    data_, episode_step = defaultdict(list), 0
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        if cost_reverse:
            data_["costs"].append(1.0 - dataset["costs"][i])
        else:
            data_["costs"].append(dataset["costs"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(episode_data["rewards"], gamma=1)
            episode_data["cost_returns"] = discounted_cumsum(episode_data["costs"], gamma=1)
            traj.append(episode_data)
            traj_len.append(episode_step)
            # reset trajectory buffer
            data_, episode_step = defaultdict(list), 0
        episode_step += 1

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


def get_nearest_point(original_data: np.ndarray,
                      sampled_data: np.ndarray,
                      max_rew_decrease: float = 1,
                      beta: float = 1):
    idxes = []
    original_idx = np.arange(0, original_data.shape[0])
    # for i in trange(sampled_data.shape[0], desc="Calculating nearest point"):
    for i in range(sampled_data.shape[0]):
        p = sampled_data[i, :]
        mask = original_data[:, 0] <= p[0]
        # mask = np.logical_and(original_data[:, 0] <= p[0], original_data[:, 0] >= p[0] - 5)
        delta = original_data[mask, :] - p
        dist = np.hypot(delta[:, 0], delta[:, 1])
        idx = np.argmin(dist)
        idxes.append(original_idx[mask][idx])
    counts = dict(Counter(idxes))

    new_idxes = []
    # dist_fun = lambda x: np.exp(max_rew_decrease / (x + beta))
    dist_fun = lambda x: 1 / (x + beta)
    for idx, num in counts.items():
        new_idxes.append(idx)
        if num > 1:
            p = original_data[idx, :]
            mask = original_data[:, 0] <= p[0]

            # the associated data should be: 1) smaller than the current cost 2) greater than certain reward
            mask = np.logical_and(original_data[:, 0] <= p[0], original_data[:, 1] >= p[1] - max_rew_decrease)
            delta = original_data[mask, :] - p
            dist = np.hypot(delta[:, 0], delta[:, 1])
            dist = dist_fun(dist)
            sample_idx = np.random.choice(dist.shape[0], size=num - 1, p=dist / np.sum(dist))
            new_idxes.extend(original_idx[mask][sample_idx.tolist()])
    return new_idxes


def augmentation(trajs: list,
                 deg: int = 3,
                 max_rew_decrease: float = 1,
                 beta: float = 1,
                 augment_percent: float = 0.3,
                 max_reward: float = 1000.0,
                 min_reward: float = 0.0):
    '''
    Calculate sample probability according to the inverse distance 
    between each trajectory(c, r) to the fitted Pareto Frontier
    '''
    rew_ret, cost_ret = [], []
    for i, traj in enumerate(trajs):
        r, c = traj["returns"][0], traj["cost_returns"][0]
        rew_ret.append(r)
        cost_ret.append(c)
    rew_ret = np.array(rew_ret, dtype=np.float64)  # type should be float64
    cost_ret = np.array(cost_ret, dtype=np.float64)

    pareto = oapackage.ParetoDoubleLong()
    for i in range(rew_ret.shape[0]):
        w = oapackage.doubleVector((-cost_ret[i], rew_ret[i]))
        pareto.addvalue(w, i)

    # print pareto number
    pareto.show(verbose=1)
    pareto_idx = list(pareto.allindices())

    cost_ret_pareto = cost_ret[pareto_idx]
    rew_ret_pareto = rew_ret[pareto_idx]
    pareto_frontier = np.poly1d(np.polyfit(cost_ret_pareto, rew_ret_pareto, deg=deg))

    sample_num = int(augment_percent * cost_ret.shape[0])
    # the augmented data should be within the cost return range of the dataset
    cost_ret_range = np.linspace(np.min(cost_ret), np.max(cost_ret), sample_num)
    pf_rew_ret = pareto_frontier(cost_ret_range)
    max_reward = max_reward * np.ones(pf_rew_ret.shape)
    min_reward = min_reward * np.ones(pf_rew_ret.shape)
    # sample the rewards that are above the pf curve and within the max_reward
    sampled_rew_ret = np.random.uniform(low=pf_rew_ret + min_reward, high=max_reward, size=sample_num)

    # associate each sampled (cost, reward) pair with a trajectory index
    original_data = np.hstack([cost_ret[:, None], rew_ret[:, None]])
    sampled_data = np.hstack([cost_ret_range[:, None], sampled_rew_ret[:, None]])
    nearest_idx = get_nearest_point(original_data, sampled_data, max_rew_decrease, beta)

    # relabel the dataset
    aug_trajs = []
    for i, target in zip(nearest_idx, sampled_data):
        target_cost_ret, target_rew_ret = target[0], target[1]
        associated_traj = copy.deepcopy(trajs[i])
        cost_ret, rew_ret = associated_traj["cost_returns"], associated_traj["returns"]
        cost_ret += target_cost_ret - cost_ret[0]
        rew_ret += target_rew_ret - rew_ret[0]
        aug_trajs.append(associated_traj)
    return nearest_idx, aug_trajs, pareto_frontier


def compute_sample_prob(dataset, pareto_frontier, beta):
    rew_ret, cost_ret = [], []
    for i, traj in enumerate(dataset):
        r, c = traj["returns"][0], traj["cost_returns"][0]
        rew_ret.append(r)
        cost_ret.append(c)
    rew_ret = np.array(rew_ret, dtype=np.float64)  # type should be float64
    cost_ret = np.array(cost_ret, dtype=np.float64)

    prob_fun = lambda x: 1 / (x + beta)
    sample_prob = []
    for i in trange(cost_ret.shape[0], desc="Calculating sample prob"):
        r, c = rew_ret[i], cost_ret[i]
        dist_fun = lambda x: (x - c)**2 + (pareto_frontier(x) - r)**2
        sol = minimize(dist_fun, x0=c, method="bfgs", tol=1e-4)
        x = np.max([0, (sol.x)[0]])
        dist = np.sqrt(dist_fun(x))
        prob = prob_fun(dist)
        sample_prob.append(prob)
    sample_prob /= np.sum(sample_prob)
    return sample_prob


def compute_cost_sample_prob(dataset, cost_transform=lambda x: 50 - x):
    sample_prob = []
    for i, traj in enumerate(dataset):
        c = cost_transform(traj["cost_returns"][0])
        sample_prob.append(c)
    sample_prob = np.array(sample_prob)
    sample_prob[sample_prob < 0] = 0
    sample_prob /= np.sum(sample_prob)
    return sample_prob


def gauss_kernel(size, std=1.0):
    size = int(size)
    x = np.linspace(-size, size, 2 * size + 1)
    g = np.exp(-(x**2 / std))
    return g


def compute_start_index_sample_prob(dataset, prob=0.4):
    sample_prob_list = []
    for i, traj in enumerate(dataset):
        n = np.sum(traj["costs"])
        l = len(traj["costs"])
        if prob * l - n <= 0:
            x = 100
        else:
            x = n * (1 - prob) / (prob * l - n)
        # in case n=0
        if x <= 0:
            x = 1
        costs = np.array(traj["costs"])
        kernel = gauss_kernel(10, 10)
        costs = np.convolve(costs, kernel)
        costs = costs[10:-10] + x
        sample_prob = costs / costs.sum()
        sample_prob_list.append(sample_prob)
    return sample_prob_list


# some utils functionalities specific for Decision Transformer
def pad_along_axis(arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def select_optimal_trajectory(trajs, rmin=0, cost_bins=60, max_num_per_bin=1):
    rew, cost = [], []
    for i, traj in enumerate(trajs):
        r, c = traj["returns"][0], traj["cost_returns"][0]
        rew.append(r)
        cost.append(c)

    xmin, xmax = min(cost), max(cost)
    xbin_step = (xmax - xmin) / cost_bins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(cost)):
        if rew[i] < rmin:
            continue
        x_bin_idx = (cost[i] - xmin) // xbin_step
        bin_hashmap[x_bin_idx].append(i)

    # start filtering
    def sort_index(idx):
        return rew[idx]

    indices = []
    for v in bin_hashmap.values():
        idx = heapq.nlargest(max_num_per_bin, v, key=sort_index)
        indices += idx

    traj2 = []
    for i in indices:
        traj2.append(trajs[i])
    return traj2


def random_augmentation(trajs: list,
                        augment_percent: float = 0.3,
                        aug_rmin: float = 0,
                        aug_rmax: float = 600,
                        aug_cmin: float = 5,
                        aug_cmax: float = 50,
                        cgap: float = 5,
                        rstd: float = 1,
                        cstd: float = 0.25):
    '''
    Calculate sample probability according to the inverse distance 
    between each trajectory(c, r) to the fitted Pareto Frontier
    '''
    rew_ret, cost_ret = [], []
    for i, traj in enumerate(trajs):
        r, c = traj["returns"][0], traj["cost_returns"][0]
        rew_ret.append(r)
        cost_ret.append(c)

    # [traj_num]
    rew_ret = np.array(rew_ret, dtype=np.float64)  # type should be float64
    cost_ret = np.array(cost_ret, dtype=np.float64)
    cmin = np.min(cost_ret)

    num = int(augment_percent * cost_ret.shape[0])
    sampled_cr = np.random.uniform(low=(aug_cmin, aug_rmin), high=(aug_cmax, aug_rmax), size=(num, 2))

    idxes = []
    original_data = np.hstack([cost_ret[:, None], rew_ret[:, None]])
    original_idx = np.arange(0, original_data.shape[0])
    # for i in trange(sampled_data.shape[0], desc="Calculating nearest point"):
    for i in range(sampled_cr.shape[0]):
        p = sampled_cr[i, :]
        boundary = max(p[0] - cgap, cmin + 1)
        mask = original_data[:, 0] <= boundary
        # mask = np.logical_and(original_data[:, 0] <= p[0], original_data[:, 0] >= p[0] - 5)
        delta = original_data[mask, :] - p
        dist = np.hypot(delta[:, 0], delta[:, 1])
        idx = np.argmin(dist)
        idxes.append(original_idx[mask][idx])

    # relabel the dataset
    aug_trajs = []
    for i, target in zip(idxes, sampled_cr):
        target_cost_ret, target_rew_ret = target[0], target[1]
        associated_traj = copy.deepcopy(trajs[i])
        cost_ret, rew_ret = associated_traj["cost_returns"], associated_traj["returns"]
        cost_ret += target_cost_ret - cost_ret[0] + np.random.normal(loc=0, scale=cstd, size=cost_ret.shape)
        rew_ret += target_rew_ret - rew_ret[0] + np.random.normal(loc=0, scale=rstd, size=rew_ret.shape)
        aug_trajs.append(associated_traj)
    return idxes, aug_trajs


class SequenceDataset(IterableDataset):

    def __init__(
        self,
        dataset: dict,
        seq_len: int = 10,
        reward_scale: float = 1.0,
        cost_scale: float = 1.0,
        deg: int = 3,
        pf_sample: bool = False,
        max_rew_decrease: float = 1.0,
        beta: float = 1.0,
        augment_percent: float = 0,
        max_reward: float = 1000.0,
        min_reward: float = 5,
        cost_reverse: bool = False,
        pf_only: bool = False,
        rmin: float = 0,
        cost_bins: int = 60,
        npb: int = 5,
        cost_sample: bool = False,
        cost_transform=lambda x: 50 - x,
        prob: float = 0.4,
        start_sampling: bool = False,
        random_aug: float = 0,
        aug_rmin: float = 0,
        aug_rmax: float = 600,
        aug_cmin: float = 5,
        aug_cmax: float = 50,
        cgap: float = 5,
        rstd: float = 1,
        cstd: float = 0.2,
    ):
        self.original_data, info = process_sequence_dataset(dataset, cost_reverse)
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.seq_len = seq_len
        self.start_sampling = start_sampling

        self.aug_data = []
        if pf_only:
            print("*" * 100)
            print("Using pareto frontier data points only!!!!!")
            print("*" * 100)
            self.dataset = select_optimal_trajectory(self.original_data, rmin, cost_bins, npb)
        elif random_aug > 0:
            self.idx, self.aug_data = random_augmentation(
                self.original_data,
                random_aug,
                aug_rmin,
                aug_rmax,
                aug_cmin,
                aug_cmax,
                cgap,
                rstd,
                cstd,
            )
        elif augment_percent > 0:
            # sampled data and the index of its "nearest" point in the dataset
            self.idx, self.aug_data, self.pareto_frontier = augmentation(self.original_data, deg, max_rew_decrease,
                                                                         beta, augment_percent, max_reward, min_reward)
        self.dataset = self.original_data + self.aug_data
        print(
            f"original data: {len(self.original_data)}, augment data: {len(self.aug_data)}, total: {len(self.dataset)}"
        )

        # self.state_mean = info["obs_mean"]
        # self.state_std = info["obs_std"]
        if cost_sample:
            self.sample_prob = compute_cost_sample_prob(self.dataset, cost_transform)
        elif pf_sample:
            self.sample_prob = compute_sample_prob(self.dataset, self.pareto_frontier, 1)
        else:
            self.sample_prob = None

        # compute every trajectories start index sampling prob:
        if start_sampling:
            self.start_idx_sample_prob = compute_start_index_sample_prob(dataset=self.dataset, prob=prob)

    def compute_pareto_return(self, cost):
        return self.pareto_frontier(cost)

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx:start_idx + self.seq_len]
        actions = traj["actions"][start_idx:start_idx + self.seq_len]
        returns = traj["returns"][start_idx:start_idx + self.seq_len]
        cost_returns = traj["cost_returns"][start_idx:start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        costs = traj["costs"][start_idx:start_idx + self.seq_len]

        episode_cost = traj["cost_returns"][0] * self.cost_scale

        # states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        cost_returns = cost_returns * self.cost_scale
        # pad up to seq_len if needed
        mask = np.hstack([np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])])
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)
            cost_returns = pad_along_axis(cost_returns, pad_to=self.seq_len)
            costs = pad_along_axis(costs, pad_to=self.seq_len)

        return states, actions, returns, cost_returns, time_steps, mask, episode_cost, costs

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            # compute start index sampling prob
            if self.start_sampling:
                start_idx = np.random.choice(self.dataset[traj_idx]["rewards"].shape[0],
                                             p=self.start_idx_sample_prob[traj_idx])
            else:
                start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)
            
            
class TransitionDataset(IterableDataset):
    
    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False):
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.sample_prob = None
        self.state_init = state_init
        self.dataset_size = self.dataset["observations"].shape[0]
        
        self.dataset["done"] = np.logical_or(self.dataset["terminals"], self.dataset["timeouts"]).astype(np.float32)
        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0

    def get_dataset_states(self):
        init_state_propotion = self.dataset["is_init"].mean()
        obs_std = self.dataset["observations"].std(0, keepdims=True)
        act_std = self.dataset["actions"].std(0, keepdims=True)
        return init_state_propotion, obs_std, act_std

    def __prepare_sample(self, idx):
        observations = self.dataset["observations"][idx, :]
        next_observations = self.dataset["next_observations"][idx, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        costs = self.dataset["costs"][idx] * self.cost_scale
        done = self.dataset["done"][idx]
        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, next_observations, actions, rewards, costs, done, is_init
        return observations, next_observations, actions, rewards, costs, done

    def __iter__(self):
        while True:
            idx = np.random.choice(self.dataset_size, p=self.sample_prob)
            yield self.__prepare_sample(idx)
