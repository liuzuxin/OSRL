import copy
import heapq
import random
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import oapackage
except ImportError:
    print("OApackage is not installed, can not use CDT.")
from scipy.optimize import minimize
from torch.nn import functional as F  # noqa
from torch.utils.data import IterableDataset
from tqdm.auto import trange  # noqa


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def process_bc_dataset(dataset: dict, cost_limit: float, gamma: float, bc_mode: str):
    """
    Processes a givne dataset for behavior cloning and its variants.

    Args:
        dataset (dict): A dictionary containing the dataset to be processed.
        cost_limit (float): The maximum cost allowed for the dataset.
        gamma (float): The discount factor used to compute the returns.
        bc_mode (str): The behavior cloning mode to use. Can be one of:
            - "all": All trajectories are used for behavior cloning.
            - "multi-task": All trajectories are used for behavior cloning, and the cost is appended as a feature.
            - "safe": Only trajectories with cost below the cost limit are used for behavior cloning.
            - "risky": Only trajectories with cost above twice the cost limit are used for behavior cloning.
            - "frontier": Only trajectories near the Pareto frontier are used for behavior cloning.
            - "boundary": Only trajectories near the cost limit are used for behavior cloning.
        frontier_fn (function, optional): A function used to compute the frontier. 
                                          Required if bc_mode is "frontier".
        frontier_range (float, optional): The range around the frontier to use for selecting trajectories. 
                                           Required if bc_mode is "frontier".
    
    Returns:
        dict: A dictionary containing the processed dataset.

    """

    # get the indices of the transitions after terminal states or timeouts
    done_idx = np.where((dataset["terminals"] == 1) | (dataset["timeouts"] == 1))[0]

    n_transitions = dataset["observations"].shape[0]
    dataset["cost_returns"] = np.zeros_like(dataset["costs"])
    dataset["rew_returns"] = np.zeros_like(dataset["rewards"])
    cost_ret, rew_ret = [], []
    pareto_frontier, pf_mask = None, None

    # compute episode returns
    for i in range(done_idx.shape[0]):
        start = 0 if i == 0 else done_idx[i - 1] + 1
        end = done_idx[i] + 1
        # compute the cost and reward returns for the segment
        cost_returns = discounted_cumsum(dataset["costs"][start:end], gamma=gamma)
        reward_returns = discounted_cumsum(dataset["rewards"][start:end], gamma=gamma)
        dataset["cost_returns"][start:end] = cost_returns[0]
        dataset["rew_returns"][start:end] = reward_returns[0]
        cost_ret.append(cost_returns[0])
        rew_ret.append(reward_returns[0])

    # compute Pareto Frontier
    if bc_mode == "frontier":
        cost_ret = np.array(cost_ret, dtype=np.float64)
        rew_ret = np.array(rew_ret, dtype=np.float64)
        rmax, rmin = np.max(rew_ret), np.min(rew_ret)

        pareto = oapackage.ParetoDoubleLong()
        for i in range(rew_ret.shape[0]):
            w = oapackage.doubleVector((-cost_ret[i], rew_ret[i]))
            pareto.addvalue(w, i)
        pareto.show(verbose=1)
        pareto_idx = list(pareto.allindices())
        cost_ret_pareto = cost_ret[pareto_idx]
        rew_ret_pareto = rew_ret[pareto_idx]

        for deg in [0, 1, 2]:
            pareto_frontier = np.poly1d(
                np.polyfit(cost_ret_pareto, rew_ret_pareto, deg=deg))
            pf_rew_ret = pareto_frontier(cost_ret_pareto)
            ss_total = np.sum((rew_ret_pareto - np.mean(rew_ret_pareto))**2)
            ss_residual = np.sum((rew_ret_pareto - pf_rew_ret)**2)
            r_squared = 1 - (ss_residual / ss_total)
            if r_squared >= 0.9:
                break

        pf_rew_ret = pareto_frontier(dataset["cost_returns"])
        pf_mask = np.logical_and(
            pf_rew_ret - (rmax - rmin) / 5 <= dataset["rew_returns"],
            dataset["rew_returns"] <= pf_rew_ret + (rmax - rmin) / 5)

    # select the transitions for behavior cloning based on the mode
    selected_transition = np.zeros((n_transitions, ), dtype=int)
    if bc_mode == "all" or bc_mode == "multi-task":
        selected_transition = np.ones((n_transitions, ), dtype=int)
    elif bc_mode == "safe":
        # safe trajectories
        selected_transition[dataset["cost_returns"] <= cost_limit] = 1
    elif bc_mode == "risky":
        # high cost trajectories
        selected_transition[dataset["cost_returns"] >= 2 * cost_limit] = 1
    elif bc_mode == "boundary":
        # trajectories that are near the cost limit
        mask = np.logical_and(0.5 * cost_limit < dataset["cost_returns"],
                              dataset["cost_returns"] <= 1.5 * cost_limit)
        selected_transition[mask] = 1
    elif bc_mode == "frontier":
        selected_transition[pf_mask] = 1
    else:
        raise NotImplementedError

    for k, v in dataset.items():
        dataset[k] = v[selected_transition == 1]
    if bc_mode == "multi-task":
        dataset["observations"] = np.hstack(
            (dataset["observations"], dataset["cost_returns"].reshape(-1, 1)))

    print(
        f"original size = {n_transitions}, cost limit = {cost_limit}, filtered size = {np.sum(selected_transition == 1)}"
    )


def process_sequence_dataset(dataset: dict, cost_reverse: bool = False):
    '''
    Processe a given dataset into a list of trajectories, each containing information about 
    the observations, actions, rewards, costs, returns, and cost returns for a single episode.
    
    Args:

        dataset (dict): A dictionary representing the dataset, 
                        with keys "observations", "actions", "rewards", "costs", "terminals", and "timeouts", 
                        each containing numpy arrays of corresponding data.
        cost_reverse (bool): An optional boolean parameter that indicates whether the cost should be reversed.
        
    Returns:
        traj (list): A list of dictionaries, each representing a trajectory.
        info (dict): A dictionary containing additional information about the trajectories

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
            episode_data["cost_returns"] = discounted_cumsum(episode_data["costs"],
                                                             gamma=1)
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
    """
    Given two arrays of data, finds the indices of the original data that are closest
    to each sample in the sampled data, and returns a list of those indices.
    
    Args:
        original_data: A 2D numpy array of the original data.
        sampled_data: A 2D numpy array of the sampled data.
        max_rew_decrease: A float representing the maximum reward decrease allowed.
        beta: A float used in calculating the distance between points.
    
    Returns:
        A list of integers representing the indices of the original data that are closest
        to each sample in the sampled data.
    """

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
    dist_fun = lambda x: 1 / (x + beta)
    for idx, num in counts.items():
        new_idxes.append(idx)
        if num > 1:
            p = original_data[idx, :]
            mask = original_data[:, 0] <= p[0]

            # the associated data should be: 1) smaller than the current cost 2) greater than certain reward
            mask = np.logical_and(original_data[:, 0] <= p[0],
                                  original_data[:, 1] >= p[1] - max_rew_decrease)
            delta = original_data[mask, :] - p
            dist = np.hypot(delta[:, 0], delta[:, 1])
            dist = dist_fun(dist)
            sample_idx = np.random.choice(dist.shape[0],
                                          size=num - 1,
                                          p=dist / np.sum(dist))
            new_idxes.extend(original_idx[mask][sample_idx.tolist()])
    return new_idxes


def grid_filter(x,
                y,
                xmin=-np.inf,
                xmax=np.inf,
                ymin=-np.inf,
                ymax=np.inf,
                xbins=10,
                ybins=10,
                max_num_per_bin=10,
                min_num_per_bin=1):
    xmin, xmax = max(min(x), xmin), min(max(x), xmax)
    ymin, ymax = max(min(y), ymin), min(max(y), ymax)
    xbin_step = (xmax - xmin) / xbins
    ybin_step = (ymax - ymin) / ybins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(x)):
        if x[i] < xmin or x[i] > xmax or y[i] < ymin or y[i] > ymax:
            continue
        x_bin_idx = (x[i] - xmin) // xbin_step
        y_bin_idx = (y[i] - ymin) // ybin_step
        bin_hashmap[(x_bin_idx, y_bin_idx)].append(i)
    # start filtering
    indices = []
    for v in bin_hashmap.values():
        if len(v) > max_num_per_bin:
            # random sample max_num_per_bin indices
            indices += random.sample(v, max_num_per_bin)
        elif len(v) <= min_num_per_bin:
            continue
        else:
            indices += v
    return indices


def filter_trajectory(cost,
                      rew,
                      traj,
                      cost_min=-np.inf,
                      cost_max=np.inf,
                      rew_min=-np.inf,
                      rew_max=np.inf,
                      cost_bins=60,
                      rew_bins=50,
                      max_num_per_bin=10,
                      min_num_per_bin=1):
    indices = grid_filter(cost,
                          rew,
                          cost_min,
                          cost_max,
                          rew_min,
                          rew_max,
                          xbins=cost_bins,
                          ybins=rew_bins,
                          max_num_per_bin=max_num_per_bin,
                          min_num_per_bin=min_num_per_bin)
    cost2, rew2, traj2 = [], [], []
    for i in indices:
        cost2.append(cost[i])
        rew2.append(rew[i])
        traj2.append(traj[i])
    return cost2, rew2, traj2, indices


def augmentation(trajs: list,
                 deg: int = 3,
                 max_rew_decrease: float = 1,
                 beta: float = 1,
                 augment_percent: float = 0.3,
                 max_reward: float = 1000.0,
                 min_reward: float = 0.0):
    """
    Applies data augmentation to a list of trajectories, 
    returning the augmented trajectories along with their indices 
    and the Pareto frontier of the original data.

    Args:
        trajs: A list of dictionaries representing the original trajectories.
        deg: The degree of the polynomial used to fit the Pareto frontier.
        max_rew_decrease: The maximum amount by which the reward of an augmented trajectory can decrease compared to the original.
        beta: The scaling factor used to weigh the distance between cost and reward when finding nearest neighbors.
        augment_percent: The percentage of original trajectories to use for augmentation.
        max_reward: The maximum reward value for augmented trajectories.
        min_reward: The minimum reward value for augmented trajectories.

    Returns:
        nearest_idx: A list of indices of the original trajectories that are nearest to each augmented trajectory.
        aug_trajs: A list of dictionaries representing the augmented trajectories.
        pareto_frontier: A polynomial function representing the Pareto frontier of the original data.
    """

    rew_ret, cost_ret = [], []
    for i, traj in enumerate(trajs):
        r, c = traj["returns"][0], traj["cost_returns"][0]
        rew_ret.append(r)
        cost_ret.append(c)
    rew_ret = np.array(rew_ret, dtype=np.float64)
    cost_ret = np.array(cost_ret, dtype=np.float64)

    # grid filer to filter outliers
    cmin, cmax = np.min(cost_ret), np.max(cost_ret)
    rmin, rmax = np.min(rew_ret), np.max(rew_ret)
    cbins, rbins = 10, 50
    max_npb, min_npb = 10, 2
    cost_ret, rew_ret, trajs, indices = filter_trajectory(cost_ret,
                                                          rew_ret,
                                                          trajs,
                                                          cost_min=cmin,
                                                          cost_max=cmax,
                                                          rew_min=rmin,
                                                          rew_max=rmax,
                                                          cost_bins=cbins,
                                                          rew_bins=rbins,
                                                          max_num_per_bin=max_npb,
                                                          min_num_per_bin=min_npb)
    print(f"after filter {len(trajs)}")
    rew_ret = np.array(rew_ret, dtype=np.float64)
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
    sampled_rew_ret = np.random.uniform(low=pf_rew_ret + min_reward,
                                        high=max_reward,
                                        size=sample_num)

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
    return nearest_idx, aug_trajs, pareto_frontier, indices


def compute_sample_prob(dataset, pareto_frontier, beta):
    """
    Computes the probability of sampling each trajectory in a given dataset.

    Args:
        dataset (list): A list of dictionaries containing the trajectories 
                        to compute the sample probabilities for.
        pareto_frontier (callable): A function that takes in a cost value and 
                                    returns the corresponding maximum reward value 
                                    on the Pareto frontier.
        beta (float): A hyperparameter that controls the shape of the probability distribution.

    Returns:
        np.ndarray: A 1D numpy array of the same length as the dataset, 
                    containing the probability of sampling each trajectory.

    """

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
    """
    Computes the sample probabilities for a given dataset based on its costs.

    Args:
        dataset (list): A list of trajectories, where each trajectory is a dictionary containing
                        a "cost_returns" key with a list of cost values.
        cost_transform (function): A function that transforms cost values.

    Returns:
        np.ndarray: A 1D numpy array of sample probabilities, normalized to sum to 1.
    """

    sample_prob = []
    for i, traj in enumerate(dataset):
        c = cost_transform(traj["cost_returns"][0])
        sample_prob.append(c)
    sample_prob = np.array(sample_prob)
    sample_prob[sample_prob < 0] = 0
    sample_prob /= np.sum(sample_prob)
    return sample_prob


def gauss_kernel(size, std=1.0):
    """
    Computes a 1D Gaussian kernel with the given size and standard deviation.
    """
    size = int(size)
    x = np.linspace(-size, size, 2 * size + 1)
    g = np.exp(-(x**2 / std))
    return g


def compute_start_index_sample_prob(dataset, prob=0.4):
    """
    computes every trajectories start index sampling probability
    """

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
def pad_along_axis(arr: np.ndarray,
                   pad_to: int,
                   axis: int = 0,
                   fill_value: float = 0.0) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def select_optimal_trajectory(trajs, rmin=0, cost_bins=60, max_num_per_bin=1):
    """
    Selects the optimal trajectories from a list of trajectories based on their returns and costs.

    Args:
        trajs (list): A list of dictionaries, where each dictionary represents a trajectory and contains
                      the keys "returns" and "cost_returns".
        rmin (float): The minimum return that a trajectory must have in order to be considered optimal.
        cost_bins (int): The number of bins to divide the cost range into.
        max_num_per_bin (int): The maximum number of trajectories to select from each cost bin.

    Returns:
        list: A list of dictionaries representing the optimal trajectories.
    """

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
    """
    Augments a list of trajectories with random noise.
    
    Args:
        trajs (list): A list of dictionaries, where each dictionary represents a trajectory
            and contains "returns" and "cost_returns" keys that hold the returns and cost returns
            for each time step of the trajectory.
        augment_percent (float, optional): The percentage of trajectories to augment.
        aug_rmin (float, optional): The minimum value for the augmented returns.
        aug_rmax (float, optional): The maximum value for the augmented returns.
        aug_cmin (float, optional): The minimum value for the augmented cost returns.
        aug_cmax (float, optional): The maximum value for the augmented cost returns.
        cgap (float, optional): The minimum distance between the augmented cost returns
        rstd (float, optional): The standard deviation of the noise to add to the returns.
        cstd (float, optional): The standard deviation of the noise to add to the cost returns.

    Returns:
        Tuple[List[int], List[Dict]]: A tuple containing two lists. The first list contains
            the indices of the original trajectories that were augmented. The second list contains
            the augmented trajectories, represented as dictionaries with "returns" and "cost_returns"
            keys.
    """

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
    sampled_cr = np.random.uniform(low=(aug_cmin, aug_rmin),
                                   high=(aug_cmax, aug_rmax),
                                   size=(num, 2))

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
        cost_ret += target_cost_ret - cost_ret[0] + np.random.normal(
            loc=0, scale=cstd, size=cost_ret.shape)
        rew_ret += target_rew_ret - rew_ret[0] + np.random.normal(
            loc=0, scale=rstd, size=rew_ret.shape)
        aug_trajs.append(associated_traj)
    return idxes, aug_trajs


class SequenceDataset(IterableDataset):
    """
    A dataset of sequential data.

    Args:
        dataset (dict): Input dataset, containing trajectory IDs and sequences of observations.
        seq_len (int): Length of sequence to use for training.
        reward_scale (float): Scaling factor for reward values.
        cost_scale (float): Scaling factor for cost values.
        deg (int): Degree of polynomial used for Pareto frontier augmentation.
        pf_sample (bool): Whether to sample data from the Pareto frontier.
        max_rew_decrease (float): Maximum reward decrease for Pareto frontier augmentation.
        beta (float): Parameter used for cost-based augmentation.
        augment_percent (float): Percentage of data to augment.
        max_reward (float): Maximum reward value for augmentation.
        min_reward (float): Minimum reward value for augmentation.
        cost_reverse (bool): Whether to reverse the cost values.
        pf_only (bool): Whether to use only Pareto frontier data points.
        rmin (float): Minimum reward value for random augmentation.
        cost_bins (int): Number of cost bins for random augmentation.
        npb (int): Number of data points to select from each cost bin for random augmentation.
        cost_sample (bool): Whether to sample data based on cost.
        cost_transform (callable): Function used to transform cost values.
        prob (float): Probability of sampling from each trajectory start index.
        start_sampling (bool): Whether to sample from each trajectory start index.
        random_aug (float): Percentage of data to augment randomly.
        aug_rmin (float): Minimum reward value for random augmentation.
        aug_rmax (float): Maximum reward value for random augmentation.
        aug_cmin (float): Minimum cost value for random augmentation.
        aug_cmax (float): Maximum cost value for random augmentation.
        cgap (float): Cost gap for random augmentation.
        rstd (float): Standard deviation of reward values for random augmentation.
        cstd (float): Standard deviation of cost values for random augmentation.
    """

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
            self.dataset = select_optimal_trajectory(self.original_data, rmin, cost_bins,
                                                     npb)
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
            self.idx, self.aug_data, self.pareto_frontier, self.indices = augmentation(
                self.original_data, deg, max_rew_decrease, beta, augment_percent,
                max_reward, min_reward)
        self.dataset = self.original_data + self.aug_data
        print(
            f"original data: {len(self.original_data)}, augment data: {len(self.aug_data)}, total: {len(self.dataset)}"
        )

        if cost_sample:
            self.sample_prob = compute_cost_sample_prob(self.dataset, cost_transform)
        elif pf_sample:
            self.sample_prob = compute_sample_prob(self.dataset, self.pareto_frontier, 1)
        else:
            self.sample_prob = None

        # compute every trajectories start index sampling prob:
        if start_sampling:
            self.start_idx_sample_prob = compute_start_index_sample_prob(
                dataset=self.dataset, prob=prob)

    def compute_pareto_return(self, cost):
        return self.pareto_frontier(cost)

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
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
        mask = np.hstack(
            [np.ones(states.shape[0]),
             np.zeros(self.seq_len - states.shape[0])])
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
                start_idx = random.randint(
                    0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


class TransitionDataset(IterableDataset):
    """
    A dataset of transitions (state, action, reward, next state) used for training RL agents.
    
    Args:
        dataset (dict): A dictionary of NumPy arrays containing the observations, actions, rewards, etc.
        reward_scale (float): The scale factor for the rewards.
        cost_scale (float): The scale factor for the costs.
        state_init (bool): If True, the dataset will include an "is_init" flag indicating if a transition
            corresponds to the initial state of an episode.

    """

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

        self.dataset["done"] = np.logical_or(self.dataset["terminals"],
                                             self.dataset["timeouts"]).astype(np.float32)
        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0

    def get_dataset_states(self):
        """
        Returns the proportion of initial states in the dataset, 
        as well as the standard deviations of the observation and action spaces.
        """
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
