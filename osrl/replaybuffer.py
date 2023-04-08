import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """
    replay buffer for training
    
    params: dict: data with keys: 
                ['observations', 'next_observations',
                'actions', 'rewards', 'costs', 'terminals', 'timeouts']
    """
    def __init__(self, data: dict, device="cpu"):
        self.data = data
        self.device = device

    def __len__(self):
        return self.data["observations"].shape[0]
    
    def __getitem__(self, idx):
        obs  = torch.tensor(self.data["observations"][idx, :], device=self.device)
        obs2 = torch.tensor(self.data["next_observations"][idx, :], device=self.device)
        act  = torch.tensor(self.data["actions"][idx, :], device=self.device)
        rew  = torch.tensor(self.data["rewards"][idx], device=self.device)
        cost = torch.tensor(self.data["costs"][idx], device=self.device)
        done = torch.tensor((self.data["terminals"][idx] & self.data["timeouts"][idx]), device=self.device)
        return obs, obs2, act, rew, cost, done
