# inspiration: https://github.com/tinkoff-ai/CORL/blob/main/algorithms/dt.py

from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa

from sdt.common import TransformerBlock


class CostDecisionTransformer(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
        time_emb: bool = True,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.cost_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=3 * seq_len,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])
        # self.action_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
        #                                  nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.state_pred_head = nn.Linear(embedding_dim, self.state_dim)
        self.cost_pred_head = nn.Linear(embedding_dim, 1)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 20 / (x + 20)  # costs_to_go = 20 * 30 / (costs_to_go.detach() + 20)
            # costs_to_go = 30 - costs_to_go
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            costs_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go)

        costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb

        # [batch_size, seq_len * 3, emb_dim], (c_0 s_0, a_0, c_1, s_1, a_1, ...)
        sequence = (torch.stack([costs_emb, state_emb, act_emb],
                                dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim))
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (torch.stack([padding_mask, padding_mask, padding_mask],
                                        dim=1).permute(0, 2, 1).reshape(batch_size, 3 * seq_len))
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len * 3, embedding_dim]
        out = self.out_norm(out)
        # [batch_size, 3, seq_len, embedding_dim]
        out = out.reshape(batch_size, seq_len, 3, self.embedding_dim).permute(0, 2, 1, 3)

        # [batch_size, seq_len, embedding_dim]
        if self.add_cost_feat:
            state_feat = out[:, 1] + costs_emb
        elif self.mul_cost_feat:
            state_feat = out[:, 1] * costs_emb
        else:
            state_feat = out[:, 1]
        action_feature = out[:, 2]

        # get predictions
        action_preds = self.action_head(
            state_feat) * self.max_action  # predict next action given state, [batch_size, seq_len, action_dim]
        cost_preds = self.cost_pred_head(action_feature)  # predict next cost return given state and action
        state_preds = self.state_pred_head(action_feature)  # predict next state given state and action

        return action_preds, cost_preds, state_preds