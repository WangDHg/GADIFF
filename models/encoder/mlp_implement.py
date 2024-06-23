from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity

from torch_geometric.nn.dense.linear import Linear


class MLP(torch.nn.Module):
    
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.,
        act: Union[str, Callable, None] = "relu",
        batch_norm: bool = True,
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        batch_norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        relu_first: bool = False,
        is_res: bool = False,
        # norm: Union[str, Callable, None] = "batch_norm",
        # norm_kwargs: Optional[Dict[str, Any]] = None,
        # plain_last: bool = True
    ):
        super().__init__()

        from class_resolver.contrib.torch import activation_resolver

        act_first = act_first or relu_first  # Backward compatibility.
        batch_norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        
        self.is_res = is_res
        
        self.channel_list = channel_list
        num_layers = len(channel_list) - 1
        # self.num_multi = torch.pow(torch.tensor(2), torch.floor(torch.log2(torch.tensor(num_layers))/2))
        # # self.num_multi = torch.pow(torch.tensor(2), torch.floor(torch.log2(torch.tensor(num_layers-2))/2))
        # # self.num_multi = 4

        self.dropout = dropout
        self.act = activation_resolver.make(act, act_kwargs)
        self.act_first = act_first

        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for i, (lin, norm) in enumerate(zip(self.lins[1:], self.norms)):
            # # if i < len(self.lins) - 1 and (i+1)%2 == 0:
            # if i < len(self.lins) - 1 and (i+1)%self.num_multi == 0:
            #     if self.act_first:
            #         x = self.act(x)
            # if self.act_first:
            #     if i < len(self.lins) - 1 and i%self.num_multi == 0:
            #         x = self.act(x)
            if self.act_first:
                x = self.act(x)
            # x = norm(x)
            # # if i < len(self.lins) - 1 and (i+1)%2 == 0:
            # if i < len(self.lins) - 1 and (i+1)%self.num_multi == 0:
            #     if not self.act_first:
            #         x = self.act(x)
            # if not self.act_first:
            #     if i < len(self.lins) - 1 and i%self.num_multi == 0:
            #         x = self.act(x)
            if not self.act_first:
                x = self.act(x)
            # x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
            # if self.is_res and i != len(self.lins[1:])-1:
            #     x = lin.forward(x)+x
            # else:
            #     x = lin.forward(x)
        return x


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'