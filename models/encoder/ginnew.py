# coding=utf-8
from typing import Callable, Optional, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from .schnet import SchNetConv

from ..common import MeanReadout, SumReadout, MultiLayerPerceptron

from torch_geometric.nn.dense.linear import Linear

from torch_geometric.nn.inits import reset

from .mlp_implement import MLP

from models.network.util.Block_process import Multi_Attn, Block, Weight_Multi_Attn

import math

class GINEConv(MessagePassing):
   
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 activation="relu", edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        
        if isinstance(activation, str):
            if activation.lower() == "leakyrelu":
                self.activation = getattr(F, "leaky_relu")
            else:
                self.activation = getattr(F, activation.lower())
            # self.activation = getattr(F, activation.lower())
        else:
            self.activation = None
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        if edge_dim is not None:
            # if hasattr(self.nn, 'in_features'):
            #     in_channels = self.nn.in_features
            # else:
            #     in_channels = self.nn.in_channels
            # self.lin = Linear(edge_dim, in_channels)
            self.lin = Linear(edge_dim, edge_dim)
        else:
            self.lin = None
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, node_attr = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        
        if node_attr is not None:
            out = torch.cat([out, node_attr], dim=1)

        return self.nn(out)


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class NodeConv(torch.nn.Module):
    def __init__(self, hidden_dim, num_convs = 3, activation='relu', short_cut=True, attn_config=None):
        super().__init__()
        self.num_convs = num_convs
        self.hidden_dim = hidden_dim
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.bond_convs = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], \
                                act=activation, is_res=True), edge_dim=hidden_dim, activation=activation)   # 8 层
        self.angle_convs = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], \
                                act=activation, is_res=True), edge_dim=hidden_dim, activation=activation)   # 8 层
        self.torsion_convs = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], \
                                act=activation, is_res=True), edge_dim=hidden_dim, activation=activation)   # 8 层
        self.radius_convs = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], \
                                act=activation, is_res=True), edge_dim=hidden_dim, activation=activation)   # 8 层

        if attn_config != None:
            self.node_edgetype_attn = Multi_Attn(
                input_dim=attn_config.input_dim, 
                output_dim=attn_config.output_dim, 
                layer_num=attn_config.layer_num, 
                hidden_dim=hidden_dim, 
                head_num=attn_config.head_num,
                drop_rate=attn_config.drop_rate,
                )
        else:
            self.node_edgetype_attn = Multi_Attn(
                input_dim=4, 
                output_dim=4, 
                layer_num=1, 
                hidden_dim=hidden_dim, 
                head_num=32,
                drop_rate=0.001,
                )
    
    def forward(self, node_attr, edge_index, edge_attr, edge_length, edge_mask, conv_idx):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """

        conv_input = node_attr # (num_node, hidden) 节点特征

        bond_hidden = self.bond_convs(
            x = conv_input[:,0,:],
            edge_index = edge_index[:, edge_mask[0]],
            edge_attr = edge_attr[edge_mask[0]]
        )
        angle_hidden = self.angle_convs(
            x = conv_input[:,1,:], 
            edge_index = edge_index[:, edge_mask[1]], 
            edge_attr = edge_attr[edge_mask[1]]
        )
        torsion_hidden = self.torsion_convs(
            x = conv_input[:,2,:], 
            edge_index = edge_index[:, edge_mask[2]], 
            edge_attr = edge_attr[edge_mask[2]]
        )
        radius_hidden = self.radius_convs(
            x = conv_input[:,3,:], 
            edge_index = edge_index[:, edge_mask[3]], 
            edge_attr = edge_attr[edge_mask[3]]
        )
        
        if conv_idx < self.num_convs - 1 and self.activation is not None:
            bond_hidden = self.activation(bond_hidden)
            angle_hidden = self.activation(angle_hidden)
            torsion_hidden = self.activation(torsion_hidden)
            radius_hidden = self.activation(radius_hidden)
        
        assert bond_hidden.shape == conv_input[:,0,:].shape                
        if self.short_cut and bond_hidden.shape == conv_input[:,0,:].shape:            # 每层多了个残差
            bond_hidden = bond_hidden + conv_input[:,0,:]
        
        assert angle_hidden.shape == conv_input[:,1,:].shape                
        if self.short_cut and angle_hidden.shape == conv_input[:,1,:].shape:            # 每层多了个残差
            angle_hidden = angle_hidden + conv_input[:,1,:]
        
        assert torsion_hidden.shape == conv_input[:,2,:].shape                
        if self.short_cut and torsion_hidden.shape == conv_input[:,2,:].shape:            # 每层多了个残差
            torsion_hidden = torsion_hidden + conv_input[:,2,:]
        
        assert radius_hidden.shape == conv_input[:,3,:].shape                
        if self.short_cut and radius_hidden.shape == conv_input[:,3,:].shape:            # 每层多了个残差
            radius_hidden = radius_hidden + conv_input[:,3,:]
        
        node_vec = torch.cat((bond_hidden.unsqueeze(1), angle_hidden.unsqueeze(1), torsion_hidden.unsqueeze(1), radius_hidden.unsqueeze(1)), dim=1)
        
        node_vec = self.node_edgetype_attn(node_vec)
        
        bond_node_hidden = node_vec[:,0,:].squeeze(1)
        angle_node_hidden = node_vec[:,1,:].squeeze(1)
        torsion_node_hidden = node_vec[:,2,:].squeeze(1)
        radius_node_hidden = node_vec[:,3,:].squeeze(1)

        return bond_node_hidden, angle_node_hidden, torsion_node_hidden, radius_node_hidden

class EdgeConv(torch.nn.Module):
    
    def __init__(self, hidden_dim = 128, edge_attr_dim = 128,
                 activation="relu", attention = False, **kwargs):
        super().__init__(**kwargs)
        self.edge_attr_dim = edge_attr_dim
        self.attention = attention
        
        self.node_conv = torch.nn.Conv1d(4, 1, 1)

        self.edge_nn = MLP([hidden_dim*2+self.edge_attr_dim, hidden_dim*3, hidden_dim*2, hidden_dim], \
                            act=activation)  # 8层
        
        if self.attention:
            self.att_nn = MLP([hidden_dim, hidden_dim, hidden_dim, hidden_dim], \
                            act=activation)  # 3层
        
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Note that there is no bias due to BN
                fan_out = m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
    
    # def forward(self, node_attr, edge_index):   
    def forward(self, node_attr, edge_input, edge_index):
        
        row, col = edge_index
        node_emb = self.node_conv(node_attr).squeeze(1)
        source_node = node_emb[row, :]
        target_node = node_emb[col, :]
        
        out = torch.cat([source_node, target_node, edge_input], dim=1)
        
        out = self.edge_nn(out)
        if self.attention:
            att_val = self.att_nn(out)
            out = out * att_val
        
        return out

class GEncoder(torch.nn.Module):

    def __init__(self, hidden_dim, num_convs=3, nn_list = None, activation='relu', short_cut=True, concat_hidden=False, attention=True, attn_config=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = nn.Embedding(119, hidden_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.node_convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
            
        for i in range(self.num_convs):
            
            self.node_convs.append(NodeConv(hidden_dim, num_convs, activation, short_cut))
            self.edge_convs.append(EdgeConv(hidden_dim, hidden_dim, activation=activation, attention=attention))

    def forward(self, z, edge_index, edge_attr, edge_length, edge_mask):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """

        node_attr = self.node_emb(z)    # (num_node, hidden)
 
        bond_node_hiddens = []
        angle_node_hiddens = []
        torsion_node_hiddens = []
        radius_node_hiddens = []
        
        conv_input = node_attr.unsqueeze(1).expand(node_attr.shape[0],4,-1) # (num_node, hidden) 节点特征
        edge_input = edge_attr

        for conv_idx in range(self.num_convs):
            
            bond_node_hidden, angle_node_hidden, torsion_node_hidden, radius_node_hidden = self.node_convs[conv_idx](conv_input, edge_index, edge_input, edge_length, edge_mask, conv_idx)
            
            bond_node_hiddens.append(bond_node_hidden)
            angle_node_hiddens.append(angle_node_hidden)
            torsion_node_hiddens.append(torsion_node_hidden)
            radius_node_hiddens.append(radius_node_hidden)
            
            conv_input = torch.cat([bond_node_hidden.unsqueeze(1), angle_node_hidden.unsqueeze(1), torsion_node_hidden.unsqueeze(1), radius_node_hidden.unsqueeze(1)], 1)
            
            edge_input = self.edge_convs[conv_idx](conv_input, edge_input, edge_index)

        if self.concat_hidden:
            bond_node_feature = torch.cat(bond_node_hiddens, dim=-1)
            angle_node_feature = torch.cat(angle_node_hiddens, dim=-1)
            torsion_node_feature = torch.cat(torsion_node_hiddens, dim=-1)
            radius_node_feature = torch.cat(radius_node_hiddens, dim=-1)
        else:
            bond_node_feature = bond_node_hiddens[-1]
            angle_node_feature = angle_node_hiddens[-1]
            torsion_node_feature = torsion_node_hiddens[-1]
            radius_node_feature = radius_node_hiddens[-1]

        return bond_node_feature, angle_node_feature, torsion_node_feature, radius_node_feature, edge_input
