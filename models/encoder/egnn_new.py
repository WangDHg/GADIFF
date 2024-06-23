from torch import nn
import torch
import math
from .ginnew import GINEConv
from .mlp_implement import MLP
from models.network.util.Block_process import Multi_Attn, Weight_Multi_Attn
from ..common import assemble_atom_pair_feature
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
from torch_scatter import scatter_add, scatter_mean, scatter_max

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), short_cut=True, attention=False, node_attn_config=None):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.edge_attention = attention
        self.short_cut = short_cut
        
        # Edge Model
        self.node_conv = torch.nn.Conv1d(4, 1, 1)
        
        self.edge_mlp = MLP([input_edge + edges_in_d, hidden_nf*3, hidden_nf*2, hidden_nf], act=act_fn.__class__.__name__, is_res=True)  # 3层
        
        if self.edge_attention:
            self.att_mlp = nn.Sequential(
                MLP([hidden_nf, hidden_nf, hidden_nf, 1], act=act_fn.__class__.__name__, is_res=True),  # 3层
                nn.Sigmoid())

        # Node Model
        # self.node_mlp = MLP([hidden_nf + nodes_att_dim, hidden_nf, hidden_nf, hidden_nf, output_nf], act=act_fn.__class__.__name__, is_res=True)
        
        self.bond_convs = GINEConv(MLP([hidden_nf + nodes_att_dim, hidden_nf, hidden_nf, hidden_nf, output_nf], act=act_fn.__class__.__name__, is_res=True), edge_dim=hidden_nf, activation=act_fn.__class__.__name__)   # 4 层
        self.angle_convs = GINEConv(MLP([hidden_nf + nodes_att_dim, hidden_nf, hidden_nf, hidden_nf, output_nf], act=act_fn.__class__.__name__, is_res=True), edge_dim=hidden_nf, activation=act_fn.__class__.__name__)
        self.torsion_convs = GINEConv(MLP([hidden_nf + nodes_att_dim, hidden_nf, hidden_nf, hidden_nf, output_nf], act=act_fn.__class__.__name__, is_res=True), edge_dim=hidden_nf, activation=act_fn.__class__.__name__)
        self.radius_convs = GINEConv(MLP([hidden_nf + nodes_att_dim, hidden_nf, hidden_nf, hidden_nf, output_nf], act=act_fn.__class__.__name__, is_res=True), edge_dim=hidden_nf, activation=act_fn.__class__.__name__)
        self.activation = act_fn

        if node_attn_config != None:
            self.node_edgetype_attn = Multi_Attn(
                input_dim=node_attn_config.input_dim, 
                output_dim=node_attn_config.output_dim, 
                layer_num=node_attn_config.layer_num, 
                hidden_dim=hidden_nf, 
                head_num=node_attn_config.head_num,
                drop_rate=node_attn_config.drop_rate,
                )
        else:
            self.node_edgetype_attn = Multi_Attn(
                input_dim=4, 
                output_dim=4, 
                layer_num=1, 
                hidden_dim=hidden_nf, 
                head_num=32,
                drop_rate=0.001,
                )

    def edge_model(self, node_attr, edge_attr, edge_index, edge_mask):
        
        row, col = edge_index
        node_emb = self.node_conv(node_attr).squeeze(1)
        source = node_emb[row, :]
        target = node_emb[col, :]
        
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.edge_attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        # if edge_mask is not None:
        #     out = out * edge_mask
        return out, mij
    
    def node_model(self, node_attr, edge_index, edge_attr, edge_mask, edge_length=None, node_type=None, conv_idx=None):
        conv_input = node_attr # (num_node, hidden) 节点特征

        bond_hidden = self.bond_convs(
            x = conv_input[:,0,:],
            edge_index = edge_index[:, edge_mask[0]],
            edge_attr = edge_attr[edge_mask[0]],
            node_attr = node_type,
        )
        angle_hidden = self.angle_convs(
            x = conv_input[:,1,:], 
            edge_index = edge_index[:, edge_mask[1]], 
            edge_attr = edge_attr[edge_mask[1]],
            node_attr = node_type,
        )
        torsion_hidden = self.torsion_convs(
            x = conv_input[:,2,:], 
            edge_index = edge_index[:, edge_mask[2]], 
            edge_attr = edge_attr[edge_mask[2]],
            node_attr = node_type,
        )
        radius_hidden = self.radius_convs(
            x = conv_input[:,3,:], 
            edge_index = edge_index[:, edge_mask[3]], 
            edge_attr = edge_attr[edge_mask[3]],
            node_attr = node_type,
        )
        
        if self.activation is not None:
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

    def forward(self, h, edge_index, edge_attr=None, edge_length=None, node_type=None, node_mask=None, edge_mask=None, conv_idx=None):
        
        edge_feat, mij = self.edge_model(h, edge_attr, edge_index, edge_mask)
        bond_node_hidden, angle_node_hidden, torsion_node_hidden, radius_node_hidden = self.node_model(h, edge_index, edge_feat, edge_mask, edge_length, node_type, conv_idx)
        h = torch.cat([bond_node_hidden.unsqueeze(1), angle_node_hidden.unsqueeze(1), torsion_node_hidden.unsqueeze(1), radius_node_hidden.unsqueeze(1)], 1)
        
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method, config,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        
        input_edge = hidden_nf * 2 + edges_in_d
        self.coord_mlp_bond =  MLP(
            [input_edge, hidden_nf*2, hidden_nf, hidden_nf//2, 1],          # 3H 8层
            # act=config.mlp_act,
            act=act_fn.__class__.__name__,
            dropout= 0.001,
            is_res=True,
        )# (E_local, 1)
        
        self.coord_mlp_angle = MLP(
            [input_edge, hidden_nf*2, hidden_nf, hidden_nf//2, 1],          # 3H 8层
            # act=config.mlp_act,
            act=act_fn.__class__.__name__,
            dropout= 0.001,
            is_res=True,
        )# (E_local, 1)
        
        self.coord_mlp_torsion = MLP(
            [input_edge, hidden_nf*2, hidden_nf, hidden_nf//2, 1],          # 3H 8层
            # act=config.mlp_act,
            act=act_fn.__class__.__name__,
            dropout= 0.001,
            is_res=True,
        )# (E_local, 1)
        
        self.coord_mlp_radius = MLP(
            [input_edge, hidden_nf*2, hidden_nf, hidden_nf//2, 1],          # 3H 8层
            # act=config.mlp_act,
            act=act_fn.__class__.__name__,
            dropout= 0.001,
            is_res=True,
        )# (E_local, 1)
        
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        
        self.w_gen = Weight_Multi_Attn(
            input_dim=config.weight_func.input_dim, 
            output_dim=config.weight_func.output_dim, 
            layer_num=config.weight_func.layer_num, 
            hidden_dim=config.weight_func.hidden_dim, 
            head_num=config.weight_func.head_num,
            drop_rate=config.weight_func.drop_rate,
            )
        
        # self.edge_emb = nn.Linear(edges_in_d, hidden_nf)

    def coord_model(self, conv_input, coord, edge_index, coord_diff, edge_attr, edge_mask, edge_length,N):
        
        h_pair_bond = assemble_atom_pair_feature(
            node_attr=conv_input[:,0,:],
            edge_index=edge_index[:,edge_mask[0]],
            edge_attr=edge_attr[edge_mask[0]],
        )    # (E_local, 2H)
        
        h_pair_angle = assemble_atom_pair_feature(
            node_attr=conv_input[:,1,:],
            edge_index=edge_index[:,edge_mask[1]],
            edge_attr=edge_attr[edge_mask[1]],
        )    # (E_local, 2H)
        
        h_pair_torsion = assemble_atom_pair_feature(
            node_attr=conv_input[:,2,:],
            edge_index=edge_index[:,edge_mask[2]],
            edge_attr=edge_attr[edge_mask[2]],
        )    # (E_local, 2H)
        
        h_pair_radius = assemble_atom_pair_feature(
            node_attr=conv_input[:,3,:],
            edge_index=edge_index[:,edge_mask[3]],
            edge_attr=edge_attr[edge_mask[3]],
        )    # (E_local, 2H)
        
        if self.tanh:
            edge_inv_bond = torch.tanh(self.coord_mlp_bond(h_pair_bond)) * self.coords_range  # (E_local, 1)
            edge_inv_angle = torch.tanh(self.coord_mlp_angle(h_pair_angle)) * self.coords_range # (E_local, 1)
            edge_inv_torsion = torch.tanh(self.coord_mlp_torsion(h_pair_torsion)) * self.coords_range # (E_local, 1)
            edge_inv_radius = torch.tanh(self.coord_mlp_radius(h_pair_radius)) * self.coords_range # (E_local, 1)
        else:
            edge_inv_bond = self.coord_mlp_bond(h_pair_bond)  # (E_local, 1)
            edge_inv_angle = self.coord_mlp_angle(h_pair_angle) # (E_local, 1)
            edge_inv_torsion = self.coord_mlp_torsion(h_pair_torsion) # (E_local, 1)
            edge_inv_radius = self.coord_mlp_radius(h_pair_radius) # (E_local, 1)

        node_eq_bond = eq_transform(edge_inv_bond, coord, edge_index[:, edge_mask[0]], edge_length[edge_mask[0]])  # (N, 3)   prediction
        node_eq_angle = eq_transform(edge_inv_angle, coord, edge_index[:, edge_mask[1]], edge_length[edge_mask[1]])  # (N, 3)   prediction
        node_eq_torsion = eq_transform(edge_inv_torsion, coord, edge_index[:, edge_mask[2]], edge_length[edge_mask[2]])  # (N, 3)   prediction
        node_eq_radius = eq_transform(edge_inv_radius, coord, edge_index[:, edge_mask[3]], edge_length[edge_mask[3]])  # (N, 3)   prediction
        
        # node_attr + edge_attr
        node_bond_edge0 = scatter_mean(edge_attr[edge_mask[0]], edge_index[:,edge_mask[0]][0], dim=0, dim_size=N)
        node_bond_edge1 = scatter_mean(edge_attr[edge_mask[0]], edge_index[:,edge_mask[0]][1], dim=0, dim_size=N)
        node_angle_edge0 = scatter_mean(edge_attr[edge_mask[1]], edge_index[:,edge_mask[1]][0], dim=0, dim_size=N)
        node_angle_edge1 = scatter_mean(edge_attr[edge_mask[1]], edge_index[:,edge_mask[1]][1], dim=0, dim_size=N)
        node_torsion_edge0 = scatter_mean(edge_attr[edge_mask[2]], edge_index[:,edge_mask[2]][0], dim=0, dim_size=N)
        node_torsion_edge1 = scatter_mean(edge_attr[edge_mask[2]], edge_index[:,edge_mask[2]][1], dim=0, dim_size=N)
        node_radius_edge0 = scatter_mean(edge_attr[edge_mask[3]], edge_index[:,edge_mask[3]][0], dim=0, dim_size=N)
        node_radius_edge1 = scatter_mean(edge_attr[edge_mask[3]], edge_index[:,edge_mask[3]][1], dim=0, dim_size=N)
        
        node_bond = torch.cat((conv_input[:,0,:], node_bond_edge0, node_bond_edge1), dim=1)
        node_angle = torch.cat((conv_input[:,1,:], node_angle_edge0, node_angle_edge1), dim=1)
        node_torsion = torch.cat((conv_input[:,2,:], node_torsion_edge0, node_torsion_edge1), dim=1)
        node_radius = torch.cat((conv_input[:,3,:], node_radius_edge0, node_radius_edge1), dim=1)
        
        w = torch.cat((node_bond.unsqueeze(1), node_angle.unsqueeze(1), node_torsion.unsqueeze(1), node_radius.unsqueeze(1)), dim=1)
        
        weight_type = self.w_gen(w)
        
        node_eq_pos = weight_type[:,0].unsqueeze(-1).expand(-1,3)*node_eq_bond+weight_type[:,1].unsqueeze(-1).expand(-1,3)*node_eq_angle+weight_type[:,2].unsqueeze(-1).expand(-1,3)*node_eq_torsion+weight_type[:,3].unsqueeze(-1).expand(-1,3)*node_eq_radius   # (N)
        
        # coord = coord + agg
        # return coord, agg
        
        coord = coord + node_eq_pos
        return coord, node_eq_pos

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None, edge_length=None, N=None):
        # edge_attr = self.edge_emb(edge_attr)
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask, edge_length, N)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', node_config=None, config=None, concat_hidden=False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.concat_hidden = concat_hidden
        
        self.edge_emb = MLP([edge_feat_nf, self.hidden_nf, self.hidden_nf], act=act_fn.__class__.__name__, is_res=True)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.hidden_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method,
                                              node_attn_config=node_config))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=self.hidden_nf, act_fn=act_fn, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method, 
                                                       config=config))
        self.to(self.device)

    def forward(self, h, x, edge_index, edge_type, node_mask=None, edge_mask=None, edge_attr=None, node_type=None, edge_length=None, N=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances, edge_type)
        edge_attr = torch.cat([distances, edge_attr], dim=1)     # 24
        edge_attr = self.edge_emb(edge_attr)                     # 128
        
        bond_node_hiddens = []
        angle_node_hiddens = []
        torsion_node_hiddens = []
        radius_node_hiddens = []
        
        conv_input = h.unsqueeze(1).expand(h.shape[0],4,-1)
        
        # for i in range(0, self.n_layers):
        #     h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        # x, noise = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        
        for i in range(0, self.n_layers):
            conv_input, _ = self._modules["gcl_%d" % i](conv_input, edge_index, edge_attr=edge_attr, edge_length=edge_length, node_type=node_type, node_mask=node_mask, edge_mask=edge_mask, conv_idx=i)
            bond_node_hiddens.append(conv_input[:,0,:])
            angle_node_hiddens.append(conv_input[:,1,:])
            torsion_node_hiddens.append(conv_input[:,2,:])
            radius_node_hiddens.append(conv_input[:,3,:])
        
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
        
        conv_input = torch.cat([bond_node_feature.unsqueeze(1), angle_node_feature.unsqueeze(1), torsion_node_feature.unsqueeze(1), radius_node_feature.unsqueeze(1)], 1)
        
        x, noise = self._modules["gcl_equiv"](conv_input, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask, edge_length, N)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x, noise


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', node_config=None, config=None):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 4
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Embedding(119, self.hidden_nf)
        # self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               node_config=node_config, config=config))
        self.to(self.device)

    def forward(self, h, x, edge_index, edge_type=None, node_mask=None, edge_mask=None, node_type=None, edge_length=None, N=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        noise_con = []
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances, edge_type)     # 12
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, noise = self._modules["e_block_%d" % i](h, x, edge_index, edge_type=edge_type, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances, node_type=node_type, edge_length=edge_length, N=N)
            noise_con.append(noise.unsqueeze(0))
            edge_length = get_distance(x, edge_index).unsqueeze(-1)   # (E, 1)
        
        noise_final = torch.cat(noise_con, dim=0).sum(0)
        return noise_final, h, x



class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1   # 6
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2     # 12
        self.bond_emb = nn.Embedding(26, embedding_dim=self.dim)

    def forward(self, x, edge_type = None):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        if edge_type == None:
            return emb.detach()
        else:
            edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
            # emb_new = emb * edge_attr
            emb_new = torch.cat((emb, edge_attr), dim=-1)
            return emb_new.detach() # (num_edge, hidden)
        
        # return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
