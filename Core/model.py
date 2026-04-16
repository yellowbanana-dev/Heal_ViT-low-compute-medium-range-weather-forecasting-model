import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import healpy as hp
from torch_geometric.nn import MessagePassing

class SphericalShift(nn.Module):
    def __init__(self, nside):
        super().__init__()
        self.nside = nside
        self.num_pixels = hp.nside2npix(nside)
        self.shift_indices = self._precompute_shift_indices(direction_idx=0)
        self.reverse_shift_indices = self._precompute_shift_indices(direction_idx=4)

    def _precompute_shift_indices(self, direction_idx):
        pixels = np.arange(self.num_pixels)
        neighbors = hp.get_all_neighbours(self.nside, pixels, nest=True)
        target_neighbors = neighbors[direction_idx, :]
        missing_neighbors = (target_neighbors == -1)
        target_neighbors[missing_neighbors] = pixels[missing_neighbors]
        return torch.from_numpy(target_neighbors).long()

    def forward(self, x, mode='shift'):
        indices = self.shift_indices if mode == 'shift' else self.reverse_shift_indices
        if indices.device != x.device:
            indices = indices.to(x.device)
            self.shift_indices = self.shift_indices.to(x.device)
            self.reverse_shift_indices = self.reverse_shift_indices.to(x.device)
        return x[:, indices, :]

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N // self.window_size, self.window_size, C)
        num_windows = x.shape[1]
        x = x.view(B * num_windows, self.window_size, C)
        q = self.q_proj(x).reshape(B * num_windows, self.window_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B * num_windows, self.window_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B * num_windows, self.window_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.ln_q(q), self.ln_k(k)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B * num_windows, self.window_size, C)
        x = x.view(B, N, C)
        return self.drop(self.proj(x))

class CustomViTBlock(nn.Module):
    def __init__(self, dim, num_heads, nside, window_size=64, shift=False, mlp_ratio=4., drop=0., 
                 use_checkpoint=False):
        super().__init__()
        self.shift = shift
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.shifter = SphericalShift(nside) if shift else None
        self.attn = WindowAttention(dim, num_heads, window_size, drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        def _inner_forward(x):
            y_prime = self.norm1(x)
            attn_input = self.shifter(y_prime, mode='shift') if self.shift else y_prime
            attn_out = self.attn(attn_input)
            if self.shift:
                attn_out = self.shifter(attn_out, mode='reverse')
            return x + attn_out + self.mlp(y_prime)

        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            return _inner_forward(x)

class GNNBlock(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super().__init__(aggr='add')
        self.aggregated_dim = node_in_dim + edge_in_dim
        self.norm = nn.LayerNorm(self.aggregated_dim)
        self.lin_out = nn.Linear(self.aggregated_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, size=None):
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return self.lin_out(self.norm(aggr_out))

    def message(self, x_j, edge_attr):
        return torch.cat([edge_attr, x_j], dim=-1)

class Encoder(nn.Module):
    def __init__(self, config, use_checkpoint=False):
        super().__init__()
        self.embedder = nn.Linear(config.in_features, config.embed_dim)
        self.act = nn.GELU()
        self.gnn_block = GNNBlock(config.embed_dim, config.edge_embed_dim, config.mesh_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x_grid, edge_index_G2M, edge_attr_G2M, size=None):
        v_G = self.act(self.embedder(x_grid))
        if self.use_checkpoint and v_G.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self.gnn_block, v_G, edge_index_G2M, edge_attr_G2M, size,
                use_reentrant=False
            )
        return self.gnn_block(x=v_G, edge_index=edge_index_G2M, edge_attr=edge_attr_G2M, size=size)

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, use_checkpoint=False):
        super().__init__()
        self.gnn_block = GNNBlock(in_dim, edge_dim, out_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, edge_index, edge_attr, size=None):
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self.gnn_block, x, edge_index, edge_attr, size,
                use_reentrant=False
            )
        return self.gnn_block(x, edge_index, edge_attr, size=size)

class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, use_checkpoint=False):
        super().__init__()
        self.gnn_block = GNNBlock(in_dim, edge_dim, out_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, edge_index, edge_attr, size=None):
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self.gnn_block, x, edge_index, edge_attr, size,
                use_reentrant=False
            )
        return self.gnn_block(x, edge_index, edge_attr, size=size)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, use_checkpoint=False):
        super().__init__()
        self.gnn_block = GNNBlock(in_dim, edge_dim, out_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, edge_index, edge_attr, size=None):
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self.gnn_block, x, edge_index, edge_attr, size,
                use_reentrant=False
            )
        return self.gnn_block(x, edge_index, edge_attr, size=size)

class HEALViTModel(nn.Module):
    def __init__(self, config, graph_data):
        super().__init__()
        
        self.N_mesh = graph_data['N_mesh']
        self.N_mesh_coarse = graph_data['N_mesh_coarse']
        self.n_grid = config.n_grid
        
        self.register_buffer('edge_index_G2M', graph_data['edge_index_G2M'])
        self.register_buffer('edge_index_Mf2Mc', graph_data['edge_index_Mf2Mc'])
        self.register_buffer('edge_index_Mc2Mf', graph_data['edge_index_Mc2Mf'])
        self.register_buffer('edge_index_M2G', graph_data['edge_index_M2G'])

        self.edge_attr_G2M = nn.Parameter(torch.randn(graph_data['num_edges_G2M'], config.edge_embed_dim))
        self.edge_attr_Mf2Mc = nn.Parameter(torch.randn(graph_data['num_edges_Mf2Mc'], config.edge_embed_dim))
        self.edge_attr_Mc2Mf = nn.Parameter(torch.randn(graph_data['num_edges_Mc2Mf'], config.edge_embed_dim))
        self.edge_attr_M2G = nn.Parameter(torch.randn(graph_data['num_edges_M2G'], config.edge_embed_dim))

        self.encoder = Encoder(config, use_checkpoint=config.use_checkpoint)
        
        window_size_fine = min(64, self.N_mesh // 4)
        window_size_coarse = min(64, self.N_mesh_coarse // 4)
        
        nside_fine = int(np.sqrt(self.N_mesh / 12))
        nside_coarse = int(np.sqrt(self.N_mesh_coarse / 12))
        
        self.enc_layers = nn.ModuleList([
            CustomViTBlock(config.mesh_dim, config.num_heads, nside=nside_fine, 
                          window_size=window_size_fine, shift=False, 
                          mlp_ratio=config.mlp_ratio, drop=config.drop_rate,
                          use_checkpoint=config.use_checkpoint) 
            for _ in range(2)
        ])
        
        self.downsample = Downsample(config.mesh_dim, config.mesh_dim * 2, config.edge_embed_dim, use_checkpoint=config.use_checkpoint)
        
        self.bottleneck_layers = nn.ModuleList([
            CustomViTBlock(config.mesh_dim * 2, config.num_heads * 2, nside=nside_coarse,
                          window_size=window_size_coarse, shift=(i % 2 != 0), 
                          mlp_ratio=config.mlp_ratio, drop=config.drop_rate,
                          use_checkpoint=config.use_checkpoint) 
            for i in range(config.num_bottleneck_layers)
        ])
        
        self.upsample = Upsample(config.mesh_dim * 2, config.mesh_dim, config.edge_embed_dim, use_checkpoint=config.use_checkpoint)
        
        self.dec_layers = nn.ModuleList([
            CustomViTBlock(config.mesh_dim * 2, config.num_heads * 2, nside=nside_fine,
                          window_size=window_size_fine, shift=False, 
                          mlp_ratio=config.mlp_ratio, drop=config.drop_rate,
                          use_checkpoint=config.use_checkpoint) 
            for _ in range(2)
        ])
        
        self.decoder = Decoder(config.mesh_dim * 2, config.out_features, config.edge_embed_dim, use_checkpoint=config.use_checkpoint)

    def _batch_edge_index(self, edge_index, batch_size, src_offset, dst_offset):
        edge_list = []
        for i in range(batch_size):
            edge_list.append(edge_index + torch.tensor([[src_offset * i], [dst_offset * i]], 
                                                        device=edge_index.device))
        return torch.cat(edge_list, dim=1)
    
    def forward(self, data):
        x_grid = data.x
        batch_size = data.num_graphs
        
        edge_attr_G2M = self.edge_attr_G2M.repeat(batch_size, 1)
        edge_attr_Mf2Mc = self.edge_attr_Mf2Mc.repeat(batch_size, 1)
        edge_attr_Mc2Mf = self.edge_attr_Mc2Mf.repeat(batch_size, 1)
        edge_attr_M2G = self.edge_attr_M2G.repeat(batch_size, 1)

        N_grid_batch = batch_size * self.n_grid
        N_mesh_batch = batch_size * self.N_mesh
        N_mesh_coarse_batch = batch_size * self.N_mesh_coarse
        
        edge_index_G2M_batched = self._batch_edge_index(
            self.edge_index_G2M, batch_size, self.n_grid, self.N_mesh)
        edge_index_Mf2Mc_batched = self._batch_edge_index(
            self.edge_index_Mf2Mc, batch_size, self.N_mesh, self.N_mesh_coarse)
        edge_index_Mc2Mf_batched = self._batch_edge_index(
            self.edge_index_Mc2Mf, batch_size, self.N_mesh_coarse, self.N_mesh)
        edge_index_M2G_batched = self._batch_edge_index(
            self.edge_index_M2G, batch_size, self.N_mesh, self.n_grid)

        v_M = self.encoder(x_grid, edge_index_G2M_batched, edge_attr_G2M, 
                           size=(N_grid_batch, N_mesh_batch))
        
        C = v_M.shape[-1]
        skip = v_M.view(batch_size, self.N_mesh, C)
        for layer in self.enc_layers:
            skip = layer(skip)
        skip_flat = skip.view(batch_size * self.N_mesh, C)
        
        v_M_coarse = self.downsample(skip_flat, edge_index_Mf2Mc_batched, edge_attr_Mf2Mc,
                                      size=(N_mesh_batch, N_mesh_coarse_batch))
        
        C2 = v_M_coarse.shape[-1]
        v_M_bottle = v_M_coarse.view(batch_size, self.N_mesh_coarse, C2)
        for layer in self.bottleneck_layers:
            v_M_bottle = layer(v_M_bottle)
        v_M_bottle_flat = v_M_bottle.view(batch_size * self.N_mesh_coarse, C2)
        
        v_M_up = self.upsample(v_M_bottle_flat, edge_index_Mc2Mf_batched, edge_attr_Mc2Mf,
                               size=(N_mesh_coarse_batch, N_mesh_batch))
        
        v_M_up_3d = v_M_up.view(batch_size, self.N_mesh, -1)
        x_concat = torch.cat([skip, v_M_up_3d], dim=-1)
        
        v_M_out = x_concat
        for layer in self.dec_layers:
            v_M_out = layer(v_M_out)
        
        v_M_out_flat = v_M_out.view(batch_size * self.N_mesh, -1)
        
        return self.decoder(v_M_out_flat, edge_index_M2G_batched, edge_attr_M2G,
                            size=(N_mesh_batch, N_grid_batch))
