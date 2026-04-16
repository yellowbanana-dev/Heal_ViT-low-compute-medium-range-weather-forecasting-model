import numpy as np
import torch
import healpy as hp
from scipy.spatial import cKDTree

class HealpixGraphGenerator:
    """Generate graph structure for the given grid."""
    
    def __init__(self, config):
        self.H_grid = config.h_grid
        self.W_grid = config.w_grid
        self.N_grid = self.H_grid * self.W_grid
        
        # Adjusted mesh for larger grid
        self.nside_fine = 64
        self.nside_coarse = 32
        
        self.N_mesh = hp.nside2npix(self.nside_fine)
        self.N_mesh_coarse = hp.nside2npix(self.nside_coarse)
        
        print(f"Grid: {self.H_grid}x{self.W_grid} = {self.N_grid} points")
        print(f"Fine mesh (nside={self.nside_fine}): {self.N_mesh} nodes")
        print(f"Coarse mesh (nside={self.nside_coarse}): {self.N_mesh_coarse} nodes")

    def _get_grid_lon_lat(self):
        lon = np.linspace(0, 360, self.W_grid, endpoint=False)
        lat = np.linspace(90, -90, self.H_grid)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        return lon_grid.flatten(), lat_grid.flatten()

    def _get_mesh_vectors(self, nside, indices):
        return np.array(hp.pix2vec(nside, indices, nest=True)).T

    def _get_grid_vectors(self, lon, lat):
        return np.array(hp.ang2vec(lon, lat, lonlat=True))

    def _generate_coordinates(self):
        self.grid_lon, self.grid_lat = self._get_grid_lon_lat()
        self.grid_vec_3d = self._get_grid_vectors(self.grid_lon, self.grid_lat)
        
        ring_indices_fine = np.arange(self.N_mesh)
        self.nested_fine_mesh_indices = hp.ring2nest(self.nside_fine, ring_indices_fine)
        self.fine_mesh_vec_3d = self._get_mesh_vectors(self.nside_fine, self.nested_fine_mesh_indices)
        
        ring_indices_coarse = np.arange(self.N_mesh_coarse)
        self.nested_coarse_mesh_indices = hp.ring2nest(self.nside_coarse, ring_indices_coarse)
        self.coarse_mesh_vec_3d = self._get_mesh_vectors(self.nside_coarse, self.nested_coarse_mesh_indices)

    def _build_kd_trees(self):
        self.fine_mesh_tree = cKDTree(self.fine_mesh_vec_3d)
        self.coarse_mesh_tree = cKDTree(self.coarse_mesh_vec_3d)

    def _build_edge_index_g2m(self):
        target_mesh_nodes = hp.ang2pix(self.nside_fine, self.grid_lon, self.grid_lat, lonlat=True, nest=True)
        source_grid_nodes = np.arange(self.N_grid)
        return torch.from_numpy(np.stack([source_grid_nodes, target_mesh_nodes])).long()

    def _build_edge_index_mf2mc(self):
        source_fine_nodes_nested = self.nested_fine_mesh_indices
        target_coarse_nodes_nested = source_fine_nodes_nested // 4
        return torch.from_numpy(np.stack([source_fine_nodes_nested, target_coarse_nodes_nested])).long()

    def _build_edge_index_mc2mf(self):
        _, nearest_indices_k4 = self.coarse_mesh_tree.query(self.fine_mesh_vec_3d, k=4)
        source_coarse_nodes_k4_nested = self.nested_coarse_mesh_indices[nearest_indices_k4.flatten()]
        target_fine_nodes_k4_nested = np.repeat(self.nested_fine_mesh_indices, 4)
        return torch.from_numpy(np.stack([source_coarse_nodes_k4_nested, target_fine_nodes_k4_nested])).long()

    def _build_edge_index_m2g(self):
        _, nearest_indices_k4 = self.fine_mesh_tree.query(self.grid_vec_3d, k=4)
        source_mesh_nodes_k4_nested = self.nested_fine_mesh_indices[nearest_indices_k4.flatten()]
        target_grid_nodes_k4 = np.repeat(np.arange(self.N_grid), 4)
        return torch.from_numpy(np.stack([source_mesh_nodes_k4_nested, target_grid_nodes_k4])).long()

    def generate_graph_data(self):
        print("Generating graph data...")
        self._generate_coordinates()
        self._build_kd_trees()
        
        graph_data = {
            'edge_index_G2M': self._build_edge_index_g2m(),
            'edge_index_Mf2Mc': self._build_edge_index_mf2mc(),
            'edge_index_Mc2Mf': self._build_edge_index_mc2mf(),
            'edge_index_M2G': self._build_edge_index_m2g(),
            'N_mesh': self.N_mesh,
            'N_mesh_coarse': self.N_mesh_coarse,
        }
        graph_data['num_edges_G2M'] = graph_data['edge_index_G2M'].shape[1]
        graph_data['num_edges_Mf2Mc'] = graph_data['edge_index_Mf2Mc'].shape[1]
        graph_data['num_edges_Mc2Mf'] = graph_data['edge_index_Mc2Mf'].shape[1]
        graph_data['num_edges_M2G'] = graph_data['edge_index_M2G'].shape[1]
        
        print("Graph data generated.")
        return graph_data
