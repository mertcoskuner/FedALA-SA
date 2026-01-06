"""
FedALAClient: Federated Adaptive Local Aggregation Client
Implements the client-side logic for FedALA algorithm with ALA (Adaptive Local Aggregation) module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.fedala_config import config


class ALAModule(nn.Module):
    """
    Adaptive Local Aggregation (ALA) Module
    
    For each layer p, maintains element-wise weights W_i^p that control
    the combination of local and global model parameters.
    """
    
    def __init__(self, model, init_weight=0.5):
        """
        Initialize ALA module with element-wise weights for each parameter.
        
        Args:
            model: The neural network model (e.g., GCN)
            init_weight: Initial weight value (0.5 = equal weight to local and global)
        """
        super(ALAModule, self).__init__()
        self.ala_weights = nn.ParameterDict()
        
        # [FIX]: Store original names to maintain order and map to ALA names
        self.original_names = []
        
        # Initialize ALA weights for each parameter in the model
        for name, param in model.named_parameters():
            # [FIX]: Replace '.' with '_' to avoid KeyError in nn.ParameterDict
            ala_name = name.replace('.', '_')
            
            self.original_names.append(name) # Store original name for aggregation
            
            # Create element-wise weights with same shape as parameter
            init_val = torch.full_like(param, init_weight)
            # Store as parameter (will be learned)
            self.ala_weights[ala_name] = nn.Parameter(init_val)
    
    def aggregate(self, local_params, global_params):
        """
        Aggregate local and global parameters using ALA weights.
        
        Formula: Θ_i^t = W_i^p ⊙ Θ_i^{t-1} + (1 - W_i^p) ⊙ Θ^{t-1}
        
        Args:
            local_params: Dictionary of local model parameters (Θ_i^{t-1})
            global_params: List of global model parameters (Θ^{t-1})
            
        Returns:
            Dictionary of aggregated parameters
        """
        aggregated = {}
        
        # Map global_params list to a dictionary using original names (assuming list is ordered)
        global_dict = {
            name: param
            for name, param in zip(self.original_names, global_params)
        }
        
        # Apply ALA aggregation for each parameter
        for original_name in self.original_names:
            ala_name = original_name.replace('.', '_')
            
            # Get ALA weight for this parameter (apply sigmoid to ensure [0, 1])
            w = torch.sigmoid(self.ala_weights[ala_name])
            
            # Aggregate: w * local + (1-w) * global
            local_param = local_params[original_name]
            global_param = global_dict[original_name]
            
            # Perform aggregation
            aggregated[original_name] = w * local_param + (1 - w) * global_param
        
        return aggregated
    
    def get_weights(self):
        """Get current ALA weights (after sigmoid) indexed by ala_name (modified name)"""
        return {name: torch.sigmoid(w) for name, w in self.ala_weights.items()}


class FedALAClient(BaseClient):

    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        
        # Initialize ALA module
        self.ala_module = ALAModule(self.task.model, init_weight=config["ala_init"])
        self.ala_module.to(device)
        
        # Store previous local model parameters
        self.prev_local_params = None
        
        # Structure-Aware settings
        self.use_structure_aware = config.get("use_structure_aware", False)
        self.topo_lambda = config.get("topo_loss_lambda", 0.1)
        self.topo_anchor_threshold = config.get("topo_anchor_threshold", 0.8)
        
        # Optimizer for ALA weights (only update W_i^p, not model parameters)
        self.ala_optimizer = torch.optim.Adam(
            self.ala_module.parameters(),
            lr=config["ala_lr"]
        )
    
    def _save_local_params(self):
        """Save current local model parameters"""
        # Dictionary keys are original names (with dots)
        self.prev_local_params = {
            name: param.data.clone() 
            for name, param in self.task.model.named_parameters()
        }
    
    def _compute_topology_loss(self, embeddings, edge_index, degrees):


        max_degree = degrees.max().float()
        anchor_mask = (degrees.float() / max_degree) >= self.topo_anchor_threshold
        
        if not anchor_mask.any():
            return torch.tensor(0.0, device=embeddings.device)
        
        anchor_embeddings = embeddings[anchor_mask]
        anchor_degrees = degrees[anchor_mask]
        
        anchor_embeddings_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(anchor_embeddings_norm, anchor_embeddings_norm.t())
        
        degree_matrix = anchor_degrees.unsqueeze(1) - anchor_degrees.unsqueeze(0)
        degree_similarity = torch.exp(-torch.abs(degree_matrix) / max_degree)
        
        loss = F.mse_loss(similarity_matrix, degree_similarity)
        
        return loss

    
    def execute(self):
        # Save previous local parameters before aggregation
        if self.prev_local_params is None:
            self._save_local_params()
        
        # Get global model parameters from server
        global_params = self.message_pool["server"]["weight"]
        
        # Use ALA module to aggregate local and global parameters
        aggregated_params = self.ala_module.aggregate(
            self.prev_local_params,
            global_params
        )
        
        # Apply aggregated parameters to model
        with torch.no_grad():
            for name, param in self.task.model.named_parameters():
                param.data.copy_(aggregated_params[name])
        
        # Standard local training
        self.task.train()
        
        # If Structure-Aware variant is enabled, update ALA weights
        if self.use_structure_aware:
            self._update_ala_weights_with_topology()
        
        # Save current local parameters for next round
        self._save_local_params()
    
    def _update_ala_weights_with_topology(self):
        
        model = self.task.model
        data = self.task.data
        
        model.train()
        embeddings, logits = model(data)
        
        loss_fn = self.task.default_loss_fn
        if hasattr(self.task, 'train_mask') and self.task.train_mask is not None:
            train_mask = self.task.train_mask
            if hasattr(self.task, 'y'):
                y = self.task.y
            else:
                y = data.y
            task_loss = loss_fn(logits[train_mask], y[train_mask])
        else:
            task_loss = loss_fn(logits, data.y)
        
        edge_index = data.edge_index
        degrees = torch.zeros(data.x.size(0), device=data.x.device)
        degrees = degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=edge_index.device))
        
        topo_loss = self._compute_topology_loss(embeddings, edge_index, degrees)
        
        combined_loss = task_loss + self.topo_lambda * topo_loss
        
        model_params_requires_grad = {}
        for name, param in model.named_parameters():
            model_params_requires_grad[name] = param.requires_grad
            param.requires_grad = False
        
        self.ala_optimizer.zero_grad()
        combined_loss.backward()
        self.ala_optimizer.step()
        
        for name, param in model.named_parameters():
            param.requires_grad = model_params_requires_grad[name]
        
    
    def send_message(self):
  
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "ala_weights": self.ala_module.get_weights() 
        }