"""
FedALAServer: Federated Adaptive Local Aggregation Server
Implements the server-side logic for FedALA algorithm.

The server aggregates client updates using weighted averaging (same as FedAvg),
but clients use ALA module for adaptive local aggregation.
"""

import torch
from openfgl.flcore.base import BaseServer


class FedALAServer(BaseServer):
    """
    FedALAServer implements the server-side logic for FedALA algorithm.
    
    The server-side aggregation is similar to FedAvg:
    - Weighted average of client model parameters
    - Weight is proportional to number of samples
    
    The key difference is on the client side (ALA aggregation).
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initialize FedALAServer.
        
        Args:
            args: Arguments containing model and training configurations
            global_data: Global dataset accessible by the server
            data_dir: Directory containing the data
            message_pool: Pool for managing messages between server and clients
            device: Device to run computations on
        """
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
    
    def execute(self):
        """
        Execute server-side aggregation.
        
        Aggregates client model updates using weighted averaging:
        Θ^t = Σ_i (n_i / n_total) * Θ_i^t
        
        where:
        - n_i: number of samples for client i
        - n_total: total number of samples across all clients
        - Θ_i^t: model parameters from client i after local training
        """
        with torch.no_grad():
            # Calculate total number of samples
            num_tot_samples = sum([
                self.message_pool[f"client_{client_id}"]["num_samples"] 
                for client_id in self.message_pool["sampled_clients"]
            ])
            
            # Weighted aggregation
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                # Weight proportional to number of samples
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                # Aggregate parameters
                for (local_param, global_param) in zip(
                    self.message_pool[f"client_{client_id}"]["weight"],
                    self.task.model.parameters()
                ):
                    if it == 0:
                        # Initialize with first client's weighted parameters
                        global_param.data.copy_(weight * local_param)
                    else:
                        # Add subsequent clients' weighted parameters
                        global_param.data += weight * local_param
    
    def send_message(self):
        """
        Send message to clients containing the aggregated global model parameters.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }

