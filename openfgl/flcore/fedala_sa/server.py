import torch
from openfgl.flcore.base import BaseServer

class FedALAServer(BaseServer):
    """
    FedALAServer implements the server-side logic for FedALA algorithm.
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.fedala_sa_option = getattr(args, 'fedala_sa_option', None)
    
    def execute(self):
        """
        Execute server-side aggregation.
        """
        with torch.no_grad():
            # Calculate total weight (samples or samples * degree)
            total_weight = 0.0
            client_weights = {}
            
            for client_id in self.message_pool["sampled_clients"]:
                msg = self.message_pool[f"client_{client_id}"]
                num_samples = msg["num_samples"]
                
                if self.fedala_sa_option == 1 and "avg_degree" in msg:
                    # Option 1: Scale by average node degree
                    # Weight = num_samples * avg_degree
                    # This increases importance of clients with denser subgraphs
                    weight = num_samples * msg["avg_degree"]
                else:
                    weight = num_samples
                
                client_weights[client_id] = weight
                total_weight += weight
            
            # Weighted aggregation
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = client_weights[client_id] / total_weight
                
                # Aggregate parameters
                for (local_param, global_param) in zip(
                    self.message_pool[f"client_{client_id}"]["weight"],
                    self.task.model.parameters()
                ):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
    
    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }

