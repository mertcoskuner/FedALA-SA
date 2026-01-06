import torch
import torch.nn as nn
import torch.nn.functional as F
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.fedala_config import config


try:
    from openfgl.utils.privacy_utils2 import clip_gradients, add_noise
except ImportError:
    print("!!!!: 'openfgl.utils.privacy_utils' .")
    def clip_gradients(*args, **kwargs): pass
    def add_noise(*args, **kwargs): pass

class ALAModule(nn.Module):
    """
    Adaptive Local Aggregation (ALA) Module
    """
    
    def __init__(self, model, init_weight=0.5, target_layers=None):
        super(ALAModule, self).__init__()
        self.ala_weights = nn.ParameterDict()
        self.original_names = []
        
        for name, param in model.named_parameters():
            if target_layers is not None:
                is_target = False
                for layer_name in target_layers:
                    if layer_name in name:
                        is_target = True
                        break
                if not is_target:
                    continue

            ala_name = name.replace('.', '_')
            self.original_names.append(name)
            init_val = torch.full_like(param, init_weight)
            self.ala_weights[ala_name] = nn.Parameter(init_val)
    
    def aggregate(self, local_params, global_params):
        aggregated = {}
        # [GÜVENLİK] Global parametreleri isim sırasına göre map ediyoruz
        ordered_names = [n for n in local_params.keys()]
        global_dict = {
            name: param
            for name, param in zip(ordered_names, global_params) 
        }
        
        for name, param in local_params.items():
            if name in self.original_names:
                ala_name = name.replace('.', '_')
                w = torch.sigmoid(self.ala_weights[ala_name])
                global_p = global_dict[name]
                aggregated[name] = w * param + (1 - w) * global_p
            else:
                aggregated[name] = global_dict[name]
                
        return aggregated
    
    def get_weights(self):
        return {name: torch.sigmoid(w) for name, w in self.ala_weights.items()}


class FedALAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        
        self.fedala_sa_option = getattr(args, 'fedala_sa_option', None)
        self.mu = getattr(args, 'fedala_sa_mu', 0.01) # Proximal term weight
        
        # Option 3: Classifier only logic
        target_layers = None
        if self.fedala_sa_option == 3:
            all_names = [n for n, _ in self.task.model.named_parameters()]
            classifier_names = [n for n in all_names if 'classifier' in n or 'fc' in n]
            if not classifier_names:
                target_layers = all_names[-2:] if len(all_names) >= 2 else all_names
            else:
                target_layers = classifier_names

        self.ala_module = ALAModule(self.task.model, init_weight=config["ala_init"], target_layers=target_layers)
        self.ala_module.to(device)
        
        self.prev_local_params = None
        
        self.ala_optimizer = torch.optim.Adam(
            self.ala_module.parameters(),
            lr=config["ala_lr"]
        )
        
        
        self.avg_degree = 0.0
       
        if hasattr(self.task.data, 'edge_index'):
            num_edges = self.task.data.edge_index.shape[1]
            num_nodes = self.task.data.x.shape[0]
            if num_nodes > 0:
                self.avg_degree = num_edges / num_nodes
            
    def _save_local_params(self):
        self.prev_local_params = {
            name: param.data.clone() 
            for name, param in self.task.model.named_parameters()
        }

    def train_prox(self):
        """
        Train with Proximal Term (Option 2) + Structure-Aware Privacy
        Loss = TaskLoss + (mu / 2) * ||w - w_global||^2
        """
        global_params = {n: p.data.clone().detach() for n, p in self.task.model.named_parameters()}
        
        self.task.model.train()
        
        for _ in range(self.args.local_epochs):
            self.task.optim.zero_grad()
            
            data = self.task.processed_data["data"] if hasattr(self.task, "processed_data") else self.task.data
            
            if hasattr(self.task, 'train_mask') and self.task.train_mask is not None:
                train_mask = self.task.train_mask
            elif hasattr(self.task.processed_data, 'train_mask'):
                train_mask = self.task.processed_data["train_mask"]
            else:
                train_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=data.y.device)

            # Forward pass
            embedding, logits = self.task.model(data)
            
            
            loss_vec = F.cross_entropy(logits[train_mask], data.y[train_mask], reduction='none')
            
            
            prox_loss = 0.0
            for name, param in self.task.model.named_parameters():
                prox_loss += torch.norm(param - global_params[name]) ** 2
            
            
            total_loss_vec = loss_vec + (self.mu / 2) * prox_loss
            
            
            dp_mech = getattr(self.args, 'dp_mech', 'no_dp')
            
            if dp_mech != "no_dp":
                
                clip_gradients(self.task.model, total_loss_vec, total_loss_vec.shape[0], dp_mech, getattr(self.args, 'grad_clip', 1.0))
                
                
                self.task.optim.step()
                
                
                degree_val = self.avg_degree if hasattr(self, 'avg_degree') else None
                
                
                add_noise(self.args, self.task.model, total_loss_vec.shape[0], avg_degree=degree_val)
                
            else:
                
                loss = total_loss_vec.mean()
                loss.backward()
                self.task.optim.step()

    def execute(self):
        if self.prev_local_params is None:
            self._save_local_params()
        
        global_params = self.message_pool["server"]["weight"]
        
        # Aggregate
        aggregated_params = self.ala_module.aggregate(
            self.prev_local_params,
            global_params
        )
        
        # Apply aggregated parameters
        with torch.no_grad():
            for name, param in self.task.model.named_parameters():
                param.data.copy_(aggregated_params[name])
        
        # Train
        if self.fedala_sa_option == 2:
            self.train_prox()
        else:
            self.task.train()
            
        self._update_ala_weights()
        self._save_local_params()

    def _update_ala_weights(self):
        # Standard ALA weight update
        self.task.model.eval() 
        global_params = self.message_pool["server"]["weight"]
        
        ordered_names = [n for n in self.prev_local_params.keys()]
        global_dict = {
            name: param
            for name, param in zip(ordered_names, global_params)
        }
        
        from torch.func import functional_call
        
        for _ in range(self.args.local_epochs): 
            self.ala_optimizer.zero_grad()
            
            temp_params = {}
            for name, param in self.prev_local_params.items():
                if name in self.ala_module.original_names:
                    ala_name = name.replace('.', '_')
                    w = torch.sigmoid(self.ala_module.ala_weights[ala_name])
                    global_p = global_dict[name]
                    temp_params[name] = w * param.detach() + (1 - w) * global_p.detach()
                else:
                    temp_params[name] = global_dict[name].detach()
            
            # functional_call ile forward pass
            state_dict = dict(self.task.model.named_buffers())
            state_dict.update(temp_params)
            
            data = self.task.processed_data["data"] if hasattr(self.task, "processed_data") else self.task.data
            
            try:
                embedding, logits = functional_call(self.task.model, state_dict, (data,))
                
                if hasattr(self.task, 'train_mask') and self.task.train_mask is not None:
                     loss = self.task.loss_fn(logits[self.task.train_mask], data.y[self.task.train_mask])
                else:
                     loss = self.task.loss_fn(logits, data.y)
                
                loss.backward()
                self.ala_optimizer.step()
            except Exception as e:
            
                pass

    def send_message(self):
        msg = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
        if self.fedala_sa_option == 1:
            msg["avg_degree"] = self.avg_degree
            
        self.message_pool[f"client_{self.client_id}"] = msg