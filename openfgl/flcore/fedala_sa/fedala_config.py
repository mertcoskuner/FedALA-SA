"""
FedALA Configuration
Configuration parameters for Federated Adaptive Local Aggregation (FedALA) algorithm
"""

config = {
   
    "ala_lr": 1e-3,
    "ala_init": 0.5,
    
    "fedala_sa_option": 2,       
    "fedala_sa_mu": 0.01,       
    
    "use_structure_aware": True, 
    
   
    "dp_mech": "no_dp",        
    
   
    "grad_clip": 1.0,          
    "dp_eps": 5.0,               
    "dp_delta": 1e-5,            
    "noise_multiplier": 0.1,     
    
    
    "topo_loss_lambda": 0.1,      
    "topo_anchor_threshold": 0.8, 
}