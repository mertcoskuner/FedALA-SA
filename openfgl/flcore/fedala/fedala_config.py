"""
FedALA Configuration
Configuration parameters for Federated Adaptive Local Aggregation (FedALA) algorithm
"""

config = {
    # ALA (Adaptive Local Aggregation) parameters
    "ala_lr": 1e-3,
    "ala_init": 0.5,
    
    # FedALA-SA (Structure-Aware) parameters
    "use_structure_aware": True,  
    "topo_loss_lambda": 0.1,      # Topoloji kaybının etkisi
    "topo_anchor_threshold": 0.8, # Anchor düğüm belirleme eşiği
}