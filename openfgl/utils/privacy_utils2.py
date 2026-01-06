import numpy as np
import torch

def clip_gradients(net: torch.nn.Module, loss_train: torch.Tensor, num_train: int, dp_mech: str, grad_clip: float):
    # Loss skaler ise (tek sayı)
    if loss_train.dim() == 0:
        loss_train.backward()
        # Global Norm Clipping (Safe Mode)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        return

    # Loss vektör ise (Per-sample - Reduction='none' modu)
    clipped_grads = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
    
    for iter in range(num_train):
        net.zero_grad()
        loss_train[iter].backward(retain_graph=True)
        
        if dp_mech == 'laplace':
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip, norm_type=1)
        elif dp_mech == 'gaussian':
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip, norm_type=2)
            
        for name, param in net.named_parameters():
            if param.grad is not None:
                clipped_grads[name] += param.grad

    net.zero_grad()
    for name, param in net.named_parameters():
        if param.requires_grad:
            clipped_grads[name] /= num_train
            param.grad = clipped_grads[name]

def Laplace_noise(args, dataset_size: int, x: torch.FloatTensor, sensitivity_multiplier=1.0):
    client_frac = getattr(args, 'client_frac', 0.1)
    times = args.num_rounds * client_frac
    each_query_eps = args.dp_eps / times
    sensitivity = 2 * args.lr * args.grad_clip / dataset_size
    sensitivity *= sensitivity_multiplier
    scale = sensitivity / each_query_eps
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=scale, size=x.shape)).to(x.device)
    return noise

def Gaussian_noise(args, dataset_size: int, x: torch.FloatTensor, sensitivity_multiplier=1.0):
    client_frac = getattr(args, 'client_frac', 0.1)
    times = args.num_rounds * client_frac
    each_query_eps = args.dp_eps / times
    
    # [DÜZELTME BURADA] dp_delta yoksa varsayılan 1e-5 kullan
    delta_val = getattr(args, 'dp_delta', 1e-5)
    each_query_delta = delta_val / times
    
    sensitivity = 2 * args.lr * args.grad_clip / dataset_size
    sensitivity *= sensitivity_multiplier
    
    scale = sensitivity * np.sqrt(2 * np.log(1.25 / each_query_delta)) / each_query_eps
    noise = torch.from_numpy(np.random.normal(loc=0, scale=scale, size=x.shape)).to(x.device)
    return noise

def add_noise(args, net: torch.nn.Module, dataset_size: int, avg_degree=None):
    sensitivity_multiplier = 1.0
    if avg_degree is not None and getattr(args, 'use_structure_aware', False):
        sensitivity_multiplier = 1.0 + np.log(1.0 + avg_degree) * 0.1
    
    with torch.no_grad():
        if args.dp_mech == 'laplace':
            for param in net.parameters():
                if param.requires_grad:
                    noise = Laplace_noise(args, dataset_size, param.data, sensitivity_multiplier)
                    param.data.add_(noise)
        elif args.dp_mech == 'gaussian':
            for param in net.parameters():
                if param.requires_grad:
                    noise = Gaussian_noise(args, dataset_size, param.data, sensitivity_multiplier)
                    param.data.add_(noise)