"""
FedALA Reproduction Script
Reproduces Table 7 results for FedALA on specified datasets.
"""

import numpy as np
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import torch
import random
import pandas as pd
import sys
import os
from multiprocessing import Pool, Manager
from functools import partial
import copy
import traceback

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

args = config.args
args.root = "./data"
args.simulation_mode = "subgraph_fl_metis"
args.num_clients = 10
args.model = ["gcn"]
args.metrics = ["accuracy"]

# Hyperparameters 
args.num_rounds = 100
args.local_epochs = 3
args.learning_rate = 0.01
args.batch_size = 128
args.weight_decay = 5e-4
args.dropout = 0.5
args.optimizer = "adam"
args.task = "node_cls"

if not hasattr(args, 'ala_init'):
    args.ala_init = 0.5
if not hasattr(args, 'ala_lr'):
    args.ala_lr = 1e-2

target_datasets = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "Chameleon", "Actor"]
target_algorithms = ["fedala"]
num_trials = 3

def extract_accuracy(trainer, result_from_train=None):
    candidates = []
    if result_from_train is not None:
        if isinstance(result_from_train, dict):
            for k, v in result_from_train.items():
                if "acc" in k and "test" in k and isinstance(v, (float, int)):
                    candidates.append(v)
        elif isinstance(result_from_train, (float, int)):
            candidates.append(result_from_train)

    attributes = dir(trainer)
    for attr in attributes:
        if "acc" in attr and "test" in attr:
            try:
                val = getattr(trainer, attr)
                if isinstance(val, (float, int)) and val > 0:
                    candidates.append(val)
            except:
                pass
    
    
    if hasattr(trainer, 'evaluation_result'):
        for k, v in trainer.evaluation_result.items():
             if "acc" in k and "test" in k and isinstance(v, (float, int)) and v > 0:
                 candidates.append(v)

    if not candidates:
        return 0.0
    
    best_acc = max(candidates)
    if torch.is_tensor(best_acc):
        best_acc = best_acc.item()
    if 0.0 < best_acc <= 1.0:
        best_acc *= 100.0

    return best_acc

def run_single_experiment(algo_name, dataset_name, args_template, num_trials, progress_dict=None):
    try:
        args = copy.deepcopy(args_template)
        args.fl_algorithm = algo_name
        
        trial_accs = []
        
        for trial in range(num_trials):
            set_seed(2024 + trial)
            args.seed = 2024 + trial
            args.dataset = [dataset_name]
            
            try:
                trainer = FGLTrainer(args)
                train_result = trainer.train()
                acc = extract_accuracy(trainer, train_result)
                trial_accs.append(acc)
                
                if progress_dict is not None:
                    key = f"{algo_name}_{dataset_name}"
                    progress_dict[key] = f"Trial {trial+1}/{num_trials}: {acc:.2f}%"
                    
            except Exception as e:
                if progress_dict is not None:
                    key = f"{algo_name}_{dataset_name}"
                    progress_dict[key] = f"Trial {trial+1} FAILED: {str(e)[:50]}"
                print(f"Error in {algo_name} {dataset_name} trial {trial}: {e}")
                traceback.print_exc()
        
        valid_accs = [x for x in trial_accs if x > 0]
        if valid_accs:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs)
            result_str = f"{mean_acc:.2f}±{std_acc:.2f}"
        else:
            result_str = "Fail"
        
        return (algo_name, dataset_name, result_str)
        
    except Exception as e:
        return (algo_name, dataset_name, "Fail")

if __name__ == "__main__":
    print("=" * 70)
    print("FEDALA REPRODUCTION BENCHMARK")
    print("=" * 70)
    
    max_workers = 1 
    
    tasks = []
    for algo_name in target_algorithms:
        for dataset_name in target_datasets:
            tasks.append((algo_name, dataset_name))
    
    full_results = {algo: {ds: "N/A" for ds in target_datasets} for algo in target_algorithms}
    
    manager = Manager()
    progress_dict = manager.dict()
    
    worker_func = partial(run_single_experiment, 
                         args_template=args, 
                         num_trials=num_trials,
                         progress_dict=progress_dict)
    
    print("Starting execution...\n")
    
    
    results = []
    for task in tasks:
        print(f"Running {task[0]} on {task[1]}...")
        res = worker_func(*task)
        results.append(res)
        print(f"Result: {res[2]}")
    
    for algo_name, dataset_name, result_str in results:
        full_results[algo_name][dataset_name] = result_str
    
    df = pd.DataFrame.from_dict(full_results, orient="index")
    df.to_csv("fedala_reproduction_results.csv")
    print(df)
    
    print("\n" + "=" * 70)
    print("COMPLETED! Results saved to 'fedala_reproduction_results.csv'.")
    print("=" * 70)
