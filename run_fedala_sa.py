"""
FedALA-SA Final Benchmark Script + Auto Plotting
Runs the 3 critical scenarios defined for the paper:
1. Baseline (Standard FedALA)
2. Innovation (FedALA-SA Option 2)
3. Privacy (FedALA-SA + Differential Privacy)
+ Scalability Test (10 vs 20 Clients)
+ Generates Graphs automatically (Including Convergence)
"""

import numpy as np
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import torch
import random
import pandas as pd
import sys
import os
import copy
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Agg')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

args = config.args
args.root = "./data"
args.simulation_mode = "subgraph_fl_metis"
args.model = ["gcn"]
args.metrics = ["accuracy"]

args.num_rounds = 50
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


target_datasets = ["Cora"]

num_trials = 3

scenarios = [

    {
        "id": "1_Baseline_10",
        "name": "Baseline (Vanilla FedALA)",
        "option": 0,
        "structure_aware": False,
        "dp_mech": "no_dp",
        "clients": 10
    },
    {
        "id": "2_Opt1_Degree_10",
        "name": "FedALA-SA (Opt 1 - Degree)",
        "option": 1,
        "structure_aware": True,
        "dp_mech": "no_dp",
        "clients": 10
    },
    {
        "id": "3_Opt3_Layers_10",
        "name": "FedALA-SA (Opt 3 - Layers)",
        "option": 3,
        "structure_aware": True,
        "dp_mech": "no_dp",
        "clients": 10
    },
    {
        "id": "4_Opt2_Innovation_10",
        "name": "FedALA-SA (Opt 2 - Proximal)",
        "option": 2,
        "structure_aware": True,
        "dp_mech": "no_dp",
        "clients": 10
    },
    {
        "id": "5_Privacy_10",
        "name": "FedALA-SA + Privacy",
        "option": 2,
        "structure_aware": True,
        "dp_mech": "gaussian",
        "clients": 10
    },

   
    {
        "id": "6_Baseline_20",
        "name": "Baseline (Vanilla FedALA)",
        "option": 0,
        "structure_aware": False,
        "dp_mech": "no_dp",
        "clients": 20
    },
    {
        "id": "7_Opt1_Degree_20",
        "name": "FedALA-SA (Opt 1 - Degree)",
        "option": 1,
        "structure_aware": True,
        "dp_mech": "no_dp",
        "clients": 20
    },
    {
        "id": "8_Opt3_Layers_20",
        "name": "FedALA-SA (Opt 3 - Layers)",
        "option": 3,
        "structure_aware": True,
        "dp_mech": "no_dp",
        "clients": 20
    },
    {
        "id": "9_Innovation_20",
        "name": "FedALA-SA (Opt 2 - Proximal)",
        "option": 2,
        "structure_aware": True,
        "dp_mech": "no_dp",
        "clients": 20
    },
    {
        "id": "10_Privacy_20",
        "name": "FedALA-SA + Privacy",
        "option": 2,
        "structure_aware": True,
        "dp_mech": "gaussian",
        "clients": 20
    }
]

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

def run_single_experiment(scenario, dataset_name, args_template, num_trials):
    scenario_name = scenario["name"]
    print(f"\n--- Running: {scenario_name} on {dataset_name} ---")
    
    try:
        current_args = copy.deepcopy(args_template)
        current_args.fl_algorithm = "fedala_sa"
        current_args.dataset = [dataset_name]
        
        current_args.fedala_sa_option = scenario["option"]
        current_args.use_structure_aware = scenario["structure_aware"]
        current_args.dp_mech = scenario["dp_mech"]
        current_args.num_clients = scenario["clients"]
        
        current_args.fedala_sa_mu = 0.01
        current_args.grad_clip = 1.0
        current_args.dp_eps = 5.0
        current_args.noise_multiplier = 0.1
        current_args.topo_loss_lambda = 0.1
        
        trial_accs = []
        captured_history = [] 
        
        for trial in range(num_trials):
            seed_val = 2024 + trial
            set_seed(seed_val)
            current_args.seed = seed_val
            
            print(f"  > Trial {trial+1}/{num_trials} (Seed: {seed_val})...", end=" ", flush=True)
            
            try:
                trainer = FGLTrainer(current_args)
                train_result = trainer.train()
                acc = extract_accuracy(trainer, train_result)
                trial_accs.append(acc)
                print(f"Acc: {acc:.2f}%")
                
                
                if trial == 0:
                   
                    if hasattr(trainer, 'test_acc_track'):
                        captured_history = trainer.test_acc_track
                    elif hasattr(trainer, 'stats') and 'test_acc' in trainer.stats:
                        captured_history = trainer.stats['test_acc']
                    elif hasattr(trainer, 'metrics') and 'test_acc' in trainer.metrics:
                        captured_history = trainer.metrics['test_acc']
                # ---------------------------------------------
                
            except Exception as e:
                print(f"FAILED: {str(e)[:100]}")
                traceback.print_exc()
        
        valid_accs = [x for x in trial_accs if x > 0]
        if valid_accs:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs)
            result_str = f"{mean_acc:.2f}±{std_acc:.2f}"
            print(f"  >>> RESULT: {result_str}")
        else:
            result_str = "Fail"
            mean_acc = 0.0
            std_acc = 0.0
        
        return {
            "Scenario ID": scenario["id"],
            "Method": scenario_name,
            "Dataset": dataset_name,
            "Clients": scenario["clients"],
            "Option": scenario["option"],
            "Privacy": scenario["dp_mech"],
            "Result (Mean±Std)": result_str,
            "Mean Acc": mean_acc,
            "Std Acc": std_acc,
            "Acc History": captured_history
        }
        
    except Exception as e:
        print(f"Critical Error in {scenario_name}: {e}")
        return None



def generate_plots(df):
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY PLOTS...")
    
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
   
    try:
        df_10 = df[df["Clients"] == 10].copy()
        desired_order = ["Baseline (Vanilla FedALA)", "FedALA-SA (Opt 2 - Proximal)", "FedALA-SA + Privacy"]
        plot_order = [m for m in desired_order if m in df_10["Method"].unique()]
        
        plt.figure(figsize=(10, 6))
        bars = sns.barplot(data=df_10, x="Method", y="Mean Acc", order=plot_order, palette="viridis", capsize=.1)
        plt.ylim(min(df_10["Mean Acc"])-5, max(df_10["Mean Acc"])+5)
        plt.title("Ablation Study: Impact of Structure Awareness & Privacy", fontsize=14, fontweight='bold')
        plt.ylabel("Test Accuracy (%)")
        plt.xlabel("")
        plt.xticks(rotation=10)
        
        for p in bars.patches:
            height = p.get_height()
            if height > 0:
                plt.text(p.get_x() + p.get_width()/2., height + 0.5, f'{height:.2f}%', ha="center", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("fedala_sa_ablation_study.png", dpi=300)
        print(">> Saved: fedala_sa_ablation_study.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting Ablation: {e}")

def plot_learning_curve(dataset_name, history_dict):
    """
    Accuracy vs Round grafiğini çizer.
    """
    if not history_dict:
        return

    print(f"   > Plotting Convergence for {dataset_name}...")
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    palette = sns.color_palette("tab10", n_colors=len(history_dict))
    
    for i, (method_name, acc_list) in enumerate(history_dict.items()):
        # Eğer liste boşsa veya 0 ise atla
        if not acc_list or len(acc_list) < 2: 
            continue
            
        rounds = range(1, len(acc_list) + 1)
        sns.lineplot(
            x=rounds, 
            y=acc_list, 
            label=method_name, 
            linewidth=2.5,
            color=palette[i]
        )

    plt.title(f"Convergence ({dataset_name}): Accuracy vs. Rounds", fontsize=15, fontweight='bold')
    plt.ylabel("Test Accuracy (%)", fontweight='bold')
    plt.xlabel("Communication Rounds", fontweight='bold')
    plt.legend(title="Method", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    filename = f"fedala_sa_convergence_{dataset_name}.png"
    plt.savefig(filename, dpi=300)
    print(f"     Saved: {filename}")
    plt.close()

def plot_full_scalability():
    csv_path = "fedala_sa_final_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return

    sns.set_theme(style="whitegrid")
    
    def clean_name(row):
        name = row["Method"]
        if "Baseline" in name: return "Baseline"
        if "Opt 1" in name: return "Opt 1 (Degree)"
        if "Opt 2" in name and "Privacy" not in name: return "Opt 2 (Proximal)"
        if "Opt 3" in name: return "Opt 3 (Layers)"
        if "Privacy" in name: return "Opt 2 + Privacy"
        return name

    df["ShortName"] = df.apply(clean_name, axis=1)
    
    
    try:
        plt.figure(figsize=(10, 7))
        hue_order = ["Baseline", "Opt 1 (Degree)", "Opt 3 (Layers)", "Opt 2 (Proximal)", "Opt 2 + Privacy"]
        palette = {"Baseline": "gray", "Opt 1 (Degree)": "cornflowerblue", "Opt 3 (Layers)": "teal", "Opt 2 (Proximal)": "darkorange", "Opt 2 + Privacy": "purple"}
        
        sns.lineplot(data=df, x="Clients", y="Mean Acc", hue="ShortName", style="ShortName", palette=palette, hue_order=hue_order, markers=True, markersize=12, linewidth=3)
        plt.title("Full Scalability Analysis", fontsize=15, fontweight='bold')
        plt.ylabel("Test Accuracy (%)")
        plt.xticks([10, 20])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("graph_full_scalability.png", dpi=300)
        plt.close()
        print(">> Saved: graph_full_scalability.png")
    except Exception as e:
        print(f"Error scalability plot 1: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("FEDALA-SA FINAL BENCHMARK + PLOTTING (With Convergence Curves)")
    print("=" * 80)
    
    all_results = []
    

    convergence_data = {ds: {} for ds in target_datasets}
    
   
    for dataset_name in target_datasets:
        print(f"\nProcessing Dataset: {dataset_name}...")
        
        for scenario in scenarios:
            res = run_single_experiment(scenario, dataset_name, args, num_trials)
            
            if res:
                all_results.append(res)
                
              
                if res["Clients"] == 10: 
                    history = res.get("Acc History", [])
               
                    if history and isinstance(history, list) and len(history) > 0:
                        convergence_data[dataset_name][res["Method"]] = history
    
   
    if all_results:
        df = pd.DataFrame(all_results)
        
        
        df_clean = df.drop(columns=["Acc History"], errors='ignore')
        csv_filename = "fedala_sa_final_results.csv"
        df_clean.to_csv(csv_filename, index=False)
        
        print("\n" + "=" * 80)
        print(f"BENCHMARK COMPLETED. Results saved to {csv_filename}")
        
        
        print(df_clean[["Method", "Dataset", "Clients", "Result (Mean±Std)"]])
        
    
        print("\n--- Generating Summary Plots ---")
        generate_plots(df)
        plot_full_scalability()
            
        
        print("\n--- Generating Convergence Plots (Per Dataset) ---")
        for dataset_name, method_histories in convergence_data.items():
            if method_histories: 
                plot_learning_curve(dataset_name, method_histories)
        
        print("=" * 80)
    else:
        print("\nNo results obtained.")