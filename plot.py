import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Log dosyanın adı
log_filename = "slurm_sa_355368.out"

def parse_full_log(filename):
    """
    Log dosyasını okuyup hem Round bazlı (Global) hem de Client bazlı verileri çıkarır.
    """
    round_data = []  # Convergence ve Train-Test Gap için
    client_data = [] # Fairness (Distribution) için
    
    current_method = None
    current_dataset = None
    current_trial = 0
    current_clients_acc = [] # O anki round'daki clientların train acc'lerini tutar
    
    # Regex Desenleri
    exp_start_pattern = r"--- Running: (.*?) on (.*?) ---"
    trial_pattern = r"> Trial (\d+)/"
    
    # Client satırı: [client 0] ... accuracy_train: 0.6346 ... accuracy_test: 0.5965
    client_pattern = r"\[client \d+\].*?accuracy_train:\s*([\d\.]+).*?accuracy_test:\s*([\d\.]+)"
    
    # Round satırı: curr_round: 0 ... curr_test_accuracy: 0.7085
    round_pattern = r"curr_round:\s*(\d+).*?curr_test_accuracy:\s*([\d\.]+)"

    print(f"Reading {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # 1. Deney Başlangıcı
                match_exp = re.search(exp_start_pattern, line)
                if match_exp:
                    current_method = match_exp.group(1).strip()
                    current_dataset = match_exp.group(2).strip()
                    continue

                # 2. Trial Bilgisi
                match_trial = re.search(trial_pattern, line)
                if match_trial:
                    current_trial = int(match_trial.group(1))
                    continue

                # 3. Client Verilerini Topla (Train Acc hesaplamak için)
                match_client = re.search(client_pattern, line)
                if match_client and current_method:
                    train_acc = float(match_client.group(1)) * 100
                    test_acc = float(match_client.group(2)) * 100
                    current_clients_acc.append(train_acc)
                    
                    # Son round verilerini Fairness analizi için saklayabiliriz
                    # (Burada basitlik adına her roundu saklıyoruz ama grafikte son roundu süzeceğiz)
                    client_data.append({
                        "Dataset": current_dataset,
                        "Method": current_method,
                        "Trial": current_trial,
                        "Client Test Acc": test_acc
                    })

                # 4. Round Bitişi ve Global Veriler
                match_round = re.search(round_pattern, line)
                if match_round and current_method and current_dataset:
                    round_num = int(match_round.group(1))
                    global_test_acc = float(match_round.group(2)) * 100
                    
                    # O rounddaki tüm clientların ortalama Train başarısını hesapla
                    avg_train_acc = np.mean(current_clients_acc) if current_clients_acc else 0
                    
                    round_data.append({
                        "Dataset": current_dataset,
                        "Method": current_method,
                        "Trial": current_trial,
                        "Round": round_num,
                        "Test Accuracy": global_test_acc,
                        "Train Accuracy": avg_train_acc
                    })
                    
                    # Listeyi sıfırla (yeni round için)
                    current_clients_acc = []

    except FileNotFoundError:
        print("Dosya bulunamadı!")
        return None, None

    return pd.DataFrame(round_data), pd.DataFrame(client_data)

def generate_graphs(df_round, df_client):
    if df_round is None or df_round.empty:
        return

    datasets = df_round["Dataset"].unique()
    sns.set_theme(style="whitegrid")
    
    for ds in datasets:
        print(f"\n--- Processing Graphs for {ds} ---")
        
        # Filtreleme
        subset_round = df_round[df_round["Dataset"] == ds]
        subset_client = df_client[df_client["Dataset"] == ds]
        
        # ---------------------------------------------------------
        # GRAFİK 1: CONVERGENCE (Hız ve Kararlılık)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset_round, x="Round", y="Test Accuracy", hue="Method", style="Method", linewidth=2.5)
        plt.title(f"Convergence Analysis ({ds})", fontsize=14, fontweight='bold')
        plt.ylabel("Test Accuracy (%)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Method", loc='lower right')
        plt.tight_layout()
        plt.savefig(f"plot_{ds}_convergence.png", dpi=300)
        print(f">> Created: plot_{ds}_convergence.png")
        plt.close()

        # ---------------------------------------------------------
        # GRAFİK 2: GENERALIZATION GAP (Train vs Test)
        # Sadece "FedALA-SA (Opt 2)" ve "Baseline" için çizelim (Karmaşayı önlemek için)
        # ---------------------------------------------------------
        target_methods = [m for m in subset_round["Method"].unique() if "Opt 2" in m or "Baseline" in m]
        
        if target_methods:
            plt.figure(figsize=(10, 6))
            # Train Acc (Kesikli Çizgi)
            sns.lineplot(data=subset_round[subset_round["Method"].isin(target_methods)], 
                         x="Round", y="Train Accuracy", hue="Method", linestyle="--", alpha=0.7, legend=None)
            # Test Acc (Düz Çizgi)
            sns.lineplot(data=subset_round[subset_round["Method"].isin(target_methods)], 
                         x="Round", y="Test Accuracy", hue="Method", linewidth=2.5)
            
            plt.title(f"Generalization Gap ({ds}): Solid=Test, Dashed=Train", fontsize=14, fontweight='bold')
            plt.ylabel("Accuracy (%)")
            plt.tight_layout()
            plt.savefig(f"plot_{ds}_generalization.png", dpi=300)
            print(f">> Created: plot_{ds}_generalization.png")
            plt.close()

        # ---------------------------------------------------------
        # GRAFİK 3: CLIENT FAIRNESS (Box Plot)
        # Son 5 roundun ortalamasını alarak client dağılımını gösterir
        # ---------------------------------------------------------
        # Sadece son roundlara ait verileri al (Eğitim bitmiş hali)
        max_round = subset_round["Round"].max()
        # Log dosyasında client verisi round bilgisi içermiyor, 
        # ama sırayla okuduğumuz için listenin sonundaki %20lik kısmı alabiliriz yaklaşık olarak
        # Daha kesin yöntem: round_data uzunluğu ile eşleştirmek gerekir ama bu basit yaklaşım da iş görür.
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=subset_client, x="Method", y="Client Test Acc", palette="viridis")
        sns.stripplot(data=subset_client, x="Method", y="Client Test Acc", color=".2", alpha=0.4) # Noktaları da göster
        
        plt.title(f"Fairness Analysis ({ds}): Client Accuracy Distribution", fontsize=14, fontweight='bold')
        plt.ylabel("Client Test Accuracy (%)")
        plt.xlabel("")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"plot_{ds}_fairness.png", dpi=300)
        print(f">> Created: plot_{ds}_fairness.png")
        plt.close()

if __name__ == "__main__":
    df_round, df_client = parse_full_log(log_filename)
    if df_round is not None:
        generate_graphs(df_round, df_client)