import csv
import glob, os
from plotting import compare_performance_plots

def save_performance(name: str, loss_data: list, recon_data: list, kl_data: list):
    with open(f'{name}.csv', 'w', newline='') as f:
        w = csv.writer(f, delimiter=';')
        for i in range(len(loss_data)):
            w.writerow([loss_data[i], recon_data[i], kl_data[i]])

def load_performance(path: str):
    loss_data, recon_data, kl_data = [], [], []
    for file in os.listdir(f'{path}/performance'):
        if file.endswith(".csv"):
            with open(f'{path}/performance/{file}', 'r') as f:
                r = csv.reader(f, delimiter=';')
                for row in r:
                    loss_data.append(float(row[0]))
                    recon_data.append(float(row[1]))
                    kl_data.append(float(row[2]))
    return loss_data, recon_data, kl_data

def create_comparison(base_path, names: list[str]):
    loss_data_arr, recon_data_arr, kl_data_arr = [], [], []
    for name in names:
        data = load_performance(f'{base_path}/{name}')
        loss_data_arr.append(data[0])
        recon_data_arr.append(data[1])
        kl_data_arr.append(data[2])
    compare_performance_plots(loss_data_arr, recon_data_arr, kl_data_arr, names, base_path, f'{base_path}/comparison_ld_7.png')

if __name__ == "__main__":
    base_path = 'saves/CCVAE/permutation'
    create_comparison(base_path, ['hyperparam_test_lr_0.001_hd_512_ld_7', 'hyperparam_test_lr_0.001_hd_1024_ld_7', 'hyperparam_test_lr_0.0005_hd_1024_ld_7', 'hyperparam_test_lr_0.0005_hd_512_ld_7'])