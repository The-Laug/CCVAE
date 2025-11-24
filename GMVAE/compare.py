import csv
from plots import compare_performance_plot

def save_performance(name: str, loss_data: list, recon_data: list, ki_data: list):
    with open('name.csv', 'w') as f:
        w = csv.writer(delimiter=',')
        w.writerow(["loss", "recon", "ki"])
        for i in range(len(loss_data)):
            w.writerow(loss_data[i], recon_data[i], ki_data[i])

def load_performance(name: str) -> list, list, list:
    with open(f'{name}.csv', 'r'):
        r = csv.reader(delimiter=',')
        for row in r:
            print(', '.join(row))

def create_comparison(names: list[str]):
    data = []
    for name in names:
        data.append((name, load_performance(name)))
    
    # create plots
    compare_performance_plot(data)
    ...