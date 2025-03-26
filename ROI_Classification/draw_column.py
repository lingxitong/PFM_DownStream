import matplotlib.pyplot as plt
import numpy as np
import argparse

def draw_column(args,data):
    models = args.model_name_list
    colors = ['#E0C05A', '#D0B6F3', '#6D9D7A', '#99C6D2', '#F1A7C2']
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.15  
    x = np.arange(len(data))  
    for i, (dataset, roi, means, stds) in enumerate(data):
        for j in range(5):  
            ax.bar(x[i] + (j - 2) * width, means[j], width, yerr=stds[j], capsize=2, color=colors[j], edgecolor=(0,0,0,0.2), error_kw=dict(ecolor=(0,0,0,0.45), lw=1))

    ax.set_ylabel('Supervised performance')
    model_means = np.zeros(len(models))
    for i in range(len(models)):
        model_means[i] = np.mean([data[j][2][i] for j in range(len(data))])
    for i in range(len(models)):
        ax.axhline(y=model_means[i], color=colors[i], linestyle='--', linewidth=1, label=f'{models[i]} mean')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dataset}\n({roi})" for dataset, roi, _, _ in data], rotation=45, ha="right")
    ax.set_ylim(20, 100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    # ax.legend(handles, models)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(handles, models, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.tight_layout()
    plt.savefig(f'{args.colunm_plots_save_path}')


if __name__ == '__main__':
    draw_data = [
        ["CRC tissue class.", "CRC-100K-NORM, n = 7,180 ROIs", [94.38, 94.35, 94.97, 94.93, 94.37], [0, 0, 0, 0, 0]],
        ["CRC tissue class.", "CRC-100K-RAW, n = 7,180 ROIs", [82.83, 88.63, 94.53, 83.97, 85.06], [0, 0, 0, 0, 0]],
        ["CRC MSI screening.", "TCGA, n = 32,361 ROIs", [72.00, 70.37, 69.83, 68.70, 70.73], [0, 0, 0, 0, 0]],
        ["CRC polyp class.", "UniToPatho, n = 2,399 ROIs", [42.97, 50.29, 49.27, 51.21, 54.92], [0, 0, 0, 0, 0]],
        ["ESCA subtyping.", "TCGA + Ext., n = 178,187 ROIs", [72.97, 83.22, 78.01, 79.54, 75.06], [0, 0, 0, 0, 0]],
        ["CRC polyp class.", "MHIST, n = 977 ROIs", [69.15, 82.85, 78.32, 78.85, 82.67], [0, 0, 0, 0, 0]],
        ["CRC tumor screening.", "CAMEL, n = 4,621 ROIs", [86.846, 91.479, 90.563, 87.146, 91.94], [0, 0, 0, 0, 0]],
        ["Stomach lesion class.", "STLC, n = 22,665 ROIs", [77.09, 84.61, 86.32, 65.45, 89.23], [0, 0, 0, 0, 0]],
        ["Stomach tumor screening.", "n = 22,665 ROIs", [85, 86, 87, 88, 89], [0, 0, 0, 0, 0]],
        ["Stomach ESD seg.", "n = 22,665 ROIs", [70.80, 73.86, 73.64, 67.97, 73.94], [0, 0, 0, 0, 0]],]
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--colunm_plots_save_path', default='', type=str)
    argparser.add_argument('--model_name_list', default=['Model_Name1','Model_Name2','Model_Name3'], type=list)
    args = argparser.parse_args()
    draw_column(args,draw_data)