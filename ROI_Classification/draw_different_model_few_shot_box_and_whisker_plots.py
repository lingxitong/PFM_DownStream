import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse


def draw_few_shot_performance(args):
    DS_NAME = args.dataset_name
    csv_path = args.few_shot_csv_save_path
    df = pd.read_csv(csv_path)
    df_long = pd.melt(df, var_name='Model_Shot', value_name='Accuracy')
    df_long[['Model', 'Shot']] = df_long['Model_Shot'].str.extract(r'([A-Za-z0-9\.\-]+)_name_(\d+)shot_bacc')
    df_long['Shot'] = df_long['Shot'].astype(int)
    colors = ['#E0C05A', '#D0B6F3', '#6D9D7A', '#99C6D2', '#F1A7C2']
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='Shot', y='Accuracy', hue='Model', data=df_long, palette=colors)
    plt.xlabel('Shot', fontsize=14)
    plt.ylabel('Balanced Accuracy', fontsize=14)
    sns.despine()
    plt.title(f'Few-shot Performance on {DS_NAME}', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.draw_save_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--few_shot_csv_save_path', default='', type=str)
    argparser.add_argument('--dataset_name', default='', type=str)
    argparse.add_argument('--draw_save_path',default='',type=str)

