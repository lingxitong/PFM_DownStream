import os
import pandas as pd
import numpy as np
import argparse

def get_few_shot_csv(args):
    few_shot_dirs = args.model_few_shot_dir_list
    model_names = args.model_name_list
    shots = args.shot_list
    save_dict = {}
    for model_name,log_dir in zip(model_names,few_shot_dirs):
        for shot in shots:
            shot_csv_path = os.path.join(log_dir,f'results_{shot}_shot_episodes.csv')
            df = pd.read_csv(shot_csv_path)
            bacc_100 = np.array(df[f'Kw{shot}s_bacc'].to_list())
            save_dict[f'{model_name}_name_{shot}shot_bacc'] = bacc_100
    save_df = pd.DataFrame(save_dict)
    save_df.to_csv(args.few_shot_csv_save_path,index=False)
    print(f'Saved few shot csv at {args.few_shot_csv_save_path}')
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--few_shot_csv_save_path', default='', type=str)
    argparse.add_argument('--shot_list', default=[1,2,4,8,16,32,64,128,256], type=list)
    argparse.add_argument('--model_name_list', default=['Model_Name1','Model_Name2'], type=list)
    argparse.add_argument('--model_few_shot_dir_list', default=['Dir1','Dir2'], type=list)
    args = argparser.parse_args()
    get_few_shot_csv(args)