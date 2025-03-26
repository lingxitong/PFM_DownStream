import numpy as np
from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils.get_encoder import get_pathology_encoder
from utils.get_transforms import get_transforms
from utils.extract_patch_features import extract_patch_features_from_dataloader
from utils.common_utils import load_class2id_mapping
from utils.roi_dataset import TSNE_ROIDataSet

# Assuming you have a function to get the encoded images and their labels
def get_encoded_images_and_labels(pathology_model,dataset_csv,transform,device,class2id_txt):
    roi_dataset = TSNE_ROIDataSet(dataset_csv, transform['val'],class2id_txt)
    roi_loader = torch.utils.data.DataLoader(roi_dataset,batch_size=256,shuffle=True,num_workers=8,collate_fn=TSNE_ROIDataSet.collate_fn)
    roi_assert = extract_patch_features_from_dataloader(pathology_model,roi_loader,device)
    roi_feats = torch.Tensor(roi_assert['embeddings'])
    roi_labels = torch.Tensor(roi_assert['labels']).type(torch.long)
    
    return roi_feats,roi_labels

def visualize_tsne(args):
    transform = get_transforms(args.resize_size)
    model = get_pathology_encoder(args.model_name)
    model = model.to(args.device)
    roi_feats, roi_labels = get_encoded_images_and_labels(model, args.dataset_csv, args.domain, transform, args.device, args.class2id_txt)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(roi_feats)
    plt.figure(figsize=(10, 8))
    dataset_name = args.dataset_name
    plt.title(f't-SNE for {args.model_name} on {dataset_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    class2id_mapping = load_class2id_mapping(args.class2id_txt)
    id2class_mapping = {v: k for k, v in class2id_mapping.items()}
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(id2class_mapping))))
    norm = BoundaryNorm(np.arange(len(id2class_mapping) + 1) - 0.5, len(id2class_mapping))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=roi_labels, cmap=cmap, norm=norm, s=5, alpha=0.6)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(len(id2class_mapping))]
    labels = [id2class_mapping[i] for i in range(len(id2class_mapping))]
    plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-SNE Visualization of Encoded Images')
    parser.add_argument('--model_name', type=str, default='', help='Name of the model to use for encoding images')
    parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset to use for encoding images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for encoding images')
    parser.add_argument('--resize_size', type=int, default=224, help='Size to resize images to before encoding')
    parser.add_argument('--dataset_csv', type=str, default='datasets/CRC-MSI-TSNE.csv' ,help='Path to the dataset CSV file')
    parser.add_argument('--class2id_txt', type=str, default='datasets/CRC-MSI.txt', help='Path to the class2id TXT file')
    parser.add_argument('--save_path', type=str,default='/path/to/your/tsne.png',help='Path to save the t-SNE plot')
    args = parser.parse_args()
    visualize_tsne(args)
    