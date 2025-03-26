import os
import logging
import timm
import torch
import torch.nn as nn
import warnings 
from collections import OrderedDict


warnings.filterwarnings("ignore")
from torchvision import transforms
from .common_utils import create_DigePath_based_model

Name2Chkpt = {
'UNI':'UNI/pytorch_model.bin', # UNI
'Gigapath': 'GigaPath_weights/pytorch_model.bin', # Gigapath
'Virchow-v2': 'Virchow_2_weights/pytorch_model.bin', # Virchow-v2
'Conch-v1_5': 'Conch_1_5_weights/conch_v1_5_pytorch_model.bin', # Conch-v1.5
'Ctranspath': 'Ctranspath_weights/ctranspath.pth'} # Ctranspath
def get_pathology_encoder(model_name: str,num_classes: int = 0,trainable_head: bool = False):
    if model_name == 'UNI':
        uni_kwargs = {
        'model_name': 'vit_large_patch16_224',
        'img_size': 224, 
        'patch_size': 16, 
        'init_values': 1e-5, 
        'num_classes': num_classes, 
        'dynamic_img_size': True}
        model = timm.create_model(**uni_kwargs)
        state_dict = torch.load(Name2Chkpt[model_name], map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")
        print(f'--------successfully load UNI with {num_classes} classes head and {trainable_head} trainable head--------')
        return model
    elif model_name == 'Gigapath':
        gig_config = {
        "architecture": "vit_giant_patch14_dinov2",
        "num_classes": 0,
        "num_features": 1536,
        "global_pool": "token",
        "model_args": {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-05,
        "mlp_ratio": 5.33334,
        "num_classes": num_classes}} 
        model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **gig_config['model_args'])
        state_dict = torch.load(Name2Chkpt[model_name], map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    elif model_name == 'Virchow-v2':
        from timm.layers import SwiGLUPacked
        virchow_config = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "mlp_ratio": 5.3375,
        "reg_tokens": 4,
        "global_pool": "",
        "dynamic_img_size": True}
        model = timm.create_model("vit_huge_patch14_224", pretrained=False,mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,**virchow_config)
        state_dict = torch.load(Name2Chkpt[model_name], map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    elif model_name == 'Conch-v1_5':
        from .conch_v1_5_config import ConchConfig
        from .build_conch_v1_5 import build_conch_v1_5
        checkpoint_path = Name2Chkpt[model_name]
        conch_v1_5_config = ConchConfig()
        model = build_conch_v1_5(conch_v1_5_config, checkpoint_path)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    elif model_name == 'Ctranspath':
        from .ctrans import ctranspath
        checkpoint_path = Name2Chkpt[model_name]
        model = ctranspath()
        model.head = nn.Identity()
        state_dict = torch.load(checkpoint_path,weights_only=True)
        model.load_state_dict(state_dict['model'], strict=True)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    
