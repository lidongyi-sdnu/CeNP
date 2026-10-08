from utils.io import read_patch_data
from utils.core import PseudoBag

import os
import os.path as osp
import numpy as np
# from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt

# 定义路径
input_dir = "/home/student5/data8t/WSI_cls/temp_train_6/"  # 原包文件夹路径
output_base_dir = "/home/student5/data8t/WSI_cls/pseudo_bags_6/"  # 伪包存储的根目录

# 创建伪包存储的根目录
# if os.path.exists(output_base_dir):
#     import shutil
#     shutil.rmtree(output_base_dir, ignore_errors=True)

os.makedirs(output_base_dir, exist_ok=True)

NUM_CLUSTER = 8 # the number of clusters
NUM_PSEB = 30 # the number of pseudo-bags
NUM_FT = 8 # # fine-tuning times

pt_files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
total_files = len(pt_files)  # 文件总数

for idx, pt_file in enumerate(pt_files, start=1):
    print(f"[info] Processing file {idx}/{total_files}: {pt_file}")
    if pt_file.endswith(".pt"):
        full_path = os.path.join(input_dir, pt_file)

        # 创建以原包命名的输出文件夹
        bag_name = os.path.splitext(pt_file)[0]  # 去除 .pt 后缀
        output_dir = os.path.join(output_base_dir, bag_name)

        if os.path.exists(output_dir):
            print(f"[skip] Output directory {output_dir} already exists. Skipping.")
            continue  # 目录存在则跳过

        os.makedirs(output_dir, exist_ok=True)
        # load WSI features
        bag_feats = read_patch_data(full_path, dtype='torch').to(torch.float)
        print(f"[info] Instance number = {bag_feats.shape[0]}; feature dim = {bag_feats.shape[1]}.")
        PB = PseudoBag(NUM_PSEB, NUM_CLUSTER, clustering_method='ProtoDiv', proto_method= 'mean', pheno_cut_method= 'quantile',
                       iter_fine_tuning=NUM_FT)
        PB.ptype = torch.mean(bag_feats, dim=0)
        print(PB.ptype)
        # normalize the bag prototype
        PB.norm_ptype = F.normalize(PB.ptype, p=2, dim=-1)

        # calculate the distance of all instances from the bag prototype (cosine distance)
        dis, limits = PB.protodiv_measure_distance(bag_feats)
        print(type(dis))
        print(dis)
        print(limits)
        # label_pseudo_bag: the indicator of pseudo-bags
        # pseudo_bags: a list of each pseudo-bag's features
        label_pseudo_bag, pseudo_bags = PB.divide(bag_feats, ptype=None, ret_pseudo_bag=True)

        for i, pseudo_bag in enumerate(pseudo_bags):
            output_path = os.path.join(output_dir, f"pseudo_bag_{i+1}.pt")
            torch.save(pseudo_bag, output_path)
            print(f"Saved pseudo-bag {i} to {output_path}")

        print(f"All pseudo-bags saved to {output_dir}.")

