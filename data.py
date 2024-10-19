import torch
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
# import mxnet as mx

# class mxImageNetDataset(Dataset):
#     def __init__(self,
#                  rec_file,
#                  transform=None,
#                  to_rgb=True):
#         assert transform is not None
#         self.transform = transform
#         self.to_rgb = to_rgb

#         self.record = mx.recordio.MXIndexedRecordIO(
#             rec_file + '.idx', rec_file + '.rec', 'r')
#         self.rec_index = list(sorted(self.record.keys))

#         self.reckey2info = dict()
#         index_file = rec_file + '.index'
#         with open(index_file) as f:
#             lines = f.readlines()
#         for line in lines:
#             split_parts = line.strip().split("\t")
#             reckey, label, cls_name = split_parts[0], split_parts[2], split_parts[3]
#             self.reckey2info[int(reckey)] = [label, cls_name]

#         print("#images: ", len(self.rec_index), self.rec_index[:5])

#     def __getitem__(self, idx):
#         key = self.rec_index[idx]
#         img = self.record.read_idx(int(key))
#         head, im = mx.recordio.unpack_img(img)  # NOTE: BGR
#         cls = head.label  # label in rec is numpy array.

#         if self.to_rgb:
#             im = im[:, :, ::-1]
#         im = Image.fromarray(im)
#         im = self.transform(im)

#         return im, int(cls)

#     def __len__(self):
#         return len(self.rec_index)
    
class ImageNetPrepareBAEDatasetOld(Dataset):
    def __init__(self, bin_dir, split='train'):
        binary_path = os.path.join(bin_dir, split+'_ids.bin')
        labels_path = os.path.join(bin_dir, split+'_lbs.bin')
        
        self.binary = np.memmap(binary_path, dtype=np.uint8, mode='r')
        self.labels = np.memmap(labels_path, dtype=np.uint16, mode='r')

        self.binary = np.reshape(self.binary, (-1, 64, 16, 16))

    def __getitem__(self, idx):
        bin_i = torch.from_numpy(self.binary[idx].astype(np.float32)).float()
        lb_i = int(self.labels[idx])

        return bin_i, lb_i

    def __len__(self):
        return self.labels.shape[0]
    
class ImageNetPrepareBAEDataset(Dataset):
    def __init__(self, bin_dir, codebook, latent_res=16, split='train'):
        self.binary_path = os.path.join(bin_dir, split+'_ids.bin')
        self.labels_path = os.path.join(bin_dir, split+'_lbs.bin')

        self.length = np.memmap(self.labels_path, dtype=np.uint16, mode='r').shape[0]
        self.codebook = codebook
        self.latent_res = latent_res
        
    def __getitem__(self, idx):
        elem = self.codebook*self.latent_res*self.latent_res
        
        binary = np.fromfile(self.binary_path, dtype=np.uint8, count=elem, offset=elem*idx)
        labels = np.fromfile(self.labels_path, dtype=np.uint16, count=1, offset=2*idx)

        binary = np.reshape(binary, (self.codebook, self.latent_res, self.latent_res))

        bin_i = torch.from_numpy(binary.astype(np.float32)).float()
        lb_i = int(labels)

        return bin_i, lb_i

    def __len__(self):
        return self.length
    
if __name__ == '__main__':
    ds_old = ImageNetPrepareBAEDatasetOld('../imagenet_data_prepare/imagenet-bae-ema-64')
    ds_new = ImageNetPrepareBAEDataset('../imagenet_data_prepare/imagenet-bae-ema-64')
    
    print(len(ds_old))
    print(len(ds_new))
    
    for i, data in enumerate(range(len(ds_old))):
        data_old = ds_old[i][0]
        label_old = ds_old[i][1]
        
        data_new = ds_new[i][0]
        label_new = ds_new[i][1]
        
        if not ((label_old == label_new) and (data_old == data_new).all()):
            print("Something wrong!!!")
            break
        
        