import torch
import numpy as np
import transforms
import time
import preprocess
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils
import models
import sys
import densenet
import densenet2
# from torch.nn import init

in_z = 0
# volpath = '/home/alex/LiverSegmentation/5mm_train_3'
# segpath = '/home/alex/LiverSegmentation/5mm_train_3'
volpath = '/home/alex/samsung_512/CMR_PC/folds/fold_5/train'
segpath = '/home/alex/samsung_512/CMR_PC/folds/fold_5/train'
batch_size = 10
t_batch_size = 4
sqr = transforms.Square()
pad = transforms.Pad(36)
aff = transforms.Affine()
crop = transforms.RandomCrop(224)
scale = transforms.Scale(256)
rotate = transforms.Rotate(0.5, 30)
noise = transforms.Noise(0.02)
flip = transforms.Flip()
orig_dim = 256
transform_plan = [sqr, scale, aff, rotate, crop, flip, noise]
lr = 1e-4
initial_depth = in_z*2+1
series_names = ['Mag']
seg_series_names = ['AV']

#todo
#add different type of scaling
#add depth padding
# model = models.Net23(2)
# model = densenet.densenet121()
model = densenet2.densenet121()
model.cuda()
# model.load_state_dict(torch.load(\
        # '/home/alex/samsung_512/CMR_PC/checkpoint.pth'))
# model.load_state_dict(torch.load(\
        # '/home/alex/samsung_512/CMR_PC/checkpoint_dense.pth'))

optimizer = optim.RMSprop(model.parameters(), lr=lr)

out_z, center_crop_sz = utils.get_out_size(orig_dim, in_z,\
        transform_plan, model)
center = transforms.CenterCrop(center_crop_sz)
depth_center = transforms.DepthCenterCrop(out_z)

# t_transform_plan = []
# t_transform_plan = transform_plan
# t_out_z, t_center_crop_sz = utils.get_out_size(orig_dim, in_z,\
        # t_transform_plan, model)
# t_center = transforms.CenterCrop(t_center_crop_sz)
# t_depth_center = transforms.DepthCenterCrop(t_out_z)

t0 = time.time()

# weight = torch.FloatTensor(preprocess.get_weights(volpath, segpath,
    # 8, clip_val=1, nrrd=False)).cuda()

# print(weight)
# print('generating weights took {:.2f} seconds'.format(time.time()-t0))
counter = 0
print_interval = 400
model.train()
for i in range(1):
    # model.train()
    weight = torch.FloatTensor([0.2,0.8]).cuda()
    # weight = torch.FloatTensor([0.0012, 0.0525, 0.9463]).cuda()
    vol, seg, inds = preprocess.get_batch(volpath, segpath, batch_size, in_z,\
            out_z, center_crop_sz, series_names, seg_series_names,\
            transform_plan, 8, nrrd=True)
