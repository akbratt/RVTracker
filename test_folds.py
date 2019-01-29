import torch
import numpy as np
import transforms
import preprocess
import utils
import models
# import pickle
import re
# import PyQt5.QtCore as QtCore
# import pyqtgraph as pg
# import vol_display
# import matplotlib.pyplot as plt
# import densenet
import os
# import densenet2

in_z = 0


folds_folder = '/home/alex/samsung_512/CMR_PC/folds'
out_file = '/home/alex/samsung_512/CMR_PC/folds_fakes'
double_vol = False

batch_size = 5
t_batch_size = 1
pad = transforms.Pad(2)
sqr = transforms.Square()
center = transforms.CenterCrop2(224)
crop = transforms.RandomCrop(256)
scale = transforms.Scale(256)
orig_dim = 256
transform_plan = [sqr, scale, center]
lr = 1e-5
initial_depth = in_z*2+1
num_labels=2
num_labels_final = 2
series_names = ['Mag']
seg_series_names = ['AV']


model = models.Net23(2)
# model = densenet2.densenet121(True)
# model = densenet.densenet121()
model.cuda()

# model.load_state_dict(torch.load(\
        # '/home/alex/samsung_512/CMR_PC/checkpoint.pth'))

# model = densenet.densenet121()
# model.cuda()

folds = os.listdir(folds_folder)
dice, jacc, haus, assd, times = [], [], [], [], []
for fold_i in folds:
    print('')
    print(fold_i)
    print('')
    test_volpath = os.path.join(folds_folder, fold_i + '/test')
    test_segpath = test_volpath

    model.load_state_dict(torch.load(\
            os.path.join(folds_folder, fold_i) + '/res.pth'))

    ind = 6

    f = preprocess.gen_filepaths(test_segpath)

    mult_inds = []
    for i in f:
        if 'volume' in i:
            mult_inds.append(int(re.findall('\d+', i)[0]))

    mult_inds = sorted(mult_inds)

# mult_inds = [3,6,12,19,25,49,54,56,58,62,64,66,71,75,78,80,85,86,90,93,\
            # 96,123,124,125,126,127,128,129,130]
# mult_inds = [27]
    mult_inds = np.unique(mult_inds)

    volpaths, segpaths = utils.get_paths(mult_inds, f, series_names, \
            seg_series_names, test_volpath, test_segpath)

    t_transform_plan = transform_plan

    d, j, h, a, t = utils.test_net_cheap(test_volpath, test_segpath, mult_inds, in_z, model,\
            t_transform_plan, orig_dim, batch_size, out_file, num_labels,\
            num_labels_final, volpaths, segpaths, nrrd=True,\
            vol_only=double_vol, get_dice=True, make_niis=False)

    dice.append(d)
    jacc.append(j)
    haus.append(h)
    assd.append(a)
    times.append(t)

print(len(dice))
dice = np.concatenate(dice)
jacc = np.concatenate(jacc)
haus = np.concatenate(haus)
assd = np.concatenate(assd)
times = np.concatenate(times)
out_csv = '/home/alex/samsung_512/CMR_PC/res_ref.csv'
np.savetxt(out_csv, np.stack([dice,jacc,haus,assd,times],1), fmt='%.2f')
print(utils.get_CI(dice))
print(utils.get_CI(jacc))
print(utils.get_CI(haus))
print(utils.get_CI(assd))
print(utils.get_CI(times))
