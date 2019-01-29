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

in_z = 0

fold = 1
for fold in [0,1,2,3,4,5,6,7]:
    test_volpath = '/home/alex/projects/Beecy_Echo/nii_renamed'.format(\
            fold)
    test_segpath = '/home/alex/projects/Beecy_Echo/nrrd_folds/fold_{}/test'.format(\
            fold)
    out_file = '/home/alex/projects/Beecy_Echo/fakes'
    double_vol = False
    model = models.Net23(3)
    model.cuda()

    model.load_state_dict(torch.load(\
            '/home/alex/projects/Beecy_Echo/nrrd_folds/fold_{}/checkpoint.pth'.format(\
            fold)))

    batch_size = 24
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
    num_labels=3
    num_labels_final = 3
    series_names = ['echo']
# series_names = ['Phase']
    seg_series_names = ['echo']


# model = models.Net23(2)
# model.cuda()

# model.load_state_dict(torch.load(\
            # '/home/alex/samsung_512/CMR_PC/checkpoint.pth'))

# model = densenet.densenet121()
# model.cuda()

# model.load_state_dict(torch.load(\
            # '/home/alex/samsung_512/CMR_PC/checkpoint_dense.pth'))

# model.eval()
# model.train()

    ind = 6
# mult_inds = [3,6,12,19,25,27,33,47]

    # f = preprocess.gen_filepaths(test_segpath)
    f_s = preprocess.gen_filepaths(test_segpath)
    f_v = preprocess.gen_filepaths(test_volpath)

    mult_inds = []
    for i in f_s:
        if 'segmentation' in i:
            mult_inds.append(int(re.findall('\d+', i)[0]))

    mult_inds = sorted(mult_inds)
    print(mult_inds)

# mult_inds = [3,6,12,19,25,49,54,56,58,62,64,66,71,75,78,80,85,86,90,93,\
            # 96,123,124,125,126,127,128,129,130]
    # mult_inds = [10550,10551]
    mult_inds = np.unique(mult_inds)

    volpaths, segpaths = utils.get_paths(mult_inds, f_s, f_v, series_names, \
            seg_series_names, test_volpath, test_segpath)

    t_transform_plan = transform_plan

    utils.test_net_cheap(test_volpath, test_segpath, mult_inds, in_z, model,\
            t_transform_plan, orig_dim, batch_size, out_file, num_labels,\
            num_labels_final, volpaths, segpaths, nrrd=True,\
            vol_only=double_vol, get_dice=False, make_niis=True)

# a, b, c = vol_display.masked_display(vol_out, out_out)
# a, b = vol_display.vol_display(out1)

# print(np.mean(np.array(dices)))

# if __name__ == '__main__':
        # import sys
        # if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            # pg.QtGui.QApplication.exec_()
