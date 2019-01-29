import torch
import numpy as np
import transforms
import preprocess
import utils
import models
import re

###################### Network Parameters ######################

batch_size = 24
checkpoint_path = 'path to model checkpoint'
test_volpath = 'path to test volumes'
test_segpath = 'path to test segmentations'
num_labels = 3 #labels: lateral annulus, medial annulus, background

crop_size = 224
original_size = 256
out_file = 'path to store output'

################################################################

sqr = transforms.Square()
center = transforms.CenterCrop2(crop_size)
scale = transforms.Scale(original_size)
transform_plan = [sqr, scale, center]
series_names = ['echo']
seg_series_names = ['echo']

model = models.Net23(num_labels)
model.cuda()

model.load_state_dict(torch.load(checkpoint_path))

f_s = preprocess.gen_filepaths(test_segpath)
f_v = preprocess.gen_filepaths(test_volpath)

mult_inds = []
for i in f_s:
    if 'segmentation' in i:
        mult_inds.append(int(re.findall('\d+', i)[0]))

mult_inds = sorted(mult_inds)

mult_inds = np.unique(mult_inds)

volpaths, segpaths = utils.get_paths(mult_inds, f_s, f_v, series_names, \
        seg_series_names, test_volpath, test_segpath)

t_transform_plan = transform_plan

utils.test_net_cheap(test_volpath, test_segpath, mult_inds, 0, model,\
        t_transform_plan, original_size, batch_size, out_file, num_labels,\
        num_labels, volpaths, segpaths, nrrd=True,\
        vol_only=False, get_dice=False, make_niis=True)
