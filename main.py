import torch
# import numpy as np
import transforms
import time
import preprocess
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import utils
import models
import sys
# import densenet
# import densenet2
# import re
# from torch.nn import init

in_z = 0
fold = 7
volpath = '/home/alex/projects/Beecy_Echo/nii_renamed'.format(fold)
segpath = '/home/alex/projects/Beecy_Echo/nrrd_folds/fold_{}/train'.format(fold)
test_volpath = '/home/alex/projects/Beecy_Echo/nii_renamed/'.format(fold)
test_segpath = '/home/alex/projects/Beecy_Echo/nrrd_folds/fold_{}/test'.format(fold)

batch_size = 48
sqr = transforms.Square()
pad = transforms.Pad(36)
aff = transforms.Affine()
crop = transforms.RandomCrop(224)
scale = transforms.Scale(256)
rotate = transforms.Rotate(0.5, 30)
noise = transforms.Noise(0.02)
flip = transforms.Flip()
orig_dim = 256
# transform_plan = [sqr, scale, aff, rotate, crop, flip, noise]
transform_plan = [sqr, scale, aff, rotate, crop, noise]
lr = 1e-4
initial_depth = in_z*2+1
series_names = ['echo']
seg_series_names = ['echo']


center = transforms.CenterCrop2(224)
t_transform_plan = [sqr, scale, center]

model = models.Net23(3)

model.cuda()
# model.load_state_dict(torch.load(\
        # '/home/alex/samsung_512/CMR_PC/checkpoint_dense.pth'))

optimizer = optim.RMSprop(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=lr)

out_z, center_crop_sz = utils.get_out_size(orig_dim, in_z,\
        transform_plan, model)
center = transforms.CenterCrop(center_crop_sz)
depth_center = transforms.DepthCenterCrop(out_z)

t0 = time.time()

# weight = torch.FloatTensor(preprocess.get_weights(volpath, segpath,
    # 8, clip_val=1, nrrd=False)).cuda()

# print(weight)
# print('generating weights took {:.2f} seconds'.format(time.time()-t0))
counter = 0
print_interval = 200
model.train()
progress = [[0,0,0,0,0]]
weight = torch.FloatTensor([0.1,0.45, 0.45]).cuda()
for i in range(200000000000000000000000000000000):
    # model.train()
    # weight = torch.FloatTensor([0.0012, 0.0525, 0.9463]).cuda()
    vol, seg, inds = preprocess.get_batch(volpath, segpath, batch_size, in_z,\
            out_z, center_crop_sz, series_names, seg_series_names,\
            transform_plan, 8, nrrd=True)
    vol = torch.unsqueeze(vol, 1)
    vol = Variable(vol).cuda()
    seg = Variable(seg).cuda().long()

    out = model(vol).squeeze()

    loss = F.cross_entropy(out, seg, weight=weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    counter += 1

    sys.stdout.write('\r{:.2f}%'.format(counter*batch_size/print_interval))
    sys.stdout.flush()

    if counter*batch_size >= print_interval and i > 0:

        counter = 0
        # t_vol, t_seg, t_inds = preprocess.get_batch(test_volpath,\
                # test_segpath, 2, in_z,\
                # out_z, center_crop_sz, series_names, seg_series_names,\
                # transform_plan, 8, nrrd=True)
        # t_vol = torch.unsqueeze(t_vol, 1)
        # t_vol = Variable(t_vol).cuda()
        # t_seg = Variable(t_seg).cuda().long()

        # t_out = model(t_vol).squeeze()

        #display train set example
        # vol_ind = utils.get_liver(seg)
        # v = vol.data.cpu().numpy()[vol_ind].squeeze()
        # real_seg = seg[vol_ind].data.cpu().numpy()
        # out_plt = out.data.cpu().numpy()[vol_ind]
        # out_plt = np.argmax(out_plt, axis=0)
        # fake_seg = out_plt
        # masked_real = utils.makeMask(v, real_seg, out.size()[-3], 0.5)
        # masked_fake = utils.makeMask(v, fake_seg, out.size()[-3], 0.5)

        # display test set sample
        # vol_ind = utils.get_liver(t_seg)
        # v = t_vol.data.cpu().numpy()[vol_ind].squeeze()
        # real_seg = t_seg[vol_ind].data.cpu().numpy()
        # out_plt = t_out.data.cpu().numpy()[vol_ind]
        # out_plt = np.argmax(out_plt, axis=0)
        # fake_seg = out_plt
        # masked_real = utils.makeMask(v, real_seg, out.size()[-3], 0.5)
        # masked_fake = utils.makeMask(v, fake_seg, out.size()[-3], 0.5)

        # fig = plt.figure(1)
        # v = fig.add_subplot(2,2,1)
        # v.set_title('real-{}'.format(t_inds[vol_ind]))
        # plt.imshow(masked_real)
        # sreal = fig.add_subplot(2,2,2)
        # sreal.set_title('fake')
        # plt.imshow(masked_fake)
        # test_real = fig.add_subplot(2,2,3)
        # test_real.set_title('test_real')
        # plt.imshow(masked_real)
        # test = fig.add_subplot(2,2,4)
        # test.set_title('test fake')
        # fig.tight_layout()
        # plt.imshow(masked_fake)
        # plt.savefig('/home/alex/projects/Beecy_Echo'+\
                # '/nrrd_folds/fold_{}/out.png'.format(fold), dpi=200)
        # plt.clf()

        torch.save(model.state_dict(),'/home/alex/projects/' +\
                'Beecy_Echo/nrrd_folds/fold_{}/checkpoint.pth'.format(fold))

        print(('\rStep {} completed in {:.2f} sec ; Loss = {:.2f}')\
                .format(i*batch_size,time.time()-t0, loss.data.cpu().item()))

        t0 = time.time()
        # print(weight)
