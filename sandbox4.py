import torch
import numpy as np
import transforms
import time
import preprocess
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils
import models
import nibabel as nib
import os

in_z = 6
test_volpath = '/home/alex/LiverSegmentation/5mm_test_2'
test_segpath = '/home/alex/LiverSegmentation/5mm_test_2'
volpath = '/home/alex/LiverSegmentation/thick'
segpath = '/home/alex/LiverSegmentation/thick_segs'
batch_size = 25
t_batch_size = 1
pad = transforms.Pad(8)
crop = transforms.RandomCrop(400)
scale = transforms.Scale(256)
orig_dim = 512
transform_plan = [pad, crop]
lr = 1e-4
initial_depth = in_z*2+1


model = models.UNet()
model.cuda()

model.load_state_dict(torch.load(\
        '/home/alex/LiverSegmentation/5mm_model2_48hr.pth'))
model.eval()

t_transform_plan = [pad]
# t_transform_plan = transform_plan
t_out_z, t_center_crop_sz = utils.get_out_size(orig_dim, in_z,\
        t_transform_plan, model)
t_center = transforms.CenterCrop(t_center_crop_sz)
t_depth_center = transforms.DepthCenterCrop(t_out_z)


# def dense2less_dense(vol, add=False):
    # vs = np.array(vol.shape)
    # vs[0] += 2*in_z
    # bigvol = np.zeros(vs)
    # for i in range(len(vol)):
        # if add:
            # bigvol[i:i+in_z*2+1] += vol[i]
        # else:
            # bigvol[i:i+in_z*2+1] = vol[i]
        # print(i)
    # return bigvol

def dense2less_dense(vol, add=False):
    vs = np.array(vol.shape)
    vs[0] += 2*in_z
    if vs.shape[0] == 5:
        vs = np.array([vs[0], vs[1], vs[3], vs[4]])
        vol = np.transpose(vol,[0,2,1,3,4])
    elif vs.shape[0] == 4:
        vs = np.array([vs[0], vs[2], vs[3]])
    bigvol = np.zeros(vs)
    for i in range(len(vol)):
        if add:
            bigvol[i:i+vol[i].shape[0]] += vol[i]
        else:
            bigvol[i:i+vol[i].shape[0]] = vol[i]
    return bigvol.squeeze()

def open_nii(volpath, segpath, ind):
    volpath = os.path.join(volpath, 'volume-' + str(ind) + '.nii')
    segpath = os.path.join(segpath, 'segmentation-' + str(ind) + '.seg.nrrd')
    vol, seg = preprocess.get_nii_nrrd(volpath, segpath)
    assert np.shape(vol)[2] == np.shape(seg)[2]
    return vol, seg


def get_subvols(vol, seg, slice_inds, in_z, out_z, \
        center_crop_sz, txforms=None):
    vols = []
    segs = []

    center = transforms.CenterCrop(center_crop_sz)
    depth_center = transforms.DepthCenterCrop(out_z)
    for i in slice_inds:
        if in_z == 0:
            nascent_vol = vol[:,:,i]
            nascent_seg = seg[:,:,i]
        else:
            nascent_vol = vol[:,:,i-in_z:i+1+in_z]
            assert nascent_vol.shape[2]==in_z*2+1
            nascent_vol = np.squeeze(np.split(nascent_vol,\
                    nascent_vol.shape[2], axis=2))

            nascent_seg = seg[:,:,i-in_z:i+1+in_z]
            nascent_seg = depth_center.engage(nascent_seg)
            nascent_seg = np.squeeze(np.split(nascent_seg,\
                    nascent_seg.shape[2], axis=2))

        if txforms is not None:
            for j in txforms:
                nascent_vol, nascent_seg = \
                        j.engage(nascent_vol, nascent_seg, in_z, out_z)
            vols.append(nascent_vol)
            nascent_seg = np.array(center.engage(nascent_seg, out_z > 1))
            segs.append(nascent_seg)
    vols = np.array(vols).squeeze()
    segs = np.array(segs).squeeze()
    vols = preprocess.rot_and_flip(vols)
    segs = preprocess.rot_and_flip(segs)
    vols = vols-np.min(vols)
    vols = vols/np.max(vols)

    return vols, segs

ind = 6
# mult_inds = [3,6,12,19,25,27,33,47]
mult_inds = [25]
dices = []
for ind in mult_inds:
    print("\nProcessing index " + str(ind))
    vol, seg = open_nii(test_volpath, test_segpath, ind)
    num_slices = np.arange(np.shape(vol)[2])
    num_slices = num_slices[in_z:-in_z]
    slice_inds = num_slices
# slice_inds = [num_slices[-1], num_slices[-2]]
    for slice_ind in slice_inds:
        assert slice_ind >= np.min(num_slices)\
                and slice_ind <= np.max(num_slices)

    tvol, tseg = get_subvols(vol,seg, slice_inds, in_z, t_out_z,\
            t_center_crop_sz, t_transform_plan)
    slices = tvol.shape[0]
    tvol = np.expand_dims(tvol,1)
    tvol = np.array_split(tvol,slices//batch_size+1)
    tseg = np.array_split(tseg,slices//batch_size+1)


    tout = []
    for (i, tv) in enumerate(tvol):
        tv = torch.from_numpy(tv).float()
        tv = Variable(tv, volatile=True).cuda()
        # tv = Variable(tseg).cuda()

        tout.append(model(tv).squeeze().data.cpu().numpy())
        print(i)

    tseg = np.concatenate(tseg,axis=0)
    tvol = np.concatenate(tvol,axis=0)
    tout = np.concatenate(tout,axis=0)

    tseg = dense2less_dense(tseg)
    tvol = dense2less_dense(tvol)
    tout = dense2less_dense(tout, add=True)
    # pth = '/home/alex/LiverSegmentation/tseg.npy'
    tseg_hot = utils.get_hot(tseg, tout.shape[-3])
    tout_hot = np.argmax(tout,axis=1)
    tout_hot = utils.get_hot(tout_hot, tout.shape[-3])
    dice = utils.dice(tseg_hot[1:],tout_hot[1:])

    print(dice)
    dices.append(dice)
    n = 67

    tvol=t_center.engage(tvol[n], False)
    tv = np.copy(tvol)
    tout = np.argmax(tout[n],axis=0)
    tvol = np.zeros_like(tvol)
    tvol = np.where(tout>0, 1, tvol)
    print(tvol.shape)
    print(tout.shape)
    tmasked = utils.makeMask(tvol, tout, 4,0.5)
    plt.imshow(tmasked)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('mask_only.png', dpi=500)
    plt.imshow(tv, cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('CT.png', dpi=500)


# ii = -4
# tseg = tseg[ii]
# tvol = tvol[ii]
# tout = tout[ii]
# for i in np.arange(60,75):
        # tslice_ind = i
        # treal_seg = tseg[tslice_ind]
        # tout_plt = tout[tslice_ind]
        # tv = t_center.engage(tvol[tslice_ind], False)
        # tout_plt = np.argmax(tout_plt, axis=0)

# # tvol_ind, tslice_ind = utils.get_liver_from_vol(tseg)
# # tv = t_depth_center.engage(tvol)
# # tv = t_center.engage(tv[tvol_ind, 0, tslice_ind], False)
# # treal_seg = tseg[tvol_ind, tslice_ind]
# # tout_plt = tout[tvol_ind, :, tslice_ind]
# # tout_plt = np.argmax(tout_plt, axis=0)
        # tfake_seg = tout_plt
        # tmasked_real = utils.makeMask(tv, treal_seg, tout.shape[-3],0.5)
        # tmasked_fake = utils.makeMask(tv, tfake_seg, tout.shape[-3],0.5)
        # plt.imshow(tmasked_fake)
        # plt.show()
print(np.mean(np.array(dices)))
