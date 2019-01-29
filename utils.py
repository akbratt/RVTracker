from skimage import color
import preprocess
import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.autograd import Variable
import nibabel as nib
import scipy.ndimage
import sys
import re
import scipy as sp
import scipy.stats
import time
import skimage.segmentation as skseg
import sklearn.metrics.pairwise as pair
import nrrd
# import skimage.draw as draw
from scipy.ndimage import measurements
import pandas as pd


def seg2mask(seg, num_labels):
    r_mask = np.zeros((seg.shape[0], seg.shape[1], 3))
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    for i in np.arange(1,num_labels):
        r_mask = np.where(np.expand_dims(seg, -1)==i,colors[i-1],r_mask)
        # r_mask[:,:,i][np.where(seg==i)] = 1
    return r_mask

def makeMask(vol, seg, num_labels, alpha):
    if 'torch' in str(type(vol)):
        if 'cuda' in str(type(vol)):
            vol = vol.data.cpu().numpy()
            seg = seg.data.cpu().numpy()
        elif 'Variable' in str(type(vol)):
            vol = vol.data.numpy()
            seg = seg.data.numpy()
        else:
            vol = vol.numpy()
            seg = seg.numpy()
    if np.max(vol) > 1:
        vol = vol - np.min(vol)
        vol = vol/np.max(vol)
    color_mask = seg2mask(seg, num_labels)

    # Construct RGB version of grey-level image
    img_color = np.dstack((vol,vol,vol))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def get_liver(seg):
    if 'Variable' in str(type(seg)):
        seg = seg.data.cpu().numpy()
    elif 'torch' in str(type(seg)):
        seg = seg.cpu().numpy()
    bleh = np.array([len(np.unique(s)) for s in seg])
    livers = np.where(bleh>1)[0]
    ind = 0
    if livers.size is not 0:
        ind = np.random.choice(livers)
    return ind

def get_liver_from_vol(seg):
    if 'Variable' in str(type(seg)):
        seg = seg.data.cpu().numpy()
    elif 'torch' in str(type(seg)):
        seg = seg.numpy()
    bleh = np.array([[len(np.unique(s_)) for s_ in s] for s in seg])
    bleh2 = np.array([np.where(b>1)[0] for b in bleh])
    bleh3 = np.array([len(i) for i in bleh2])
    bleh4 = np.array(np.where(bleh3 > 0)[0])
    vol_ind = 0
    slice_ind = 0
    if bleh4.size is not 0:
        vol_ind = np.random.choice(bleh4)
        slice_ind = np.random.choice(bleh2[vol_ind])
    return vol_ind, slice_ind

class Image_Scroll_Helper(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='Greys_r')
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        # print(np.max(self.X[:,:,self.ind]))
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.autoscale()
        self.im.axes.figure.canvas.draw()

def image_scroll(vol):
    fig, ax = plt.subplots(1, 1)

    scroller = Image_Scroll_Helper(ax, vol)

    # ax.imshow(vol[:,:,442], cmap='Greys_r')
    fig.canvas.mpl_connect('scroll_event', scroller.onscroll)
    plt.show()

def get_out_size(orig_dim, in_z, transform_plan, net):
    dummy = np.random.rand(1, in_z*2+1, orig_dim, orig_dim)
    for i in transform_plan:
        dummy, _ = i.engage(dummy, dummy)
    dummy = np.squeeze(dummy)

    dummy = np.stack([dummy, dummy])
    dummy = np.expand_dims(dummy, axis=1)
    # print(dummy.shape)
    dummy = Variable(torch.from_numpy(dummy).float()).cuda()
    dummy = net(dummy)
    if len(dummy.size()) == 4:
        dummy = dummy.unsqueeze(2)
    out_dim = dummy.size()[-1]
    out_depth = dummy.size()[2]
    return out_depth, out_dim

def vol_seg_concat(vol, seg):
    chan2 = vol.clone()*0
    shape = vol.size()[-3:]
    dimx = seg.size()[-1]
    dimy = seg.size()[-2]
    dimz = seg.size()[-3]
    z0 = (shape[0] - dimz)//2
    z1 = (shape[0] + dimz)//2
    x0 = (shape[2] - dimx)//2
    x1 = (shape[2] + dimx)//2
    y0 = (shape[1] - dimy)//2
    y1 = (shape[1] + dimy)//2

    chan2[:, :, z0:z1, y0:y1, x0:x1] = seg
    return torch.cat([vol, chan2], dim=1)

def get_hot(seg, num_labels):
    hot = [np.where(seg==a, 1, 0) for a in range(num_labels)]
    hot = np.stack(hot, axis=0)
    return hot

def get_unhot(seg):
    return np.argmax(seg, 0)

def dice(real, fake):
    dice = np.sum(fake[real==1])*2.0 / (np.sum(fake) + np.sum(real))
    return dice

def jaccard(real, fake):
    jaccard = np.sum(fake[real==1]) / (np.sum(fake) + np.sum(real) -\
            np.sum(fake[real==1]))
    return jaccard

def open_double_vol(nii_paths):
    vols = []
    segs = []
    for nii_path in nii_paths:
        vol = nib.as_closest_canonical(nib.load(nii_path))
        vol = vol.get_data().astype(np.int16)
        vols.append(vol)
        segs.append(vol.copy())

    return vols, segs

def open_nii(volpath, segpath, ind, series_names, seg_series_names, nrrd=True):
    if nrrd:
        segpaths= [os.path.join(segpath, 'segmentation-'\
                + str(ind) + a + '.seg.nrrd') for a in seg_series_names]
        volpaths = [os.path.join(volpath, 'volume-'\
                + str(ind) + a + '.nii') for a in series_names]
        series, seg_series = preprocess.get_nii_nrrd(volpaths, segpaths)
    else:
        segpath = os.path.join(segpath, 'segmentation-' + str(ind) + '.nii')
        vol, seg = preprocess.get_nii_nii(volpath, segpath)
    assert np.shape(series)[3] == np.shape(seg_series)[3]
    return series, seg_series


def get_subvols_cheap(series, seg_series, slice_inds, in_z, out_z, \
        center_crop_sz, model, num_labels, batch_size, txforms=None,\
        verbose=True):

    # get beginning index of output since the z dim is smaller than vol
    z0 = (in_z*2+1 - out_z)//2

    sz = np.array([num_labels, slice_inds.shape[0]+2*in_z, center_crop_sz,\
            center_crop_sz])

    bigout = np.zeros(sz)
    bigvol = np.zeros(sz[1:])
    bigseg = np.zeros(sz[1:])

    center = transforms.CenterCrop(center_crop_sz)
    depth_center = transforms.DepthCenterCrop(out_z)
    vols = []
    segs = []
    batch_ind = 0
    absolute_ind = 0
    for i in slice_inds:
        if in_z == 0:
            nascent_series = [vol[:,:,i] for vol in series]
            nascent_seg_series = [seg[:,:,i] for seg in seg_series]
            nascent_series = np.expand_dims(nascent_series, axis=0)
            nascent_seg_series = np.expand_dims(nascent_seg_series, axis=0)
        else:
            nascent_series = [v[:,:,i-in_z:i+1+in_z] for v in series]
            assert nascent_series[0].shape[2]==in_z*2+1
            nascent_series = [np.squeeze(np.split(v,\
                    v.shape[2], axis=2)) for v in nascent_series]

            nascent_seg_series = [s[:,:,i-in_z:i+1+in_z] for s in seg_series]
            nascent_seg_series = [depth_center.engage(s) for s in\
                    nascent_seg_series]
            nascent_seg_series = [np.squeeze(np.split(s,\
                    s.shape[2], axis=2)) for s in nascent_seg_series]

            if out_z == 1:
                nascent_seg_series = \
                        np.expand_dims(nascent_seg_series, axis=0)

        if txforms is not None:
            for j in txforms:
                nascent_series, nascent_seg_series = \
                        j.engage(nascent_series, nascent_seg_series)

            vols.append(np.squeeze(nascent_series))

            segs.append(np.squeeze(center.engage(nascent_seg_series, \
                    out_z > 1)))

            absolute_ind += 1

        if (absolute_ind >= batch_size or (i >= slice_inds[-1] and vols)):
            # nascent_vol = np.array(vols).squeeze()
            # nascent_seg = np.array(segs).squeeze()
            nascent_series = np.array(vols)
            nascent_seg_series = np.array(segs)
            nascent_series = preprocess.rot_and_flip(nascent_series)
            nascent_seg_series = preprocess.rot_and_flip(nascent_seg_series)
            nascent_series = nascent_series-np.min(nascent_series)
            nascent_series = nascent_series/np.max(nascent_series)

            if len(nascent_series.shape) < 4:
                nascent_series = np.expand_dims(nascent_series, 0)

            tv = torch.from_numpy(nascent_series).float()
            tv = Variable(tv).cuda()
            # print(i)
            if verbose:
                sys.stdout.write('\r{:.2f}%'.format(i/sz[1]))
                sys.stdout.flush()
            if in_z == 0:
                tv = tv.permute(1,0,2,3)
            tout = model(tv).data.cpu().numpy().astype(np.int8)
            if in_z == 0:
                nascent_series = nascent_series.squeeze()
                if np.array(nascent_series.shape).shape[0] < 3:
                    nascent_series = np.expand_dims(nascent_series, 0)
            for j in range(len(nascent_series)):

                bsz = len(nascent_series)
                beg = i - in_z + z0 - bsz + j + 1
                end = i - in_z + z0 - bsz + j + out_z + 1
                bigout[:,beg:end] += np.expand_dims(tout[j], 1)
                bigseg[beg:end] = nascent_seg_series[j]

                beg = i - in_z + 1 - bsz + j
                end = i + in_z - bsz + j + 2
                bigvol[beg:end] = nascent_series[j]

            absolute_ind = 0
            batch_ind += 1
            vols = []
            segs = []

    return bigout, bigvol, bigseg

def test_net_cheap(test_volpath, test_segpath, mult_inds, in_z, model,\
        t_transform_plan, orig_dim, batch_size, out_file, num_labels,\
        num_labels_final, volpaths, segpaths,\
        nrrd=True, vol_only=False, get_dice=False, make_niis=False,\
        verbose=True):

    t_out_z, t_center_crop_sz = get_out_size(orig_dim, in_z,\
            t_transform_plan, model)
    t_center = transforms.CenterCrop(t_center_crop_sz)

    dices = []
    jaccards = []
    hds = []
    assds = []
    dice_inds = []
    times = []
    for ind in range(len(mult_inds)):
        t0 = time.time()
        # print("\nProcessing index " + str(mult_inds[ind]))
        if vol_only:
            series, seg_series = open_double_vol(volpaths[ind])
            seg_series = [a*0 for a in seg_series]
        else:
            # vol, seg = open_nii(test_volpath, test_segpath, ind, series_names,\
                    # seg_series_names, nrrd)
            series, seg_series = preprocess.get_nii_nrrd(volpaths[ind],\
                    segpaths[ind])
        num_slices = np.arange(np.shape(series[0])[2])
        if in_z == 0:
            num_slices = num_slices
        else:
            num_slices = num_slices[in_z:-in_z]

        slice_inds = num_slices
        for slice_ind in slice_inds:
            assert slice_ind >= np.min(num_slices)\
                    and slice_ind <= np.max(num_slices)

        tout, tvol, tseg = get_subvols_cheap(series, seg_series, slice_inds,\
                in_z, t_out_z, t_center_crop_sz, model, num_labels,\
                batch_size, t_transform_plan, verbose=verbose)
        duration = time.time() - t0
        # tseg_orig = tseg.copy()
        tseg = np.clip(tseg, 0,2)
        times.append(duration)


        # hd, assd = 1, 1
        if get_dice:
            # hd, assd = get_dists_volumetric(tseg.astype(np.int64),\
                    # np.argmax(tout, axis=0))
            hd, assd = get_dists_non_volumetric(tseg.astype(np.int64),\
                    np.argmax(tout, axis=0))
            tseg_hot = get_hot(tseg, num_labels_final)
            tout_hot = np.argmax(tout,axis=0)
            tout_hot = np.clip(tout_hot, 0,1)
            tout_hot = get_hot(tout_hot, num_labels_final)
            dce = dice(tseg_hot[1:],tout_hot[1:])
            jc = jaccard(tseg_hot[1:], tout_hot[1:])

            if verbose:
                print(('\r{}: Duration: {:.2f} ; Dice: {:.2f} ; Jaccard: {:.2f}' +\
                        ' ; Hausdorff: {:.2f} ; ASSD: {:.2f}').format(\
                        mult_inds[ind], duration, dce, jc, np.mean(hd),\
                        np.mean(assd)))
            jaccards.append(jc)
            dices.append(dce)
            hds.append(hd)
            assds.append(assd)
            dice_inds.append(mult_inds[ind])
        else:
            if verbose:
                print('\r{}'.format(mult_inds[ind]))


        # for i in range(tout.shape[1]):
            # pic_out = makeMask(tvol[i], np.argmax(tout,0)[i], 3, 0.5)
            # pic_seg = makeMask(tvol[i], tseg_orig[i], 3, 0.5)
            # fig = plt.figure(1)
            # fig.add_subplot(111)
            # fig.tight_layout()
            # plt.imshow(pic_out)
            # plt.savefig('/home/alex/samsung_512/CMR_PC/figs/{}_{}_out.png'.format(\
                    # mult_inds[ind], i), dpi=500)
            # plt.imshow(pic_seg)
            # plt.savefig('/home/alex/samsung_512/CMR_PC/figs/{}_{}_seg.png'.format(\
                    # mult_inds[ind], i), dpi=500)
            # plt.clf()
        com_fake = []
        com_real = []

        if make_niis:
            # out_out = tout
            out_out = np.zeros_like(tout[0])
            maxes = np.argmax(tout, axis=0)
            sparse_maxes = sparsify(maxes)
            for i in range(sparse_maxes.shape[1]):
                lw1, num1 = measurements.label(sparse_maxes[1,i])
                area1 = measurements.sum(sparse_maxes[1,i],lw1,\
                        index=np.arange(lw1.max() + 1))
                areaImg1 = area1[lw1]
                sparse_maxes[1,i] = np.where(areaImg1 < np.max(areaImg1), 0, 1)
                com_lateral = list(measurements.center_of_mass(sparse_maxes[1,i]))

                lw2, num2 = measurements.label(sparse_maxes[2,i])
                area2 = measurements.sum(sparse_maxes[2,i],lw2,\
                        index=np.arange(lw2.max() + 1))
                areaImg2 = area2[lw2]
                sparse_maxes[2,i] = np.where(areaImg2 < np.max(areaImg2), 0, 1)
                com_septal = list(measurements.center_of_mass(sparse_maxes[2,i]))
                com_fake.append(com_lateral + com_septal)
            # for i in range(tout.shape[1]):
                # max1 = np.where((maxes[i] == 0) | (maxes[i] == 2),\
                        # -1000, tout[0,i])
                # m = np.argmax(max1)
                # y, x = np.unravel_index(m, out_out[i].shape)
                # rr, cc = draw.circle(y,x,5)
                # out_out[i,rr,cc] = 1

                # max2 = np.where((maxes[i] == 0) | (maxes[i] == 1),\
                        # -1000, tout[0, i])
                # m2 = np.argmax(max2)
                # y, x = np.unravel_index(m2, out_out[i].shape)
                # rr, cc = draw.circle(y,x,5)
                # out_out[i,rr,cc] = 2
            # out_out = np.flip(out_out, -1)
            maxes = np.argmax(sparse_maxes, axis=0)
            out_out = np.flip(maxes, -1)
            out_out = np.rot90(out_out, k=-1, axes=(-2,-1))
            out_out = np.transpose(out_out,[1,2,0])
            write_nrrd(out_out.astype(np.uint8), \
                    out_file + '/tout-{}.seg.nrrd'.format(\
                    mult_inds[ind]))

            seg_out = tseg
            sparse_seg = sparsify(tseg.astype(np.uint8))
            for i in range(sparse_seg.shape[1]):
                com_lateral_seg = list(measurements.center_of_mass(sparse_seg[1,i]))
                com_septal_seg = list(measurements.center_of_mass(sparse_seg[2,i]))
                com_real.append(com_lateral_seg + com_septal_seg)

            seg_out = np.flip(seg_out, -1)
            seg_out = np.rot90(seg_out, k=-1, axes=(-2,-1))
            seg_out = np.transpose(seg_out,[1,2,0])
            write_nrrd(seg_out.astype(np.uint8), \
                    out_file + '/tseg-{}.seg.nrrd'.format(\
                    mult_inds[ind]))

            tv = np.stack(t_center.engage(np.expand_dims(tvol, 0),True))
            vol_out = tv
            vol_out = np.flip(vol_out, -1)
            vol_out = np.rot90(vol_out, k=-1, axes=(-2,-1))
            vol_out = np.transpose(vol_out,[1,2,0])
            vol_out = nib.Nifti1Image(vol_out, np.eye(4))
            nib.save(vol_out, \
                    out_file + '/tvol-{}.nii'.format(\
                    mult_inds[ind]))
        # print('Jaccard summary: ' + str(get_CI(jaccards)))
            [a.extend(b) for a, b in zip(com_real, com_fake)]
            com_merged = com_real
            com_merged = [[round(b, 2) for b in a] for a in com_merged]
            headers = ['real_y_l', 'real_x_l', 'real_y_s', 'real_x_s', \
                    'fake_y_l', 'fake_x_l', 'fake_y_s', 'fake_x_s']
            df = pd.DataFrame(com_merged, columns=headers)
            df.to_csv(out_file + '/{}.csv'.format(mult_inds[ind]))

    # return vol_out, out_out, seg_out
    if get_dice:
        return np.array(dices), np.array(jaccards), np.array(hds), np.array(assds),\
                np.array(times)
    else:
        return

def get_paths(inds, f_s, f_v, series_names, seg_series_names, volpath, segpath):
    volpaths = []
    segpaths = []

    for i in inds:
        vol_files = []
        seg_files = []
        for name in series_names:
            for j in f_v:
                if 'volume' in j and name in j:
                    ind0 = int(re.findall('\d+', j)[0])
                    if i == ind0:
                        vol_files.append(os.path.join(volpath,j))

        volpaths.append(vol_files)

        for name in seg_series_names:
            for j in f_s:
                if 'segmentation' in j and name in j:
                    ind0 = int(re.findall('\d+', j)[0])
                    if i == ind0:
                        seg_files.append(os.path.join(segpath,j))
        segpaths.append(seg_files)
    return volpaths, segpaths

def get_CI(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

# get hausdorff and average symmetric surface distances for two sets of
# segmentations
def get_dists(real, fake):
    hd = []
    assd = []

    for r, f in zip(real, fake):
        f_border = skseg.find_boundaries(f)
        r_border = skseg.find_boundaries(r)

        f_coords = np.argwhere(f_border == 1).copy(order='C').astype(np.float64)
        r_coords = np.argwhere(r_border == 1).copy(order='C').astype(np.float64)
        # print(r_coords.shape)
        # print(f_coords.shape)

        euclid_distances = pair.euclidean_distances(f_coords, r_coords)

        ab = np.min(euclid_distances, axis=1)
        ba = np.min(euclid_distances, axis=0)
        hd_pre = np.max(np.concatenate([ab, ba]))
        assd_pre = np.mean(np.concatenate([ab, ba]))
        hd.append(hd_pre)
        assd.append(assd_pre)

    hd = np.array(hd)
    return hd, assd

# get hausdorff and average symmetric surface distance over a volume
def get_dists_volumetric(real, fake):
    pres = []

    for r, f in zip(real, fake):
        f_border = skseg.find_boundaries(f)
        r_border = skseg.find_boundaries(r)

        f_coords = np.argwhere(f_border == 1).copy(order='C').astype(np.float64)
        r_coords = np.argwhere(r_border == 1).copy(order='C').astype(np.float64)
        # print(r_coords.shape)
        # print(f_coords.shape)

        try:
            euclid_distances = pair.euclidean_distances(f_coords, r_coords)

            ab = np.min(euclid_distances, axis=1)
            ba = np.min(euclid_distances, axis=0)

            pre = np.concatenate([ab, ba])
            pres.append(pre)
        except ValueError:
            print('surface error')

    try:
        hd = np.max(np.concatenate(pres))
        assd = np.mean(np.concatenate(pres))
        return hd, assd
    except ValueError:
        print('no valid surfaces')
        return 0, 0

def get_dists_non_volumetric(real, fake):
    pres = []
    hds = []

    for r, f in zip(real, fake):
        f_border = skseg.find_boundaries(f)
        r_border = skseg.find_boundaries(r)

        f_coords = np.argwhere(f_border == 1).copy(order='C').astype(np.float64)
        r_coords = np.argwhere(r_border == 1).copy(order='C').astype(np.float64)
        # print(r_coords.shape)
        # print(f_coords.shape)

        try:
            euclid_distances = pair.euclidean_distances(f_coords, r_coords)

            ab = np.min(euclid_distances, axis=1)
            ba = np.min(euclid_distances, axis=0)

            pre = np.concatenate([ab, ba])
            hds.append(np.max(pre))
            pres.append(pre)
        except ValueError:
            print('surface error')

    try:
        hd = np.mean(hds)
        assd = np.mean(np.concatenate(pres))
        return hd, assd
    except ValueError:
        print('no valid surfaces')
        return 0, 0


def get_subvols_cheap_phase(series, phase_series, seg_series, slice_inds,\
        in_z, out_z, center_crop_sz, model, num_labels, batch_size,\
        txforms=None, verbose=True):

    # get beginning index of output since the z dim is smaller than vol
    z0 = (in_z*2+1 - out_z)//2

    sz = np.array([num_labels, slice_inds.shape[0]+2*in_z, center_crop_sz,\
            center_crop_sz])

    bigout = np.zeros(sz)
    bigvol = np.zeros(sz[1:])
    bigseg = np.zeros(sz[1:])
    bigphase = np.zeros(sz[1:])

    center = transforms.CenterCrop(center_crop_sz)
    depth_center = transforms.DepthCenterCrop(out_z)
    vols = []
    segs = []
    phases = []
    batch_ind = 0
    absolute_ind = 0
    for i in slice_inds:
        if in_z == 0:
            nascent_series = [vol[:,:,i] for vol in series]
            nascent_seg_series = [seg[:,:,i] for seg in seg_series]
            nascent_series = np.expand_dims(nascent_series, axis=0)
            nascent_seg_series = np.expand_dims(nascent_seg_series, axis=0)

            nascent_phase_series = [phase[:,:,i] for phase in phase_series]
            nascent_phase_series = np.expand_dims(nascent_phase_series, axis=0)
        else:
            nascent_series = [v[:,:,i-in_z:i+1+in_z] for v in series]
            assert nascent_series[0].shape[2]==in_z*2+1
            nascent_series = [np.squeeze(np.split(v,\
                    v.shape[2], axis=2)) for v in nascent_series]

            nascent_seg_series = [s[:,:,i-in_z:i+1+in_z] for s in seg_series]
            nascent_seg_series = [depth_center.engage(s) for s in\
                    nascent_seg_series]
            nascent_seg_series = [np.squeeze(np.split(s,\
                    s.shape[2], axis=2)) for s in nascent_seg_series]

            nascent_phase_series = [p[:,:,i-in_z:i+1+in_z] for p in phase_series]
            nascent_phase_series = [np.squeeze(np.split(p,\
                    p.shape[2], axis=2)) for p in nascent_phase_series]

            if out_z == 1:
                nascent_seg_series = \
                        np.expand_dims(nascent_seg_series, axis=0)

        if txforms is not None:
            for j in txforms:
                nascent_series, nascent_seg_series = \
                        j.engage(nascent_series, nascent_seg_series)

                nascent_phase_series, nascent_seg_series = \
                        j.engage(nascent_phase_series, nascent_seg_series)

            vols.append(np.squeeze(nascent_series))

            segs.append(np.squeeze(center.engage(nascent_seg_series, \
                    out_z > 1)))

            phases.append(np.squeeze(nascent_phase_series))

            absolute_ind += 1

        if (absolute_ind >= batch_size or (i >= slice_inds[-1] and vols)):
            # nascent_vol = np.array(vols).squeeze()
            # nascent_seg = np.array(segs).squeeze()
            nascent_series = np.array(vols)
            nascent_seg_series = np.array(segs)
            nascent_series = preprocess.rot_and_flip(nascent_series)
            nascent_seg_series = preprocess.rot_and_flip(nascent_seg_series)
            nascent_series = nascent_series-np.min(nascent_series)
            nascent_series = nascent_series/np.max(nascent_series)

            nascent_phase_series = np.array(phases)
            nascent_phase_series = preprocess.rot_and_flip(nascent_phase_series)
            nascent_phase_series = nascent_phase_series -\
                    np.min(nascent_phase_series)
            nascent_phase_series = nascent_phase_series/\
                    np.max(nascent_phase_series)

            if len(nascent_series.shape) < 4:
                nascent_series = np.expand_dims(nascent_series, 0)
                nascent_phase_series = np.expand_dims(nascent_phase_series, 0)

            tv = torch.from_numpy(nascent_series).float()
            tv = Variable(tv).cuda()
            # print(i)
            if verbose:
                sys.stdout.write('\r{:.2f}%'.format(i/sz[1]))
                sys.stdout.flush()
            if in_z == 0:
                tv = tv.permute(1,0,2,3)
            tout = model(tv).data.cpu().numpy().astype(np.int8)
            if in_z == 0:
                nascent_series = nascent_series.squeeze()
                if np.array(nascent_series.shape).shape[0] < 3:
                    nascent_series = np.expand_dims(nascent_series, 0)
            for j in range(len(nascent_series)):

                bsz = len(nascent_series)
                beg = i - in_z + z0 - bsz + j + 1
                end = i - in_z + z0 - bsz + j + out_z + 1
                bigout[:,beg:end] += np.expand_dims(tout[j], 1)
                bigseg[beg:end] = nascent_seg_series[j]

                beg = i - in_z + 1 - bsz + j
                end = i + in_z - bsz + j + 2
                bigvol[beg:end] = nascent_series[j]
                bigphase[beg:end] = nascent_phase_series[j]

            absolute_ind = 0
            batch_ind += 1
            vols = []
            segs = []
            phases = []

    return bigout, bigvol, bigseg, bigphase

def test_net_cheap_phase(test_volpath, test_segpath, mult_inds, in_z, model,\
        t_transform_plan, orig_dim, batch_size, out_file, num_labels,\
        num_labels_final, volpaths, segpaths,\
        nrrd=True, vol_only=False, get_dice=False, make_niis=False,\
        verbose=True):

    t_out_z, t_center_crop_sz = get_out_size(orig_dim, in_z,\
            t_transform_plan, model)
    t_center = transforms.CenterCrop(t_center_crop_sz)

    dices = []
    jaccards = []
    hds = []
    assds = []
    dice_inds = []
    times = []
    for ind in range(len(mult_inds)):
        t0 = time.time()
        # print("\nProcessing index " + str(mult_inds[ind]))
        if vol_only:
            series, seg_series = open_double_vol(volpaths[ind])
            seg_series = [a*0 for a in seg_series]
        else:
            # vol, seg = open_nii(test_volpath, test_segpath, ind, series_names,\
                    # seg_series_names, nrrd)
            series, seg_series = preprocess.get_nii_nrrd(volpaths[ind],\
                    segpaths[ind])
        num_slices = np.arange(np.shape(series[0])[2])
        if in_z == 0:
            num_slices = num_slices
        else:
            num_slices = num_slices[in_z:-in_z]

        slice_inds = num_slices
        for slice_ind in slice_inds:
            assert slice_ind >= np.min(num_slices)\
                    and slice_ind <= np.max(num_slices)

        tout, tvol, tseg = get_subvols_cheap(series, seg_series, slice_inds,\
                in_z, t_out_z, t_center_crop_sz, model, num_labels,\
                batch_size, t_transform_plan, verbose=verbose)
        duration = time.time() - t0
        # tseg_orig = tseg.copy()
        tseg = np.clip(tseg, 0,1)
        times.append(duration)


        # hd, assd = 1, 1
        if get_dice:
            # hd, assd = get_dists_volumetric(tseg.astype(np.int64),\
                    # np.argmax(tout, axis=0))
            hd, assd = get_dists_non_volumetric(tseg.astype(np.int64),\
                    np.argmax(tout, axis=0))
            tseg_hot = get_hot(tseg, num_labels_final)
            tout_hot = np.argmax(tout,axis=0)
            tout_hot = np.clip(tout_hot, 0,1)
            tout_hot = get_hot(tout_hot, num_labels_final)
            dce = dice(tseg_hot[1:],tout_hot[1:])
            jc = jaccard(tseg_hot[1:], tout_hot[1:])

            if verbose:
                print(('\r{}: Duration: {:.2f} ; Dice: {:.2f} ; Jaccard: {:.2f}' +\
                        ' ; Hausdorff: {:.2f} ; ASSD: {:.2f}').format(\
                        mult_inds[ind], duration, dce, jc, np.mean(hd),\
                        np.mean(assd)))
            jaccards.append(jc)
            dices.append(dce)
            hds.append(hd)
            assds.append(assd)
            dice_inds.append(mult_inds[ind])
        else:
            if verbose:
                print('\r{}'.format(mult_inds[ind]))


        # for i in range(tout.shape[1]):
            # pic_out = makeMask(tvol[i], np.argmax(tout,0)[i], 3, 0.5)
            # pic_seg = makeMask(tvol[i], tseg_orig[i], 3, 0.5)
            # fig = plt.figure(1)
            # fig.add_subplot(111)
            # fig.tight_layout()
            # plt.imshow(pic_out)
            # plt.savefig('/home/alex/samsung_512/CMR_PC/figs/{}_{}_out.png'.format(\
                    # mult_inds[ind], i), dpi=500)
            # plt.imshow(pic_seg)
            # plt.savefig('/home/alex/samsung_512/CMR_PC/figs/{}_{}_seg.png'.format(\
                    # mult_inds[ind], i), dpi=500)
            # plt.clf()

        if make_niis:
            # out_out = tout
            # out_out = np.argmax(out_out, axis=0)
            # out_out = np.flip(out_out, -1)
            # out_out = np.rot90(out_out, k=-1, axes=(-2,-1))
            # out_out = np.transpose(out_out,[1,2,0])
            # out_out = nib.Nifti1Image(out_out.astype(np.int16), np.eye(4))
            # nib.save(out_out, \
                    # out_file + '/tout-{}.nii'.format(\
                    # mult_inds[ind]))
            out_out = tout
            out_out = np.argmax(out_out, axis=0)
            out_out = np.flip(out_out, -1)
            out_out = np.rot90(out_out, k=-1, axes=(-2,-1))
            out_out = np.transpose(out_out,[1,2,0])
            out_out = nib.Nifti1Image(out_out.astype(np.int16), np.eye(4))
            write_nrrd(out_out.astype(np.uint8), \
                    out_file + '/tout-{}.seg.nrrd'.format(\
                    mult_inds[ind]))

            # seg_out = tseg
            # seg_out = np.flip(seg_out, -1)
            # seg_out = np.rot90(seg_out, k=-1, axes=(-2,-1))
            # seg_out = np.transpose(seg_out,[1,2,0])
            # seg_out = nib.Nifti1Image(seg_out.astype(np.int16), np.eye(4))
            # nib.save(seg_out, \
                    # out_file + '/tseg-{}.nii'.format(\
                    # mult_inds[ind]))
            seg_out = tseg
            seg_out = np.flip(seg_out, -1)
            seg_out = np.rot90(seg_out, k=-1, axes=(-2,-1))
            seg_out = np.transpose(seg_out,[1,2,0])
            write_nrrd.save(seg_out.astype(np.uint8), \
                    out_file + '/tseg-{}.seg.nrrd'.format(\
                    mult_inds[ind]))

            tv = np.stack(t_center.engage(np.expand_dims(tvol, 0),True))
            vol_out = tv
            vol_out = np.flip(vol_out, -1)
            vol_out = np.rot90(vol_out, k=-1, axes=(-2,-1))
            vol_out = np.transpose(vol_out,[1,2,0])
            nib.save(vol_out, \
                    out_file + '/tvol-{}.nii'.format(\
                    mult_inds[ind]))
        # print('Jaccard summary: ' + str(get_CI(jaccards)))

    # return vol_out, out_out, seg_out
    if get_dice:
        return np.array(dices), np.array(jaccards), np.array(hds), np.array(assds),\
                np.array(times)
    else:
        return

def bounding_box(seg):
    x = np.any(np.any(seg, axis=0), axis=1)
    y = np.any(np.any(seg, axis=1), axis=1)
    z = np.any(np.any(seg, axis=1), axis=0)
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    bbox = np.array([ymin,ymax,xmin,xmax,zmin,zmax])
    return bbox

def get_shape_origin(seg):
    bbox = bounding_box(seg)
    ymin, ymax, xmin, xmax, zmin, zmax = bbox
    shape = list(np.array([ymax-ymin, xmax-xmin, zmax-zmin]) + 1)
    origin = [ymin, xmin, zmin]
    return shape, origin

def sparsify(a):
    ncols = a.max() + 1
    if ncols < 3:
        ncols = 3
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    out = np.transpose(out, axes=(3,0,1,2))
    return out

def write_nrrd(seg, out_file):
    options = {}
    options['dimension'] = 4
    options['encoding'] = 'gzip'
    options['kinds'] = ['complex', 'domain', 'domain', 'domain']
    options['measurement frame'] = [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
    options['space'] = 'right-anterior-superior'
    options['space directions'] = [['0', '0', '0'], ['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
    options['type'] = 'unsigned char'
    options['keyvaluepairs'] = {}

    box = bounding_box(seg)
    seg_cut = seg[box[0]:box[1]+1,box[2]:box[3]+1,box[4]:box[5]+1]
    sparse = sparsify(seg_cut)[1:,:,:,:]
    shape, origin = get_shape_origin(seg)

    options['sizes'] = [np.max(seg), *shape]
    options['space origin'] = [str(a) for a in origin]

    keyvaluepairs = {}

    for i in range(np.max(seg)):
        seg_slice = sparse[i]
        name = 'Segment{}'.format(i)
        keyvaluepairs[name + '_Color'] = ' '.join([str(a) for a in np.random.rand(3)])
        keyvaluepairs[name + '_ColorAutoGenerated'] = '1'
        keyvaluepairs[name + '_Extent'] = ' '.join([str(a) for a in bounding_box(seg_slice)])
        keyvaluepairs[name + '_ID'] = 'Segment_{}'.format(i+1)
        keyvaluepairs[name + '_Name'] = 'Segment_{}'.format(i+1)
        keyvaluepairs[name + '_NameAutoGenerated'] = 1
        keyvaluepairs[name + '_Tags'] = 'TerminologyEntry:Segmentation category' +\
            ' and type - 3D Slicer General Anatomy list~SRT^T-D0050^Tissue~SRT^' +\
            'T-D0050^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'

    keyvaluepairs['Segmentation_ContainedRepresentationNames'] = 'Binary labelmap|'
    keyvaluepairs['Segmentation_ConversionParameters'] = 'placeholder'
    keyvaluepairs['Segmentation_MasterRepresentation'] = 'Binary labelmap'
    keyvaluepairs['Segmentation_ReferenceImageExtentOffset'] = ' '.join(options['space origin'])

    options['keyvaluepairs'] = keyvaluepairs
    nrrd.write(out_file, sparse, options =
            options)


