import numpy as np
import os
import nibabel as nib
import multiprocessing as mp
import transforms
import torch
import re
import nrrd
# import time
# import matplotlib.pyplot as plt

path = "/home/alex/LiverSegmentation/img/"

# Rotates and flips images becaus the .nii volumes load flipped and rotated
def rot_and_flip(img):
    img = np.rot90(img, axes=(-2, -1))
    img = np.flip(img, -1)
    #doing the below because weird error message otherwise
    img = np.clip(img, -10000, 300000)
    return img

# Turns a list of lists of volumes into a list of volumes
def unpack(vols):
    return [item for sublist in vols for item in sublist]

def get_nii_nii(volpath,segpath):
    vol = nib.as_closest_canonical(nib.load(volpath))
    vol = vol.get_data().astype(np.int16)
    seg = nib.as_closest_canonical(nib.load(segpath))
    seg = seg.get_data().astype(np.int16)

    return vol, seg

def get_nii(volpath):
    vol = nib.as_closest_canonical(nib.load(volpath))
    vol = vol.get_data().astype(np.int16)

    return vol

def get_nii_nrrd(nii_paths, nrrd_paths):
    vols = []
    for nii_path in nii_paths:
        vol = nib.as_closest_canonical(nib.load(nii_path))
        vol = vol.get_data().astype(np.int16)
        vols.append(vol)

    segs = []
    for nrrd_path in nrrd_paths:
        d, ok = nrrd.read(nrrd_path)
        if len(d.shape) < 4:
            d = np.expand_dims(d, 0)
        o = ok['keyvaluepairs']['Segmentation_ReferenceImageExtentOffset']
        for i in range(d.shape[0]):
            d[i] = np.where(d[i]==0,d[i],i+1)
        d = np.max(d,0)
        # d[np.where(d==3)] = 2
        directions = ok['space directions']
        directions = [[float(a) for a in b] for b in directions[1:]]
        wonky = False
        if len(np.where(np.array(directions)<-0.1)[0]) > 0:
            wonky = True
            d = np.flip(d, 0)

        seg = np.zeros_like(vols[0])
        o = [int(a) for a in re.findall(r'\d+', o)]
        y_offset = o[0]
        if wonky:
            y_offset = vols[0].shape[0] - d.shape[0] - o[0]
        x_offset = o[1]
        z_offset = o[2]
        try:
            seg[y_offset:d.shape[0]+y_offset, x_offset:d.shape[1]+x_offset,\
                    z_offset:d.shape[2]+z_offset] = d[:,:,0:seg.shape[-1]]
        except ValueError:
            print(nrrd_path)

        seg = nib.as_closest_canonical(nib.Nifti1Image(seg, np.eye(4)))
        seg = seg.get_data().astype(np.int16)
        segs.append(seg)
    return vols, segs

# Opens volumes and segmentations and returns random slices according to the
# parameters
def open_nii(volpaths, segpaths, ind, num, in_z, out_z, center_crop_sz,\
        series_names, seg_series_names, txforms=None,nrrd=True):
    vols = []
    segs = []
    # volpath = os.path.join(volpaths, 'volume-' + str(ind) + '.nii')
    if nrrd:
        # segpath = os.path.join(segpath, 'segmentation-' + str(ind) + '.seg.nrrd')
        series, seg_series = get_nii_nrrd(volpaths, segpaths)
    # else:
        # segpath = os.path.join(segpath, 'segmentation-' + str(ind) + '.nii')
        # vol, seg = get_nii_nii(volpaths, segpaths)
    assert np.shape(series)[3] == np.shape(seg_series)[3]
    num_slices = np.arange(np.shape(series[0])[2])
    if in_z != 0:
        num_slices = num_slices[in_z:-in_z]
    if(num_slices.size <= 2 and in_z != 0):
        print(ind)
        print(num_slices)
        print(num)
    sub_rand = np.random.choice(num_slices, size=num, replace=False)

    center = transforms.CenterCrop(center_crop_sz)
    depth_center = transforms.DepthCenterCrop(out_z)
    series = [vol.astype(np.float) for vol in series]
    for i in sub_rand:
        if in_z == 0:
            nascent_series = [vol[:,:,i] for vol in series]
            nascent_seg_series = [seg[:,:,i] for seg in seg_series]
            nascent_series = np.expand_dims(nascent_series, axis=0)
            nascent_seg_series = np.expand_dims(nascent_seg_series, axis=0)
        else:
            nascent_series = [vol[:,:,i-in_z:i+1+in_z] for vol in series]
            assert nascent_series[0].shape[2] == in_z*2+1
            nascent_series = [np.squeeze(np.split(v,\
                    v.shape[2], axis=2)) for v in nascent_series]

            nascent_seg_series = [seg[:,:,i-in_z:i+1+in_z] for seg in \
                    seg_series]
            nascent_seg_series = [depth_center.engage(s) for s in \
                    nascent_seg_series]
            nascent_seg_series = [np.squeeze(np.split(s,\
                    s.shape[2], axis=2)) for s in nascent_seg_series]

            if out_z == 1:
                nascent_seg_series = \
                        np.expand_dims(nascent_seg_series, axis=0)

        # for s in nascent_seg_series:
            # print(np.max(s))
            # print(isinstance(np.max(s), np.int))
        if txforms is not None:
            for j in txforms:
                nascent_series, nascent_seg_series = \
                        j.engage(nascent_series, nascent_seg_series)
                bad = False
                for s in nascent_seg_series:
                    m = np.max(s)
                    if np.mod(m, 1) != 0:
                       bad = True
                if bad == True:
                    print(j)
                bad = False

            vols.append(np.squeeze(nascent_series))

            segs.append(np.squeeze(center.engage(nascent_seg_series, \
                    out_z > 1)))

    return vols, segs

def make_segs(segs):
    not_liver = np.zeros_like(segs)
    liver = np.zeros_like(segs)
    lesion = np.zeros_like(segs)

    not_liver[np.where(segs==0)] = 1
    liver[np.where(segs==1)] = 1
    lesion[np.where(segs==2)] = 1

    segs = np.stack([not_liver, liver, lesion])
    segs = np.transpose(segs, [1, 0, 2, 3])
    return segs

def gen_filepaths(path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break

    return f

def get_batch(volpath, segpath, batch_size, in_z, out_z, center_crop_sz,\
        series_names, seg_series_names,\
        txforms=None, workers=1, nrrd=True):

    pool = mp.Pool(processes=workers)
    fs = gen_filepaths(segpath)
    fv = gen_filepaths(volpath)

    inds = []
    volpaths = []
    segpaths = []
    for i in fs:
        if 'segmentation' in i:
            ind = int(re.findall('\d+', i)[0])
            inds.append(ind)

    inds = np.unique(inds)
    for i in inds:
        vol_files = []
        seg_files = []
        for name in series_names:
            for j in fv:
                if 'volume' in j and name in j:
                    ind0 = int(re.findall('\d+', j)[0])
                    if i == ind0:
                        vol_files.append(os.path.join(volpath,j))

        volpaths.append(vol_files)

        for name in seg_series_names:
            for j in fs:
                if 'segmentation' in j and name in j:
                    ind0 = int(re.findall('\d+', j)[0])
                    if i == ind0:
                        seg_files.append(os.path.join(segpath,j))
        segpaths.append(seg_files)

    ind_inds = np.arange(len(inds))

    inds = np.array(inds)
    vols = []
    segs = []

    rands = np.random.choice(ind_inds, batch_size)
    unique, counts = np.unique(rands, return_counts=True)
    intermediate = inds[unique]

    volpaths = np.array(volpaths)[unique]
    segpaths = np.array(segpaths)[unique]
    unique = intermediate

    volpaths = [list(a) for a in volpaths]
    vol_inds = np.repeat(unique, counts)

    args = list(zip(volpaths, segpaths,\
        unique, counts, [in_z]*len(unique), [out_z]*len(unique),\
        [center_crop_sz]*len(unique),\
        [series_names]*len(unique), [seg_series_names]*len(unique),\
        [txforms]*len(unique), [nrrd]*len(unique)))

    out = pool.starmap(open_nii, args)
    pool.close()
    vols, segs = list(zip(*out))

    vols = np.array(unpack(vols))
    segs = np.array(unpack(segs))

    vols = rot_and_flip(vols)
    vols = torch.from_numpy(vols).float()
    vols = vols-torch.min(vols)
    vols = vols/torch.max(vols)
    segs = torch.from_numpy(np.round(rot_and_flip(segs)))
    # return vols.unsqueeze(1), segs.long(), vol_inds
    return vols, segs.long(), vol_inds

def get_segs(volpath, segpath, ind, nrrd=True):
    volpath = os.path.join(volpath, 'volume-' + str(ind) + '.nii')
    if nrrd:
        segpath = os.path.join(segpath, 'segmentation-' + str(ind) + '.seg.nrrd')
        _, seg = get_nii_nrrd(volpath, segpath)
    else:
        segpath = os.path.join(segpath, 'segmentation-' + str(ind) + '.nii')
        _, seg = get_nii_nii(volpath, segpath)

    counts = np.bincount(seg.flatten())
    if counts.shape[0] == 2:
        counts = np.insert(counts,counts.size,0)
    return counts

def get_weights(volpath, segpath, workers, clip_val=1, nrrd=True):

    pool = mp.Pool(processes=workers)
    f = gen_filepaths(segpath)

    inds = []
    for i in f:
        if 'segmentation' in i:
            inds.append(int(re.findall('\d+', i)[0]))

    inds = np.array(inds)

    args = list(zip([volpath]*len(inds), [segpath]*len(inds),\
        inds, [nrrd]*len(inds)))

    out = pool.starmap(get_segs, args)
    pool.close()

    counts = sum(out)
    counts_collapsed = np.zeros(clip_val+1)
    i = 0
    while i < clip_val:
        counts_collapsed[i] = counts[i]
        i += 1
    counts_collapsed[clip_val] = np.sum(counts[clip_val:])
    counts = counts_collapsed
    s = sum(counts)
    counts = [a/s for a in counts]
    A = np.diagflat(counts)
    B = np.ones((len(counts), 1))
    B = B/sum(B)
    Ai = np.linalg.inv(A)
    p = np.dot(Ai, B)
    p = [a/sum(p) for a in p]
    w = np.stack(p)
    return w
