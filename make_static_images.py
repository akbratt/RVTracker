import nibabel as nib
import os
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt
import utils
import preprocess

out_path = '/home/alex/samsung_256/LITS/static_images/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

in_path = '/home/alex/samsung_256/LITS/test/'

def generateFilepaths():
    for root, dirs, files in os.walk(in_path):
#        print(root)
        if len(files) > 0:
            for i in files:
                image_path = os.path.join(root, i)
                if 'volume' in image_path:
                    yield image_path

def processFile(image_path):
    ind = int(image_path.split('-')[1].split('.')[0])

    vol = nib.as_closest_canonical(nib.load(image_path))
    # seg = nib.as_closest_canonical(nib.load(image_path.replace(\
            # 'volume', 'segmentation')))
    vol = vol.get_data().astype(np.int16)
    # seg = seg.get_data().astype(np.int16)
    # num_labels = np.bincount(seg.flatten()).shape[0]


    vol = np.transpose(vol, [2,0,1])
    # seg = np.transpose(seg, [2,0,1])

    # line_sum = np.sum(seg, axis=(1,2))
    # i = np.argmax(line_sum)
    # v = vol[i,:,:]
    # s = seg[i,:,:]
    v = vol[75,:,:]


    v = preprocess.rot_and_flip(v)
    v = v.astype(np.float)
    v = v - np.min(v)
    v = v / np.max(v)
    # s = preprocess.rot_and_flip(s)

    # masked = utils.makeMask(v,s,num_labels,0.5)
    # plt.imshow(masked)
    plt.imshow(v, cmap='Greys_r')
    plt.savefig(out_path + str(ind))


if __name__ == '__main__':
    pool = multiprocessing.Pool(8)

    t0 = time.time()
    fullCrc = 0
    tblock = time.time()
    for i, crc in enumerate(pool.imap_unordered(processFile, generateFilepaths())):
        try:
            if crc is not None:
                fullCrc ^= crc
        except ValueError:
            print('oops')

        if i % 10 == 0:

            print('File: {}; Block Time: {:2f}'.format(i, time.time() - tblock))
            tblock = time.time()

#        print(fullCrc)

print('Processed {} images in {:.2f} sec'.format(i, time.time() - t0))
