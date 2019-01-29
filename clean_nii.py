import nibabel as nib
import os
import numpy as np
import multiprocessing
import time
# import transforms

out_path = '/home/alex/samsung_256/LITS/train_corrected/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

in_path = '/home/alex/samsung_256/LITS/train/'

# d_pad = transforms.DepthPad(10)

def generateFilepaths():
    for root, dirs, files in os.walk(in_path):
#        print(root)
        if len(files) > 0:
            for i in files:
                image_path = os.path.join(root, i)
                yield image_path

def processFile(image_path):
    ind = int(image_path.split('-')[1].split('.')[0])

    vol = nib.as_closest_canonical(nib.load(image_path))
    v = vol.get_data()
    # thickness = vol.header['pixdim'][3]
    # thickness_ratio = 5./thickness
    # v = nd.interpolation.zoom(v, zoom=[1,1,1./thickness_ratio], order=1)
#    v = np.flip(v,0)
    if ind >= 28 and ind <= 47:
        v = np.flip(v, 0)

    if ind >= 48 and ind <= 52 and 'segmentation' in image_path:
        v = np.flip(v,0)

    # v, _ = d_pad.engage(v, v, 0, 0)

    out = nib.Nifti1Image(v.astype(np.int16), np.eye(4))
    out_pth = out_path + image_path.split('/')[-1]
    # out_path = image_path.replace('train', 'train_corrected')
    nib.save(out, out_pth)

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

        if i % 1 == 0:
            print('File: {}; Block Time: {:2f}'.format(i, time.time() - tblock))
            tblock = time.time()

#        print(fullCrc)

print('Processed {} images in {:.2f} sec'.format(i, time.time() - t0))
