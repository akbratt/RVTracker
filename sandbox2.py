import nibabel as nib
import numpy as np
import resample
import pyqtgraph as pg
import PyQt5.QtCore as QtCore
import vol_display
from nilearn.image import new_img_like
from PIL import Image

outpath = '/home/alex/LiverSegmentation/3d_print_demo/volume-0.nii'
path = '/home/alex/LiverSegmentation/3d_print_demo/volume-0-dcm.nii'

volpath = '/home/alex/LiverSegmentation/3d_print_demo/tvol-0.nii'
segpath = '/home/alex/LiverSegmentation/3d_print_demo/tout-0.nii'

outpath2 = '/home/alex/LiverSegmentation/3d_print_demo/tout-0-corrected.nii'
path2 = '/home/alex/LiverSegmentation/3d_print_demo/tout-0.nii'
path3 = '/home/alex/LiverSegmentation/3d_print_demo/volume-0.nii'

nibvol = nib.load(outpath)

#
#vol = np.transpose(vol, [2,0,1])
#a, b, c = vol_display.masked_display(vol, vol)
# a, b = vol_display.vol_display(nibvol)

vol = nib.load(outpath)
seg = nib.load(segpath)
c, d, e = vol_display.masked_display(vol, seg)


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
