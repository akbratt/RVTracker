import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import scipy.ndimage as nd
import pyqtgraph as pg
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import multiprocessing as mp
from skimage import color

def get_vol(volpath):
    vol = np.transpose(nib.as_closest_canonical(nib.load(volpath)).get_data(), [2,0,1])
    vol = np.flip(vol,1)
#    vol = np.flip(vol,2)
    return vol

def nii2npy(vol):
    v = np.transpose(nib.as_closest_canonical(vol).get_data(), [2,0,1])
    v = np.flip(v, 1)
    return v

def screen_type(vol):
    v = vol
    if type(vol) == nib.nifti1.Nifti1Image:
        v = nii2npy(v)

    return v

def to_RGBA(vol):
    vol = vol - np.min(vol)
    vol = vol/np.max(vol) + 1e-4
    vol *= 256
    vol = np.stack((vol,vol,vol,vol))
#    vol = np.transpose(vol,[1,2,3,0]).astype(np.int)
    vol = np.transpose(vol, [1,2,0]).astype(np.int)
    return vol

def clip(img, window, level):
    upper = level + window[0]
    lower = level - window[1]
    # print(window[1])
    # print(level)
    out = np.clip(img, lower, upper)
    out = to_RGBA(out)
    out[:,:,0:3] = 255
    return out

class PACSViewbox_with_mask(pg.ViewBox):
    def __init__(self, vol, seg, img, img2, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.vol = np.array(vol)
        self.seg = np.array(seg)
        self.z = self.vol.shape[0]
        self.img = img
        self.img2 = img2
        self.i = 0
        self.windowlower = (np.max(vol) - np.min(vol))*0.25
        self.windowupper = (np.max(vol) - np.min(vol))*0.25
        self.level = np.mean(vol)
        img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                self.windowlower], self.level))
        img2.setImage(self.seg[self.i,:,:])

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
#        self.img.setImage(vol[i,:,:])
            self.i += 1
        if ev.button() == QtCore.Qt.RightButton:
            self.i -= 1

        ev.accept()

        if self.i >= self.z:
            self.i = self.z - 1
        if self.i <= 0:
            self.i = 0

        self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                self.windowlower], self.level))
        self.img2.setImage(self.seg[self.i,:,:])

    def mouseDragEvent(self, ev):
        ev.accept()
        pos = ev.pos()
        prev = ev.lastPos()

        if ev.isFinish():
            return
        elif ev.button() == 2:
            self.i -= int(pos[1] - prev[1])
            if self.i >= self.z:
                self.i = self.z - 1
            if self.i <= 0:
                self.i = 0
#            print(pos[1] - prev[1])
            self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                    self.windowlower], self.level))
            self.img2.setImage(self.seg[self.i,:,:])
        else:
            f = 0.001

            add = int(pos[1] - prev[1])*f
            if self.level + self.windowupper + add < np.max(self.vol):
                self.windowupper += add

            if self.level - self.windowlower - add > np.min(self.vol):
                self.windowlower -= add

            add2 = int(pos[0] - prev[0])*f
            if self.level + add2 + self.windowupper < np.max(self.vol) and\
                    self.level + add2 - self.windowlower > np.min(self.vol):
                        self.level += add2

            self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                    self.windowlower], self.level))
            self.img2.setImage(self.seg[self.i,:,:])


    def wheelEvent(self, ev):
        ev.accept()
#        pos = ev.pos()
#        prev = ev.lastPos()

        self.i -= np.sign(ev.delta())
        if self.i >= self.z:
            self.i = self.z - 1
        if self.i <= 0:
            self.i = 0
#            print(pos[1] - prev[1])
        self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                self.windowlower], self.level))
        self.img2.setImage(self.seg[self.i,:,:])

class PACSViewbox_without_mask(pg.ViewBox):
    def __init__(self, vol, img, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.vol = np.array(vol)
        self.z = self.vol.shape[0]
        self.img = img
        self.i = 0
        self.windowlower = (np.max(vol) - np.min(vol))*0.25
        self.windowupper = (np.max(vol) - np.min(vol))*0.25
        self.level = np.mean(vol)
        img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                self.windowlower], self.level))

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
#        self.img.setImage(vol[i,:,:])
            self.i += 1
        if ev.button() == QtCore.Qt.RightButton:
            self.i -= 1

        ev.accept()

        if self.i >= self.z:
            self.i = self.z - 1
        if self.i <= 0:
            self.i = 0

        self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                    self.windowlower], self.level))

    def mouseDragEvent(self, ev):
        ev.accept()
        pos = ev.pos()
        prev = ev.lastPos()

        if ev.isFinish():
            return
        elif ev.button() == 2:
            self.i -= int(pos[1] - prev[1])
            if self.i >= self.z:
                self.i = self.z - 1
            if self.i <= 0:
                self.i = 0
#            print(pos[1] - prev[1])

            self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                    self.windowlower], self.level))
        else:
            f = 0.001

            add = int(pos[1] - prev[1])*f
            if self.level + self.windowupper + add < np.max(self.vol):
                self.windowupper += add

            if self.level - self.windowlower - add > np.min(self.vol):
                self.windowlower -= add

            add2 = int(pos[0] - prev[0])*f
            if self.level + add2 + self.windowupper < np.max(self.vol) and\
                    self.level + add2 - self.windowlower > np.min(self.vol):
                        self.level += add2

            # print(str(self.windowupper) + ' ' + str(self.windowlower) + ' ' +\
                    # str(self.level))

            self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                    self.windowlower], self.level))

    def wheelEvent(self, ev):
        ev.accept()
#        pos = ev.pos()
#        prev = ev.lastPos()

        self.i -= np.sign(ev.delta())
        if self.i >= self.z:
            self.i = self.z - 1
        if self.i <= 0:
            self.i = 0
#            print(pos[1] - prev[1])
        self.img.setImage(clip(self.vol[self.i,:,:], [self.windowupper,\
                    self.windowlower], self.level))

def masked_display(vol, seg):
    vol = screen_type(vol)
#    vol = to_RGBA(vol)
#    vol[:,:,:,0:3] = 255

#    seg = get_vol(segpath)
    seg = screen_type(seg)
    r_mask = np.zeros((seg.shape[1], seg.shape[2], 4))
    colors = np.array([[1,0,0,0.3], [0,1,0,0.3], [0,0,1,0.3]])
    for i in np.arange(1,4):
        r_mask = np.where(np.expand_dims(seg, -1)==i,colors[i-1],r_mask)

#    print(time.time()-t0)


    w = pg.GraphicsWindow()
#    w.setWindowTitle('pyqtgraph example: GraphItem')
    img = pg.ImageItem()
    img2 = pg.ImageItem()
    view = PACSViewbox_with_mask(vol, r_mask, img, img2)

    view.setAspectLocked(True)

    view.addItem(img)
    view.addItem(img2)

    w.addItem(view)
    return w, img, img2

def vol_display(vol):
    vol = screen_type(vol)
#    vol = to_RGBA(vol)
#    vol[:,:,:,0:3] = 255

    w = pg.GraphicsWindow()
    img = pg.ImageItem()
    view = PACSViewbox_without_mask(vol, img)

    view.setAspectLocked(True)

    view.addItem(img)

    w.addItem(view)
    return w, img

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
