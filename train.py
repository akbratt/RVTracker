import torch
import transforms
import time
import preprocess
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import utils
import models
import sys
import os

###################### Network Parameters ######################

batch_size = 48
checkpoint_dir = 'checkpont directory'
volpath = 'path to image volumes'
segpath = 'path to segmentation masks'
num_labels = 3 #labels: lateral annulus, medial annulus, background

crop_size = 224
original_size = 256
num_steps = 1000000000 #number of training steps

################################################################

sqr = transforms.Square()
aff = transforms.Affine()
crop = transforms.RandomCrop(crop_size)
scale = transforms.Scale(original_size)
rotate = transforms.Rotate(0.5, 30)
noise = transforms.Noise(0.02)
transform_plan = [sqr, scale, aff, rotate, crop, noise]
lr = 1e-4
series_names = ['echo']
seg_series_names = ['echo']


model = models.Net23(3)

model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=lr)

out_z, center_crop_sz = utils.get_out_size(original_size, 0,\
        transform_plan, model)

t0 = time.time()

counter = 0
print_interval = 200
model.train()
weight = torch.FloatTensor([0.1, 0.45, 0.45]).cuda()
for i in range(num_steps):
    vol, seg, inds = preprocess.get_batch(volpath, segpath, batch_size, 0,\
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

        torch.save(model.state_dict(),os.path.join(checkpoint_dir, 'checkpoint.pth'))

        print(('\rStep {} completed in {:.2f} sec ; Loss = {:.2f}')\
                .format(i*batch_size,time.time()-t0, loss.data.cpu().item()))

        t0 = time.time()
