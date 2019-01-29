import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.utils.weight_norm as weight_norm

d = 0.0

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, x1, x2):
        cat = torch.cat([x1,x2], dim=1)
        return cat

class RM_2d(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(RM_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

class EncoderNorm_2d(nn.Module):
    def __init__(self, channels):
        super(EncoderNorm_2d, self).__init__()
        self.bn = nn.InstanceNorm2d(channels)
        # self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(x)

class RM_Envelope_2d(nn.Module):
    def __init__(self, rsmp0, rsmp1, in_chan, out_chan):
        super(RM_Envelope_2d, self).__init__()
        self.cat = Concat()

        self.down = nn.Upsample(rsmp0, mode='bilinear')
        self.rm = RM_2d(in_chan + 1, out_chan)
        self.up = nn.Upsample(rsmp1, mode='bilinear')

    def forward(self, x, f):
        f0 = self.cat(self.down(x), f)
        f0 = self.rm(f0)
        f0 = self.up(f0)

        return f0

class CRF_2d(nn.Module):
    def __init__(self):
        super(CRF_2d, self).__init__()
        # self.resample_sizes = ((2,8,8), (4,16,16), (5,32,32),\
                # (7,64,64),(9,128,128),(10,256,256),(13,512,512),(13,512,512))
        self.resample_sizes = ((8,8),(16,16),(32,32),(64,64),(128,128),\
                (128,128),(256,256),(256,256),(512,512),(512,512))

        # self.chans = [1, 512, 256, 128, 64, 32, 16, 8]
        self.chans = [1,1024,512,256,128,64,32,16,8,4]
        self.cat = Concat()

        self.down0 = nn.Upsample(self.resample_sizes[0], mode='bilinear')
        self.rm0 = RM_2d(self.chans[0],self.chans[1])
        self.up0 = nn.Upsample(self.resample_sizes[1], mode='bilinear')

        # self.f = [RM_Envelope(self.resample_sizes[a+1],\
                # self.resample_sizes[a+2], self.chans[a+1],\
                # self.chans[a+2]).cuda() for a in range(1)]

        self.f1 = RM_Envelope_2d(self.resample_sizes[1], self.resample_sizes[2],\
                self.chans[1], self.chans[2])

        self.f2 = RM_Envelope_2d(self.resample_sizes[2], self.resample_sizes[3],\
                self.chans[2], self.chans[3])

        self.f3 = RM_Envelope_2d(self.resample_sizes[3], self.resample_sizes[4],\
                self.chans[3], self.chans[4])

        self.f4 = RM_Envelope_2d(self.resample_sizes[4], self.resample_sizes[5],\
                self.chans[4], self.chans[5])

        self.f5 = RM_Envelope_2d(self.resample_sizes[5], self.resample_sizes[6],\
                self.chans[5], self.chans[6])

        self.f6 = RM_Envelope_2d(self.resample_sizes[6], self.resample_sizes[7],\
                self.chans[6], self.chans[7])

        self.f7 = RM_Envelope_2d(self.resample_sizes[7], self.resample_sizes[8],\
                self.chans[7], self.chans[8])

        self.f8 = RM_Envelope_2d(self.resample_sizes[8], self.resample_sizes[9],\
                self.chans[8], self.chans[9])

        self.f_end = nn.Conv2d(self.chans[9], 2, 3, padding=(1,1))

    def forward(self, x):
        f0 = self.down0(x)
        f0 = self.rm0(f0)
        f0 = self.up0(f0)

        f1 = self.f1(x, f0)
        f2 = self.f2(x, f1)
        f3 = self.f3(x, f2)
        f4 = self.f4(x, f3)
        f5 = self.f5(x, f4)
        f6 = self.f6(x, f5)
        f7 = self.f7(x, f6)
        f8 = self.f8(x, f7)
        f_end = self.f_end(f8)


        return f_end

class Res_Down(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Down, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)

class Res_Up(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Up, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)

class Res_Final(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Final, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, in_chan, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(in_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 3, padding=(1,1))
        # self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.conv2(x + out)

        return out


class Net23(nn.Module):

    def __init__(self, outs):
        super(Net23, self).__init__()

        self.cat = Concat()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(64)

        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(64)

        self.Rd1 = Res_Down(64, 128)
        self.Rd2 = Res_Down(128, 256)
        self.Rd3 = Res_Down(256, 512)
        self.Rd4 = Res_Down(512, 512)

        self.fudge = nn.ConvTranspose2d(512, 256, 4, stride = (2,2),\
                padding = (1,1))

        self.Ru3 = Res_Up(512,512)
        self.Ru2 = Res_Up(512,256)
        self.Ru1 = Res_Up(256,128)
        self.Ru0 = Res_Up(128,64)

        self.Rf = Res_Final(64,outs)


    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        e0 = F.relu(self.bn2(self.conv2(out)))

        e1 = self.Rd1(e0)
        e2 = self.Rd2(e1)
        e3 = self.Rd3(e2)
        e4 = self.Rd4(e3)

        d3 = self.Ru3(e4)
        d2 = self.Ru2(self.cat(d3[:,256:],e3[:,256:]))
        d1 = self.Ru1(self.cat(d2[:,128:],e2[:,128:]))
        d0 = self.Ru0(self.cat(d1[:,64:],e1[:,64:]))

        out = self.Rf(self.cat(e0[:,32:],d0[:,32:]))

        return out

class Res_Down_wn(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Down_wn, self).__init__()

        self.conv1 = weight_norm(nn.Conv2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1)))
        self.drop = nn.Dropout2d(d)

        self.conv2 = weight_norm(nn.Conv2d(out_chan, out_chan, 3,
            padding=(1,1)))


    def forward(self, x):
        x = self.drop(F.relu(self.conv1(x)))
        out = self.conv2(x)

        return F.relu(x + out)

class Res_Up_wn(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Up_wn, self).__init__()

        self.conv1 = weight_norm(nn.ConvTranspose2d(in_chan, out_chan, 4,\
                stride = (2,2), padding=(1,1)))
        self.drop = nn.Dropout2d(d)

        self.conv2 = weight_norm(nn.Conv2d(out_chan, out_chan, 3,
            padding=(1,1)))


    def forward(self, x):
        x = self.drop(F.relu(self.conv1(x)))
        out = self.conv2(x)

        return F.relu(x + out)

class Res_Final_wn(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Final_wn, self).__init__()

        self.conv1 = weight_norm(nn.Conv2d(in_chan, in_chan, 3, padding=(1,1)))
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 3, padding=(1,1))
        # self.bn2 = EncoderNorm_2d(out_chan)

    def forward(self, x):
        out = self.drop(F.relu(self.conv1(x)))
        out = self.conv2(x + out)

        return out

class Net23_wn(nn.Module):

    def __init__(self):
        super(Net23_wn, self).__init__()

        self.cat = Concat()

        self.conv1 = weight_norm(nn.Conv2d(7, 64, 3, padding=(1,1)))

        self.drop = nn.Dropout2d(d)

        self.conv2 = weight_norm(nn.Conv2d(64, 64, 3, padding=(1,1)))

        self.Rd1 = Res_Down_wn(64, 128)
        self.Rd2 = Res_Down_wn(128, 256)
        self.Rd3 = Res_Down_wn(256, 512)
        self.Rd4 = Res_Down(512, 512)

        self.Ru3 = Res_Up_wn(512,512)
        self.Ru2 = Res_Up_wn(512,256)
        self.Ru1 = Res_Up_wn(256,128)
        self.Ru0 = Res_Up_wn(128,64)

        self.Rf = Res_Final_wn(64,2)


    def forward(self, x):
        out = self.drop(F.relu(self.conv1(x)))
        e0 = F.relu(self.conv2(out))

        e1 = self.Rd1(e0)
        e2 = self.Rd2(e1)
        e3 = self.Rd3(e2)
        e4 = self.Rd4(e3)

        d3 = self.Ru3(e4)
        d2 = self.Ru2(self.cat(d3[:,256:],e3[:,256:]))
        d1 = self.Ru1(self.cat(d2[:,128:],e2[:,128:]))
        d0 = self.Ru0(self.cat(d1[:,64:],e1[:,64:]))

        out = self.Rf(self.cat(e0[:,32:],d0[:,32:]))

        return out

class Res_Down_Orig(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Down_Orig, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(x)))

        return x + out

class Res_Up_Orig(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Up_Orig, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, 4, stride = (2,2),\
                padding=(1,1))
        self.bn1 = EncoderNorm_2d(out_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(x)))

        return x + out

class Res_Final_Orig(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Res_Final_Orig, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, in_chan, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(in_chan)
        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 3, padding=(1,1))
        # self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.conv2(x + out)

        return out

class Net23_Orig(nn.Module):

    def __init__(self):
        super(Net23_Orig, self).__init__()

        self.cat = Concat()

        self.conv1 = nn.Conv2d(7, 64, 3, padding=(1,1))
        self.bn1 = EncoderNorm_2d(64)

        self.drop = nn.Dropout2d(d)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.bn2 = EncoderNorm_2d(64)

        self.Rd1 = Res_Down_Orig(64, 128)
        self.Rd2 = Res_Down_Orig(128, 256)
        self.Rd3 = Res_Down_Orig(256, 512)
        self.Rd4 = Res_Down_Orig(512, 512)

        self.fudge = nn.ConvTranspose2d(512, 256, 4, stride = (2,2),\
                padding = (1,1))

        self.Ru3 = Res_Up_Orig(512,512)
        self.Ru2 = Res_Up_Orig(512,256)
        self.Ru1 = Res_Up_Orig(256,128)
        self.Ru0 = Res_Up_Orig(128,64)

        self.Rf = Res_Final_Orig(64,2)


    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        e0 = F.relu(self.bn2(self.conv2(out)))

        e1 = self.Rd1(e0)
        e2 = self.Rd2(e1)
        e3 = self.Rd3(e2)
        e4 = self.Rd4(e3)

        d3 = self.Ru3(e4)
        d2 = self.Ru2(self.cat(d3[:,256:],e3[:,256:]))
        d1 = self.Ru1(self.cat(d2[:,128:],e2[:,128:]))
        d0 = self.Ru0(self.cat(d1[:,64:],e1[:,64:]))

        out = self.Rf(self.cat(e0[:,32:],d0[:,32:]))

        return out
