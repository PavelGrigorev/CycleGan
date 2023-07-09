import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from config import config

class ImageDatasetCV(Dataset):

    def __init__(self, anime_dir, faces_dir, size=[256, 256], normalize=True):

        super().__init__()

        self.anime_dir = None

        if anime_dir:

            self.anime_dir = anime_dir
            self.anime_idx = dict()

            for i, filename in enumerate(os.listdir(self.anime_dir)):
                self.anime_idx[i] = filename


        self.faces_dir = faces_dir
        self.faces_idx = dict()

        for i, filename in enumerate(os.listdir(self.faces_dir)):
            self.faces_idx[i] = filename


        if normalize:
            self.transforms = T.Compose([T.Resize(size),
                                         T.ToTensor(),
                                         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        else:
            self.transforms = T.Compose([T.Resize(size),
                                         T.ToTensor()
                                        ])


    def __getitem__(self, idx):

        random_idx = idx

        if self.anime_dir:

            random_idx = int(np.random.uniform(0, len(self.anime_idx.keys())))

            anime_path = os.path.join(self.anime_dir, self.anime_idx[random_idx])
            anime_img = self.transforms(Image.open(anime_path))


        faces_path = os.path.join(self.faces_dir, self.faces_idx[random_idx])
        faces_img = self.transforms(Image.open(faces_path))

        if self.anime_dir:
            return faces_img, anime_img
        else:
            return faces_img


    def __len__(self):

        if self.anime_dir:
            return min(len(self.anime_idx.keys()), len(self.faces_idx.keys()))
        else:
            return len(self.faces_idx.keys())

def Convlayer(in_ch, out_ch, kernel_size=3, stride=2,
              use_leaky=True, use_inst_norm=True, use_pad=True):


    # Convolution
    if use_pad:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 1, bias=True)
    else:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, bias=True)


    # Activation Function
    if use_leaky:
        actv = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        actv = nn.GELU()


    # Normalization
    if use_inst_norm:
        norm = nn.InstanceNorm2d(out_ch)
    else:
        norm = nn.BatchNorm2d(out_ch)


    return nn.Sequential(conv,
                         norm,
                         actv)

def Upsample(in_ch, out_ch, use_dropout=False, dropout_ratio=0.5):

    # Transposed Convolution
    convtrans = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)

    # Normalization
    norm = nn.InstanceNorm2d(out_ch)

    # Activatin Function
    actv = nn.GELU()


    if use_dropout:

        # Dropout layer
        drop = nn.Dropout(dropout_ratio)

        return nn.Sequential(convtrans,
                             norm,
                             drop,
                             actv)

    else:
        return nn.Sequential(convtrans,
                             norm,
                             actv)


class Resblock(nn.Module):

    def __init__(self, in_features, use_dropout=False, dropout_ratio=0.5):
        super().__init__()

        layers = list()

        # Padding
        layers.append(nn.ReflectionPad2d(1))

        # Convolution layer
        layers.append(Convlayer(in_features, in_features, 3, 1, use_leaky=False, use_pad=False))

        # Dropout
        if use_dropout:
            layers.append(nn.Dropout(dropout_ratio))

        # Padding
        layers.append(nn.ReflectionPad2d(1))

        # Convolution
        layers.append(nn.Conv2d(in_features, in_features, 3, 1, padding=0, bias=True))

        # Normalization
        layers.append(nn.InstanceNorm2d(in_features))


        self.res = nn.Sequential(*layers)


    def forward(self, x):

        return x + self.res(x)

class Generator(nn.Module):

    def __init__(self, in_ch, out_ch, num_res_blocks=6):

        super().__init__()

        model = list()

        # Padding layer
        model.append(nn.ReflectionPad2d(3))


        # Convolution input_channels -> 64
        model.append(Convlayer(in_ch=in_ch, out_ch=64,
                               kernel_size=7, stride=1,
                               use_leaky=False,
                               use_inst_norm=True,
                               use_pad=False))

        # Convolution 64 -> 128
        model.append(Convlayer(in_ch=64, out_ch=128,
                               kernel_size=3, stride=2,
                               use_leaky=False,
                               use_inst_norm=True,
                               use_pad=True))

        # Convolution 128 -> 256
        model.append(Convlayer(in_ch=128, out_ch=256,
                               kernel_size=3, stride=2,
                               use_leaky=False,
                               use_inst_norm=True,
                               use_pad=True))

        # Residual Block
        for _ in range(num_res_blocks):
            model.append(Resblock(in_features=256))


        # Transposed convolution 256 -> 128
        model.append(Upsample(in_ch=256, out_ch=128))

        # Transposed convolution 128 -> 256
        model.append(Upsample(in_ch=128, out_ch=64))

        # Padding Layer
        model.append(nn.ReflectionPad2d(3))

        # Convolutional layer
        model.append(nn.Conv2d(in_channels=64, out_channels=out_ch,
                               kernel_size=7, padding=0))

        # Activation function Tanh
        model.append(nn.Tanh())


        self.gen = nn.Sequential(*model)


    def forward(self, x):

        return self.gen(x)


class Discriminator(nn.Module):

    def __init__(self, in_ch, num_layers=4):

        super().__init__()

        model = list()

        # Convolution in_channels -> 64
        model.append(nn.Conv2d(in_channels=in_ch, out_channels=64,
                               kernel_size=4, stride=2, padding=1))


        # Convolutions i=1:  64 -> 64
        #              i=2:  64 -> 128
        #              i=3: 128 -> 256
        #              i=4: 256 -> 512

        for i in range(1, num_layers):
            in_chs = 64 * 2**(i-1)
            out_chs = in_chs * 2

            if i == num_layers - 1:
                model.append(Convlayer(in_chs, out_chs,
                                       kernel_size=4, stride=1))

            else:
                model.append(Convlayer(in_chs, out_chs,
                                       kernel_size=4, stride=2))

        # Convolution 512 -> 1
        model.append(nn.Conv2d(in_channels=512, out_channels=1,
                               kernel_size=4, stride=1, padding=1))

        self.disc = nn.Sequential(*model)


    def forward(self, x):

        return self.disc(x)
