import pandas as pd
import numpy as np
import gdown
import os
import time
# import torch.nn as nn
from models import Generator, ImageDatasetCV
from config import config
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms as T
from trainer import denorm


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    SEED = 42
    RADNOM_STATE = SEED


    INPUT_DIR = './input'
    OUTPUT_DIR = './output'


    if not os.path.exists("input"):
        os.makedirs("input")
    if not os.path.exists("output"):
        os.makedirs("output")


    faces_dataset = ImageDatasetCV(anime_dir=None, faces_dir=INPUT_DIR)

    faces_dataloader = DataLoader(faces_dataset, batch_size=1)
    
    if not os.path.isfile('anime_model_.ckpt'):
        gdown.download('https://drive.google.com/uc?id=1-B695LVszuTzwaBja74oBjXS41w9Fcct', 'anime_model_.ckpt')
        

    model = Generator(config.nc, config.nc)
    checkpoint = torch.load('anime_model_.ckpt', map_location=None)
    model.load_state_dict(checkpoint['gen_f2a'])

    with torch.no_grad():

        j = 0
        for faces in faces_dataloader:
            model.to(device)
            faces = faces.to(device)
            anime_gen = model(faces).cpu()
            output_image = vutils.make_grid([torch.squeeze(denorm(anime_gen))], nrow=1)

            output_image_pil = T.ToPILImage()(output_image)

            # display(output_image_pil)
            output_image_pil.save(f'{OUTPUT_DIR}/out.png', "PNG")

if __name__ == '__main__':
    main()