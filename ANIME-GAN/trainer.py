import pandas as pd
import numpy as np
import gdown
import os
import time
import torch.nn as nn
from models import Generator, Discriminator, ImageDatasetCV
from config import config

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.init as init

import torchvision.utils as vutils
from torchvision import transforms as T
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from IPython.display import clear_output, display

import itertools
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5
    
def show_train_images(tensor_image1, tensor_image2, tensor_image3, epoch):

    combined_image = vutils.make_grid([torch.squeeze(denorm(tensor_image1)),
                                       torch.squeeze(denorm(tensor_image2))
                                       ,torch.squeeze(denorm(tensor_image3))], nrow=3)

    combined_image_pil = T.ToPILImage()(combined_image)

    # display(combined_image_pil)
    combined_image_pil.save(f'{config.result_dir}/{epoch}.png', "PNG")
    
def plot_loss_history(log):
    df = pd.read_csv(log)
    # Create the traces
    trace1 = go.Scatter(x=df.epoch, y=df.loss_disc_total_avg, mode="lines", name='Total Discriminator')
    trace2 = go.Scatter(x=df.epoch, y=df.loss_disc_faces_avg, mode="lines", name='Photo Discriminator')
    trace3 = go.Scatter(x=df.epoch, y=df.loss_disc_anime_avg, mode="lines", name='Monet Discriminator')

    # Create the layout
    layout = go.Layout(title="Losses Discriminator", xaxis=dict(title="epoch"), yaxis=dict(title="loss"))

    # Create the figure with both traces
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    # Show the figure
    fig.show()
        # Create the traces
    trace1 = go.Scatter(x=df.epoch, y=df.loss_gen_total_avg, mode="lines", name='Total Generator')
    trace2 = go.Scatter(x=df.epoch, y=df.loss_gen_faces_avg, mode="lines", name='Photo Generator')
    trace3 = go.Scatter(x=df.epoch, y=df.loss_gen_anime_avg, mode="lines", name='Monet Generator')

    # Create the layout
    layout = go.Layout(title="Losses Generator", xaxis=dict(title="epoch"), yaxis=dict(title="loss"))

    # Create the figure with both traces
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    # Show the figure
    fig.show()

class CycleGAN(object):

    def __init__(self, in_ch, out_ch, epochs, device, start_lr=config.lr, lmbda=10, idt_coef=0.5, decay_epoch=0):

        # Regularization coefficients
        self.lmbda = lmbda
        self.idt_coef = idt_coef

        # Set device
        self.device = device

        # Generator Anime -> Face
        self.gen_a2f = Generator(in_ch, out_ch)

        # Generator Face -> Anime
        self.gen_f2a = Generator(in_ch, out_ch)

        # discriminator for Anime-generated images
        self.disc_m = Discriminator(in_ch)

        # discriminator for Face-generated images
        self.disc_p = Discriminator(in_ch)

        # Initialize model weights
        self.init_models()

        # Optimizator for generators
        self.adam_gen = torch.optim.Adam(itertools.chain(self.gen_a2f.parameters(),
                                                         self.gen_f2a.parameters()),
                                         lr=start_lr, betas=(config.beta, 0.999))

        # Optimizator for discriminator
        self.adam_disc = torch.optim.Adam(itertools.chain(self.disc_m.parameters(),
                                                          self.disc_p.parameters()),
                                         lr=start_lr, betas=(config.beta, 0.999))

        # Set number of epochs and start of learning rate decay
        self.epochs = epochs

        # Set decay epoch
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(self.epochs/2)

        # Set rule for learning step decay
        lambda_decay = lambda epoch: start_lr/(epoch-self.decay_epoch) if epoch > self.decay_epoch else start_lr


        # Define scheduler for generator and discriminator
        self.scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.adam_gen, lr_lambda=lambda_decay)
        self.scheduler_disc = torch.optim.lr_scheduler.LambdaLR(self.adam_disc, lr_lambda=lambda_decay)


    # Initialize weights
    def init_weights(self, net, gain=0.02):

        def init_func(m):

            # Name of the class
            classname = m.__class__.__name__

            # If class has attribute "weight" (to initialize) and
            # has either convolutional layer or linear
            if hasattr(m, 'weight') and \
               (classname.find('Conv') != -1 or classname.find('Linear') != -1):

                # Initialize weights with values drawn from normal distribution N(mean, std)
                init.normal_(m.weight.data, mean=0.0, std=gain)

                # Set bias value with constant val
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, val=0.0)

            # Initialize BatchNorm weights
            elif classname.find('BatchNorm2d') != -1:

                init.normal_(m.weight.data, mean=1.0, std=gain)
                init.constant_(m.bias.data, val=0.0)

        # Apply weight initialization to every submodule of model
        net.apply(init_func)


    # Initialize models
    def init_models(self):

        # Initialize weights
        self.init_weights(self.gen_a2f)
        self.init_weights(self.gen_f2a)
        self.init_weights(self.disc_m)
        self.init_weights(self.disc_p)

        # Set device for models
        self.gen_a2f = self.gen_a2f.to(self.device)
        self.gen_f2a = self.gen_f2a.to(self.device)
        self.disc_m = self.disc_m.to(self.device)
        self.disc_p = self.disc_p.to(self.device)



    # Enable/ disable gradients for model parameters
    def param_require_grad(self, models, requires_grad=True):
        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad


    # Cycle generation: x -> y_gen -> x_cycle
    def cycle_gen(self, x, G_x_to_y, G_y_to_x):

        y_gen = G_x_to_y(x)
        x_cycle = G_y_to_x(y_gen)

        return y_gen, x_cycle


    # Define MSE logistic loss
    def mse_loss(self, x, target):

        if target == 1:
            return nn.MSELoss()(x, torch.ones(x.size()).to(self.device))
        else:
            return nn.MSELoss()(x, torch.zeros(x.size()).to(self.device))


    # Define Generator Loss
    def loss_gen(self, idt, real, cycle, disc):


        # Identity Losses:
        loss_idt = nn.L1Loss()(idt, real) * self.lmbda * self.idt_coef

        # Cycle Losses:
        loss_cycle = nn.L1Loss()(cycle, real) * self.lmbda

        # Adversarial Losses:
        loss_adv = self.mse_loss(disc, target=1)

        # Total Generator loss:
        loss_gen = loss_cycle + loss_adv + loss_idt

        return loss_gen


    # Discriminator Loss
    # Ideal Discriminator will classify real image as 1 and fake as 0
    def loss_disc(self, real, gen):

        loss_real = self.mse_loss(real, target=1)
        loss_gen = self.mse_loss(gen, target=0)

        return (loss_real + loss_gen)/2


    # Train
    def train(self, img_dl):

        history = []
        dataset_len = img_dl.__len__()
        print_header = True

        for epoch in range(self.epochs):

            # Start measuring time for epoch
            start_time = time.time()


            # Nulify average losses for an epoch
            loss_gen_faces_avg = 0.0
            loss_gen_anime_avg = 0.0
            loss_gen_total_avg = 0.0

            loss_disc_faces_avg = 0.0
            loss_disc_anime_avg = 0.0
            loss_disc_total_avg = 0.0


            # Iterate through dataloader with images
            for i, (faces_real, anime_real) in enumerate(img_dl):

                faces_real, anime_real = faces_real.to(device), anime_real.to(device)


                # Disable gradients for discriminators during generator training
                self.param_require_grad([self.disc_m, self.disc_p], requires_grad=False)

                # Set gradients for generators to zero at the start of the training pass
                self.adam_gen.zero_grad()

                # FORWARD PASS THROUGH GENERATOR

                anime_gen, faces_cycle = self.cycle_gen(faces_real,
                                                        self.gen_f2a,
                                                        self.gen_a2f)

                faces_gen, anime_cycle = self.cycle_gen(anime_real,
                                                        self.gen_a2f,
                                                        self.gen_f2a)

                # Real Anime -> Identical Anime
                anime_idt = self.gen_f2a(anime_real)

                # Real faces -> Identical faces
                faces_idt = self.gen_a2f(faces_real)

                # DISCRIMINATOR PRECTION ON GENERATED IMAGES

                # Discriminator A: Check generated Anime
                anime_disc = self.disc_m(anime_gen)

                # Discriminator F: Check generated faces
                faces_disc = self.disc_p(faces_gen)

                # CALCULATE LOSSES FOR GENERATORS

                # Generator Losses
                loss_gen_faces = self.loss_gen(faces_idt, faces_real, faces_cycle, faces_disc)
                loss_gen_anime = self.loss_gen(anime_idt, anime_real, anime_cycle, anime_disc)


                # Total Generator loss:
                loss_gen_total = loss_gen_faces + loss_gen_anime

                # Update average Generator loss:
                loss_gen_faces_avg += loss_gen_faces.item()
                loss_gen_anime_avg += loss_gen_anime.item()
                loss_gen_total_avg += loss_gen_total.item()

                # GENERATOR BACKWARD PASS


                # Propagate loss backward
                loss_gen_total.backward()

                # Make step with optimizer
                self.adam_gen.step()

                # FORWARD PASS THROUGH DISCRIMINATORS

                # Enable gradients for discriminators during discriminator trainig
                self.param_require_grad([self.disc_m, self.disc_p], requires_grad=True)

                # Set zero gradients
                self.adam_disc.zero_grad()

                # discriminator A: Predictions on real and generated Anime:
                anime_disc_real = self.disc_m(anime_real)
                anime_disc_gen = self.disc_m(anime_gen.detach())

                # discriminator F: Predictions on real and generated faces:
                faces_disc_real = self.disc_p(faces_real)
                faces_disc_gen = self.disc_p(faces_gen.detach())

                # CALCULATE LOSSES FOR DISCRIMINATORS

                # Discriminator losses
                loss_disc_faces = self.loss_disc(faces_disc_real, faces_disc_gen)
                loss_disc_anime = self.loss_disc(anime_disc_real, anime_disc_gen)

                # Total discriminator loss:
                loss_disc_total = loss_disc_faces + loss_disc_anime

                # DISCRIMINATOR BACKWARD PASS

                # Propagate losses backward
                loss_disc_total.backward()

                self.adam_disc.step()

                # Update average Discriminator loss
                loss_disc_faces_avg += loss_disc_faces.item()
                loss_disc_anime_avg += loss_disc_anime.item()
                loss_disc_total_avg += loss_disc_total.item()

            # Calculate average losses per epoch
            loss_gen_faces_avg /= dataset_len
            loss_gen_anime_avg /= dataset_len
            loss_gen_total_avg /= dataset_len

            loss_disc_faces_avg /= dataset_len
            loss_disc_anime_avg /= dataset_len
            loss_disc_total_avg /= dataset_len

            # Estimate training time per epoch
            time_req = time.time() - start_time

            # Expand training history
            history.append([epoch, loss_gen_faces_avg,
                                   loss_gen_anime_avg,
                                   loss_gen_total_avg,
                                   loss_disc_faces_avg,
                                   loss_disc_anime_avg,
                                   loss_disc_total_avg ])

            # Step learning rate scheduler
            self.scheduler_gen.step()
            self.scheduler_disc.step()

            if (epoch + 1) % config.log_iter  == 0:
                clear_output(wait=True)
                text_log = f'''
                Epoch {epoch+1:3}, GenF {loss_gen_faces_avg:>8.2f}', GenA {loss_gen_anime_avg:>8.2f},
                DiskF {loss_disc_faces_avg:>8.2f}, DiskA {loss_disc_anime_avg:>8.2f}, Time sec {time_req:6.0f}
                '''
                print(text_log)
                show_train_images( faces_real.cpu(), anime_gen.cpu(), faces_cycle.cpu(), epoch+1 )


        # Save training history
        history = pd.DataFrame(history, columns=['epoch',
                                                 'loss_gen_faces_avg',
                                                 'loss_gen_anime_avg',
                                                 'loss_gen_total_avg',
                                                 'loss_disc_faces_avg',
                                                 'loss_disc_anime_avg',
                                                 'loss_disc_total_avg' ])

        history.to_csv('history.csv', index=False)
        plot_loss_history(log='history.csv')
        

def main():
    SEED = 42
    RADNOM_STATE = SEED
    
    # Создадим директорию для картинок-результатов.
    if not os.path.exists("result_sample"):
        os.makedirs("result_sample")
        
    # Загружаем данные с Гугл Диска, если данных нет в рабочей директории.

    if not os.path.isfile('./anime.zip'):
        gdown.download('https://drive.google.com/uc?id=1Q12fxT5drgPOw9eP0vSooZLd3rkvK7LJ', 'anime.zip')
    if not os.path.isfile('faces_dataset.zip'):
        gdown.download('https://drive.google.com/uc?id=19yA3vhY-U5bEMsyt4x2T1QAZfeo5IL4I', 'faces_dataset.zip')
        
    if not "anime" in os.listdir():
        # Specify the zip file path
        zip_file_path = "anime.zip"

        # Specify the destination folder for extraction
        extract_folder = "anime/"

        # Specify the minimum file size in bytes
        min_file_size = 13 * 1024  # 13 Kbytes
        num_files_to_extract = config.anime_size
        # Open the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:

            extracted_count = 0
            # Extract only the files larger than the minimum file size
            for file_info in zip_ref.infolist():
                if file_info.file_size > min_file_size:
                    zip_ref.extract(file_info, extract_folder)
                    extracted_count += 1
                    if extracted_count >= num_files_to_extract:
                                        break
        print(f"Total files extracted: {extracted_count}")

    # Распакуем архив с Датасетом.
    if not "faces_dataset" in os.listdir():
        zip_file_path = "faces_dataset.zip"
        extract_folder = "faces_dataset/"
        # Specify the number of files to extract
        num_files_to_extract = 1000

        # Open the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:

            # Extract the first `num_files_to_extract` files from the zip archive
            for i, file_name in enumerate(zip_ref.namelist()):
                if i >= num_files_to_extract:
                    break
                zip_ref.extract(file_name, extract_folder)

        print(f"Successfully extracted {num_files_to_extract} files.")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_dataset = ImageDatasetCV(anime_dir=config.anime_dir, faces_dir=config.faces_dir)
    
    img_dataloader = DataLoader(img_dataset,
                            batch_size=config.batch_size,
                            pin_memory=True)
    
    gan = CycleGAN(config.nc, config.nc, epochs=config.n_epoch, device=device)
    gan.train(img_dataloader)


if __name__ == '__main__':
    main()