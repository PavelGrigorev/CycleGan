# Deep Convolutional Generative Adversarial Networks using pytorch
## CycleGan from image to Anime face.
    
## Requirements
- Python 3
- pytorch ('2.0.0')
- torchvision

## Usage
- python trainer.py --dataset_dir='dataset dir' --result_dir='result dir' \[--options...\]
    - example: python trainer.py --result_dir=./result_sample
- Datasets can be downloaded from links on Google Drive.
	- anime 'https://drive.google.com/uc?id=1Q12fxT5drgPOw9eP0vSooZLd3rkvK7LJ' - zip file
	- faces 'https://drive.google.com/uc?id=19yA3vhY-U5bEMsyt4x2T1QAZfeo5IL4I' - zip file
  
- !python get_anime.py 
	- input image have to be in 'input' fold, output image could be found in 'output' fold
