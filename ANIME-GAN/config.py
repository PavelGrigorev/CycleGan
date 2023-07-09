import argparse

parser = argparse.ArgumentParser("DCGAN")
parser.add_argument('--anime_dir', type=str, default="anime/images") # anime dataset directory
parser.add_argument('--faces_dir', type=str, default="faces_dataset/faces_dataset/faces_dataset_small") # faces dataset directory
parser.add_argument('--result_dir', type=str, default="./result_sample") # log image directory
parser.add_argument('--batch_size', type=int, default=1) # batch size
parser.add_argument('--n_epoch', type=int, default=2) # epoch size
parser.add_argument('--log_iter', type=int, default=1) # print log message and save image per log_iter
parser.add_argument('--anime_size', type=int, default=300)  # noise dimension
parser.add_argument('--nc', type=int, default=3)    # input and out channel
parser.add_argument('--lr', type=float, default=0.01) # learning rate
parser.add_argument('--beta', type=float, default=0.5)
config, _ = parser.parse_known_args()