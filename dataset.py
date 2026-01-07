import glob
import os
import zipfile
import numpy as np
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset as TorchDataset

def get_mnist_datasets(data_path='./data'):
    # pre processing pipeline to get pytorch tensor ready for calculus and applies statistics normalization to increase training efficiency
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # automatically imports MNIST dataset and applies the transform defined above
    train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    return train_set, test_set

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):
        
        # store arguments in class properties
        self.args = args
        self.is_train = is_train

        # -----------------
        # create inputs
        # -----------------

        # create image path variable
        print(args['dataset_folder'])
        split_name = 'train' if is_train else 'test'
        image_path = os.path.join(args['dataset_folder'], split_name, 'images')

        #print("DEBUG listing dir:", os.listdir(image_path))


        print('image path:' + image_path)

        self.image_filenames = glob.glob(os.path.join(image_path, '*.jpg'))
        self.image_filenames.sort()

        print("Nº de imagens carregadas:", len(self.image_filenames))

        # -----------------
        # create the labels
        # -----------------

        self.labels_filenames = os.path.join(args['dataset_folder'], split_name, 'labels.txt')
        self.labels = []

        with open(self.labels_filenames, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = float(parts[1])
                self.labels.append(label)

        # Select the percentage of examples specified in args
        num_examples = round(len(self.image_filenames) * args['percentage_examples'])

        # Reduce the size of the image_fileanames and labels
        self.image_filenames = self.image_filenames[0:num_examples]
        self.labels = self.labels[0:num_examples]

        # to convert from list to tensor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        # this returns the number of samples in the dataset
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # this function will get as input an index and return the corresponding sample
        # the return of the function is a tuple (input, label), but these values must be tensors
        # basically, returuns (image_tensor, label_tensor)

        # -----------------
        # get label as tensor
        # -----------------

        label_index = int(self.labels[idx])
        label = [0]*10 # create a list of 10 zeros
        label[label_index] = 1 # set the position of the label to 1

        label_tensor = torch.tensor(label, dtype=torch.float32)

        # -----------------
        # get the image as tensor
        # -----------------

        image_filename = self.image_filenames[idx]

        image = Image.open(image_filename).convert('L') # open image and convert to grayscale
        image_tensor = self.to_tensor(image) # convert image to tensor

        return image_tensor, label_tensor