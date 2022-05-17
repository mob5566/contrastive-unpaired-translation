"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from pathlib import Path
from random import randint
from PIL import Image


class One2oneDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--align_data', action='store_true', help='load data with alignment')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.image_paths_A = sorted(make_dataset(Path(self.root) /
                                                 (opt.phase + 'A')))
        self.image_paths_B = sorted(make_dataset(Path(self.root) /
                                                 (opt.phase + 'B')))

        if self.opt.direction == 'AtoB':
            self.src_image_paths = self.image_paths_A
            self.tgt_image_paths = self.image_paths_B
        else:
            self.src_image_paths = self.image_paths_B
            self.tgt_image_paths = self.image_paths_A

        self.src_size = min(len(self.src_image_paths),
                            self.opt.max_dataset_size)
        self.tgt_size = len(self.tgt_image_paths)
        self.dataset_size = (self.src_size
                             if not self.opt.align_data
                             else min(self.src_size, self.tgt_size))
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        tgt_index = (randint(0, self.tgt_size - 1)
                     if not self.opt.align_data
                     else index)
        path_A = self.src_image_paths[index]
        path_B = self.tgt_image_paths[tgt_index]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        data_A = self.transform(img_A)
        data_B = self.transform(img_B)
        return {'A': data_A, 'B': data_B,
                'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        """Return the total number of images."""
        return self.dataset_size
