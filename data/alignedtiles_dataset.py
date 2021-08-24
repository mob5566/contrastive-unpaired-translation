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
import os
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image

from robosat.tiles import tiles_from_slippy_map

class AlignedTilesDataset(BaseDataset):
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
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
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
        if opt.phase == 'train':
            opt.phase = 'training'
        else:
            opt.phase = 'validation'

        self.dir_A = os.path.join(opt.dataroot, 'A', opt.phase, 'images')
        self.dir_B = os.path.join(opt.dataroot, 'B', opt.phase, 'images')

        self.A_tiles = [(tile, path)
                        for tile, path in tiles_from_slippy_map(self.dir_A)]
        self.tile_set = {tile[0] for tile in self.A_tiles}
        self.B_tiles = [(tile, path)
                        for tile, path in tiles_from_slippy_map(self.dir_B)]
        self.tile_set &= {tile[0] for tile in self.B_tiles}
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.A_tiles = sorted(list(filter(lambda tile: tile[0] in self.tile_set,
                                          # self.A_tiles)),
                              # key=lambda tile: tile[0])
        # self.B_tiles = sorted(list(filter(lambda tile: tile[0] in self.tile_set,
                                          # self.B_tiles)),
                              # key=lambda tile: tile[0])
        for tile, path in self.A_tiles:
            if tile not in self.tile_set:
                self.B_tiles.append((tile, self.B_tiles[0][1]))
        self.A_tiles.sort(key=lambda tile: tile[0])
        self.B_tiles.sort(key=lambda tile: tile[0])
        self.tile_set = list(self.tile_set)

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
        A_tile, A_path = self.A_tiles[index]
        B_tile, B_path = self.B_tiles[index]
        tmpA = Image.open(A_path)
        A = Image.new('RGB', tmpA.size, (255, 255, 255))
        A.paste(tmpA, mask=tmpA.split()[3])
        A = self.transform(A)    # needs to be a tensor
        B = self.transform(Image.open(B_path).convert('RGB'))    # needs to be a tensor
        A_path = '_'.join(A_path.split('/')[-3:])
        B_path = '_'.join(B_path.split('/')[-3:])
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        # return len(self.tile_set)
        return len(self.A_tiles)
