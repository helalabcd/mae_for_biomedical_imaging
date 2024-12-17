import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import math
#from helpers import make_grid
import torchvision.utils as vutils

class FixedTransform():
    def __init__(self, min_angle, max_angle, crop_height, crop_width, target_size):
        # Generate fixed parameters for all transformations
        self.angle = torch.FloatTensor(1).uniform_(min_angle, max_angle).item()
        self.position = None
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.input_size = target_size

    def __call__(self, img):
        # If position hasn't been set, choose a random position
        if self.position is None:
            w, h = img.size
            th, tw = self.crop_height, self.crop_width
            if w == tw and h == th:
                self.position = (0, 0)
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                self.position = (i, j)

        # Apply the same transformation to the image
        transformed_img = transforms.functional.rotate(img, self.angle)
        transformed_img = transforms.functional.crop(transformed_img, *self.position, self.crop_height, self.crop_width)

        # Resize the image to 224x224
        if (self.input_size != self.crop_width) or (self.input_size != self.crop_height):
            transformed_img = transforms.functional.resize(transformed_img, (input_size, input_size))
        transformed_img = transforms.ToTensor()(transformed_img)

        return transformed_img

class BioData:

    
    """
    data_sample: Pass a percentage as a float to use a sample that's smaller
    or larger than the train size!
    Images are cropped randomly so this should not be a huge issue.
    
    Example:
    data_sample=2.5 => Use 2.5 times the train examples that are actually present
    data_sample=0.3 => Only use 30% of the train examples that are actually present
    
    This is useful if you need your epochs to be a certain size, so the hyperparameters
    match up and result in the same number of steps over the loss surface
    """
    def __init__(self, args, base_dir="train", sequence_length=1, data_sample=None):
        self.sequence_length = sequence_length
        self.sequences = []
        self.sequence_start_indices = []
        self.current_start_index = 0
        self.base_dir = base_dir
        self.data_sample = data_sample
        self.args = args

        for burst in os.listdir(base_dir):
            burst_path = os.path.join(base_dir, burst, "img1")
            frames = sorted(os.listdir(burst_path))
            frames = [os.path.join(base_dir, burst, "img1", x) for x in frames if x.endswith(('.tiff', '.tif'))]
            self.sequences.append(frames)
            self.sequence_start_indices.append(self.current_start_index)
            self.current_start_index += len(frames) - self.sequence_length
    def __getitem__(self, idx):

        if idx > self._real_length():
            return self[random.randint(0, self._real_length())]

        start_item = max([x for x in self.sequence_start_indices if x <= idx])
        start_index = self.sequence_start_indices.index(start_item)
        
        local_index = idx - start_item
        files = self.sequences[start_index][local_index:local_index+self.sequence_length]
        
        # Do a random crop of the frames.
        # The crop COULD end up out of bounds due to rotating & cropping. We do a check
        # for that and choose a new random crop if that happens.
        out_of_bounds = True
        while out_of_bounds:
            fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=self.args.crop_size, crop_width=self.args.crop_size, target_size=self.args.input_size)

            tensors = []
            for file_path in files:
                with Image.open(file_path) as img:  # This will ensure the file is closed after the block
                    tensor = fixed_transformation(img)
                    tensors.append(tensor)

            ex = tensors[0]
            zero = torch.Tensor([0,0,0])
            out_of_bounds = torch.allclose(ex[:, 0, 0], zero) or \
                torch.allclose(ex[:, 0, -1], zero) or \
                torch.allclose(ex[:, -1, 0], zero) or \
                torch.allclose(ex[:, -1, -1], zero)

        stacked = torch.stack(tensors) # torch.Size([n_frames, 1, 128, 128])
        
        # Convert to a large image
        # MAE patchify is row-by-row so having one column should
        # work naturally for enlargening them
        grid = vutils.make_grid(stacked, nrow=int(math.sqrt(self.sequence_length)), padding=0)
        # Return something for y, so we can later easilly pass additional
        # data if the need arises
        dummy_y =  torch.Tensor([0])
        return grid, dummy_y
        
    def _real_length(self):
        return self.current_start_index - 1

    def __len__(self):
        if self.data_sample is not None:
            return int((self.current_start_index - 1) * self.data_sample)

        return self._real_length()
