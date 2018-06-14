import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from skimage.transform import resize

import sys

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.

        '''
        print("shape image {}".format(h,w))
        print("dim_diff {}".format(dim_diff))
        print("Pads {}".format(pad1,pad2))
        print("pad {}".format(pad))
        print("input_img shape {}".format(input_img.shape))
        '''
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')

        '''
        #Original
        fig = plt.figure()
        ax = plt.subplot(1,2,1)
        ax.set_title("Original")
        ax.imshow(img)
        #Input network
        ax = plt.subplot(1,2,2)
        ax.imshow(input_img)
        ax.set_title("Input network")
        '''

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()



        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        '''
        img = cv2.imread(img_path)
        k = cv2.waitKey(1)

        if k == 27:  # If escape was pressed exit
            cv2.destroyAllWindows()
            sys.exit()
        cv2.imshow('Image',img)
        '''
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)

            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

    def print_bboxes(self,b,path_file):
        """
        function for check the bboxes are well labeled
        :param b: (x1,x2,y1,y2)
        :param bb:
        :param name:
        :param ann:
        :return:
        """
        font_color = (0, 0, 0)

        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.75
        line_type = 1
        #b[0] xmin b[1] xmax
        #b[2] ymin b[3] ymax
        img = cv2.imread(path_file)
        bbox = cv2.rectangle(img,(b[0],b[2]),(b[1],b[3]),font_color,2)
        newimg = cv2.putText(bbox, 'Random', (b[0] - 5, b[2] - 5),font,
                                        font_scale, font_color, line_type)
        k = cv2.waitKey(1)

        if k == 27:  # If escape was pressed exit
            cv2.destroyAllWindows()
            sys.exit()

        cv2.imshow('Image', newimg)