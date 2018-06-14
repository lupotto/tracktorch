import numpy as np
import matplotlib.pyplot as plt
import sys
import xml.etree.ElementTree as ET
from lxml import etree

import re
import os
from os.path import isfile
import random
import cv2
import pickle
import pandas as pd
from collections import defaultdict
import datetime

import matplotlib as mpl
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader, Dataset
from shutil import copyfile
from mpl_toolkits.mplot3d import Axes3D

#TODO 1: list to numpy
#TODO 2: Update


class Autel(object):

    def __init__(self, data_location=os.path.join(os.path.expanduser('~')),
                     read_all_data=True, batch_size=1):
        """
        Constructor of Autel helper class for reading images and annotations
        :param data_location (str): relative path to the root folder of the data.
        :param read_all_data (boolean): if is 1st reading data or not
        :param batch_size:
        :return:
        """
        self.read_all_data = read_all_data
        self.classes_times = dict()
        self.ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
        self.batch_size = batch_size
        self.data_location = data_location
        self.img_dict = dict()
        self.img_wrong = defaultdict(list)
        self.label_wrong = defaultdict(list)

    def load_dataset(self):
        if self.read_all_data and not os.path.isfile(os.path.join(self.ROOT_DIR,'auteltools','annotation_file.pkl')):
            list_annotation = self.load_data()
            self.create_ann_file(list_annotation)
            self.generate_csv_file()
        else:
            list_annotation = self.load_pkl_file()


        self.annotations = list_annotation



    def load_data(self):
        """
        Pre: read_all_data = True
        Create and load images from the dataset in a Annotation structure.
        :return: list of Annotation
        """

        main_path = self.check_path_dataset()
        print("Loading dataset...")
        #Parsing all the images
        list_annotation = []
        for root, dirs, files in os.walk(main_path, topdown=False):

            files.sort()
            for name_file in files:
                if name_file.endswith('.jpg'):
                    self.img_dict[name_file] = os.path.join(root, name_file)
                elif name_file.endswith('.xml'):
                    annotation = self.load_annotation(root, name_file)
                    if annotation:
                        list_annotation.append(annotation)

        print("Dataset loaded with {} images".format(len(list_annotation)))


        return list_annotation


    def check_path_dataset(self):
        """
        Check the path, if exists, returns main_path, otherwise print error and finish.
        :return: main_path (str): path of the dataset
        """
        if self.data_location == os.path.expanduser('~'):
            main_path = os.path.join(os.path.join(os.path.expanduser('~'),'AutelData'))
        else:
            main_path = os.path.join(os.path.join(os.path.expanduser('~'),self.data_location,'AutelData'))
        if not os.path.isdir(main_path):
            print("Dataset path not found! Write the path relative to '~' in Autel() class")
            sys.exit()

        return main_path
    def load_annotation(self, root, name_xml):
        """
        Save all the fields of Annotation structure, also check the sizes and format of the images.
        Finally, creates a csv with the images & paths that are incorrect
        :param root (str): path for parse the .xml
        :param name_xml (str): id of the xml
        :return:
        """
        tree = ET.parse(os.path.join(root, name_xml))
        root = tree.getroot()
        name_jpg = root.find('filename').text
        annotation = False

        if name_jpg in self.img_dict:

            path_file = self.img_dict[name_jpg]
            width = root.find('size').find('width').text
            height = root.find('size').find('height').text
            depth = root.find('size').find('depth').text

            labels = self.parse_labels(root)

            if self.check_sizes(int(width), int(height), int(depth)) and self.check_name_label(labels):
                annotation = Annotation(name_jpg, path_file, width, height, depth, labels)
            else:
                self.generate_dict_wrong_image(name_jpg, path_file,
                                                int(width), int(height), int(depth), root)

        return annotation

    def check_name_label(self,labels):
        """

        :param labels:
        :return:
        """
        classes = self.load_classes()
        for label in labels:
            if label.label_name in classes:
                return True
            else:
                return False


    def create_ann_file(self,list_annotation):
        """
        Generate the Pickle for future loads of images.(created in same auteltools
            folder with name annotation_file.pkl)
        :param list_annotation (Annotation): All the Annotation structures
        :return:
        """
        print("Creating Pickle file...")
        with open(os.path.join(self.ROOT_DIR,'auteltools',"annotation_file.pkl"), "wb") as fp:  # Pickling
            pickle.dump(list_annotation, fp)

        print("Pickle created in {}".format(os.path.join(self.ROOT_DIR,'auteltools','annotation_file.pkl')))


    def generate_dict_wrong_image(self, name_jpg, path_file, width, height, depth, root):
        """
        Generate the dictionary, saving all the parameters that will be in the wrong_images_csv
        :return:
        """
        self.img_wrong['image_name'].append(name_jpg)
        self.img_wrong['path'].append(path_file)
        self.img_wrong['shape'].append("({},{},{})".format(int(width), int(height), int(depth)))
        labels = self.parse_labels(root)
        names = [labels[i].get_name() for i in range(len(labels))]
        labels = "/".join(map(str, names))
        self.img_wrong['labels'].append(labels)

    def parse_labels(self, root):
        """
        Function that parse all the labels of the xml file
        :param root:
        :return:
        """
        objects = root.findall('object')
        array_labels = []
        for i in range(len(objects)):
            name = objects[i].find('name').text
            xmin = int(objects[i].find('bndbox').find('xmin').text)
            ymin = int(objects[i].find('bndbox').find('ymin').text)
            xmax = int(objects[i].find('bndbox').find('xmax').text)
            ymax = int(objects[i].find('bndbox').find('ymax').text)
            label = Label(name,xmin,ymin,xmax,ymax)
            array_labels.append(label)

        return array_labels

    def check_sizes(self, width, height, depth):
        """
        Function for verify the size of the images
        :param width (int):
        :param height (int):
        :param depth (int):
        :return: boolean
        """
        if width == 1280 and height == 720 and depth == 3:
            return True
        return False

    def generate_csv_file(self):
        """
        Csv with images that do not meet the requirements
        :return:
        """
        if bool(self.img_wrong):
            print("Creating csv with wrong images...")
            df = pd.DataFrame(self.img_wrong,
                              columns=['image_name', 'path', 'shape', 'labels'])
            df.to_csv(os.path.join(self.ROOT_DIR,'auteltools','images_wrong.csv'))
            print("CSV created in {}".format(os.path.join(self.ROOT_DIR,'auteltools','images_wrong.csv')))

    def load_pkl_file(self):
        """
        Load pkl of the Annotation structures
        :return:
        """
        print("Loading Pickle file...")
        with open(os.path.join(self.ROOT_DIR,"auteltools/annotation_file.pkl"), "rb") as fp:
            print(os.path.join(self.ROOT_DIR,"auteltools/annotation_file.pkl"))
            annotations = pickle.load(fp)
        print("Pickle loaded")
        return annotations

    def load_images(self, batch_size = 1, rndm = True):
        """
        Return a batch of images, random or at the beginning
        :param batch_size:
        :return:
        """
        image_list = []
        if rndm:
            list_rndm = random.sample(range(1, len(self.annotations)), batch_size)
            for num in list_rndm:
                image_list.append(self.annotations[num].path_file)
        #image_list = [bch.path_file for i, bch in enumerate(self.annotations) if i < batch_size]

        return image_list


    def split_train_test(self, test_size=0.1):
        """
        Create a train.txt, test.txt and total.txt in the AutelData folder.
        :param test_size (float): Percentage of test
        :return:
        """
        main_path = self.check_path_dataset()

        train_file = open(os.path.join(main_path, 'train.txt'), 'w')
        test_file = open(os.path.join(main_path, 'test.txt'), 'w')
        total_file = open(os.path.join(main_path, 'total.txt'),'w')

        num_train = int((1-test_size) * self.__len__())

        for i, ann in enumerate(self.annotations):
            if i < num_train:
                train_file.write('{}\n'.format(ann.path_file))
            else:
                test_file.write('{}\n'.format(ann.path_file))

            total_file.write('{}\n'.format(ann.path_file))

        print("Train: {} images".format(num_train))
        print("Test: {} images".format(int(self.__len__()*test_size)))
        train_file.close()
        test_file.close()
        total_file.close()

    def create_labels_yolo(self):
        """
        Generate the .txt file with yolo format (id_label, x, y, h, w) and save it to the folder of the
        image .jpg and .xml of AutelData
        :param ann:
        :return:
        """
        classes = self.load_classes()


        #create labels

        for i, ann in enumerate(self.annotations):
            out_file = open(os.path.join(os.path.dirname(ann.path_file), '{}.txt'.format(ann.file_name.split('.')[0])),
                            'w')
            print(os.path.join(os.path.dirname(ann.path_file), '{}.txt'.format(ann.file_name.split('.')[0])))
            for label in ann.labels:
                if label.label_name not in classes:
                    continue
                idx_class = classes.index(label.label_name)
                b = (float(label.xmin), float(label.xmax), float(label.ymin), float(label.ymax))
                bb = self.convert_yolo_sizes((float(ann.width),float(ann.height)),b)
                b_draw = (int(label.xmin), int(label.xmax), int(label.ymin), int(label.ymax))
               # self.print_bboxes(b_draw,bb,label.label_name,ann)
                out_file.write(str(idx_class) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()

    def create_labels_gt(self, dest, paths = None):
        """
        Generate the .txt file with voc format (id_label, x, y, h, w) and save it to the folder specified in dest
        :param ann:
        :return:
        :param dest:
        :return:
        """
        classes = self.load_classes()
        files_out = []

        for i, ann in enumerate(self.annotations):
            if ann.path_file in paths:
                out_file = open(os.path.join(dest, '{}.txt'.format(ann.file_name.split('.')[0])), 'w')
                files_out.append(ann.path_file)
                for label in ann.labels:
                    if label.label_name not in classes:
                        continue
                    #idx_classes = classes.index(label.label_name)
                    b = (int(label.xmin), int(label.ymin), int(label.xmax), int(label.ymax))
                    out_file.write(label.label_name + " " + " ".join([str(a) for a in b]) + '\n')
                out_file.close()

        return files_out



    def create_gt_voc(self, dest, img_path, labels,aux):

        classes = self.load_classes()
        out_file = open(os.path.join(dest, '{}.txt'.format(img_path.split('.')[0])), 'w')

        for label in labels:
            if label.label_name not in classes:
                continue
            idx_classes = classes.index(label.label_name)
            b = (int(label.xmin), int(label.ymin), int(label.xmax), int(label.ymax))
            out_file.write(str(idx_classes) + " " + " ".join([str(a) for a in b]) + '\n')



    def create_predict_detect(self, dest, img_path, labels_detect):

        classes = self.load_classes()
        out_file = open(os.path.join(dest, '{}.txt'.format(img_path.split('.')[0])), 'w')

        for lbl in labels_detect:
            idx_classes = classes.index(lbl[0])
            b = (float(lbl[1]),int(lbl[2]), int(lbl[3]), int(lbl[4]), int(lbl[5]))
            out_file.write(str(idx_classes) + " " + " ".join([str(a) for a in b]) + '\n')


    def convert_yolo_sizes(self,size,box):
        """
        Conversion from xml to yolo format
        :param size:
        :param box:
        :return:
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def load_classes(self):
        """
        read the classes from the autel.names file
        :return:
        """
        name = os.path.join(self.ROOT_DIR,"data", "autel.names")
        fp = open(name, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def print_bboxes(self,b,name,path_file):
        """
        function for check the bboxes are well labeled
        :param b: (x1,x2,y1,y2)
        :param bb:
        :param name:
        :param ann:
        :return:
        """

        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.75
        line_type = 1
        #b[0] xmin b[1] xmax
        #b[2] ymin b[3] ymax
        img = cv2.imread(path_file)
        font_color = self.set_color(name)
        bbox = cv2.rectangle(img,(b[0],b[2]),(b[1],b[3]),font_color,2)
        newimg = cv2.putText(bbox, name, (b[0] - 5, b[2] - 5),font,
                                        font_scale, font_color, line_type)
        k = cv2.waitKey(1)

        if k == 27:  # If escape was pressed exit
            cv2.destroyAllWindows()
            sys.exit()

        cv2.imshow('Image', newimg)

    def set_color(self,name):
        """
        Colors of the bboxes
        :param name:
        :return: color depending on the label class
        """
        font_color = (0, 0, 0)
        if name == 'Car':
            font_color = (255, 0, 0)
        elif name == 'Person':
            font_color = (0, 0, 255)
        elif name == 'Vehicle':
            font_color = (255, 255, 255)
        elif name == 'Rider':
            font_color = (204, 204, 0)
        elif name == 'Animal':
            font_color = (255, 140, 0)
        elif name == 'Boat':
            font_color = (0, 140, 255)

        else:
            print("class not found")

        return font_color

    def count_classes(self):
        """
        creates a dictionary counting the samples per class
        :return: 
        """
        for ann in self.annotations:
            for label in ann.labels:
                if label.label_name in self.classes_times:
                    self.classes_times[label.label_name] += 1
                else:
                    self.classes_times[label.label_name] = 1

        total_sum = 0
        suma = sum(self.classes_times.values())
        for i in self.classes_times:
            num = self.classes_times[i]
            total_sum += num/suma
        print(self.classes_times)
        #print(len(self.annotations))
        print(total_sum)
        return self.classes_times

    def show_class(self,name_class,batch_size=1):
        """
        Prints and returns the paths of your class, specifing the batch of images that you want
        :param name_class:
        :return:
        """
        classes = self.load_classes()
       # if not name_class in classes:
        #    print("Class not found in dataset file")
         #   sys.exit()
        print('Classes')
        ann_batch = []
        idx = 0
        for ann in self.annotations:
            for label in ann.labels:
                if (name_class == label.label_name) and idx < batch_size:
                    ann_batch.append(ann.path_file)
                    print(ann.path_file)
                    idx += 1

        return ann_batch

    def split_train_test_classes(self, test_size=0.1):
        """
        Splits the dataset in train/test, applying a percentage for every class.
        :return:
        """
        main_path = self.check_path_dataset()

        train_file = open(os.path.join(main_path, 'train_cls.txt'), 'w')
        test_file = open(os.path.join(main_path, 'test_cls.txt'), 'w')

        dict_classes = self.count_classes()
        dict_test = dict()
        dict_train = dict()
        dict_compute = dict()
        suma = 0

        for key, value in dict_classes.items():
            dict_compute[key] = 0
            dict_test[key] = int(value * test_size)
            dict_train[key] = value - dict_test[key]

        #TODO create frequency and split of classes

        for ann in self.annotations:
            dict_compute = self.compute_labels(ann.labels)
            #dict_train= {key:dict_train[key] - dict_compute.get(key,0) for key in dict_train.keys()}

            for key in dict_train.keys():
                print(key)
                #dict_train = dict_train[key] - dict_compute.get(key,0)
                #print(dict_train)
                #if dict_train[key] < 0:
                 #   print(dict_train)

            '''
            for lbl in ann.labels:
                if lbl.label_name in dict_train:
                    if dict_train[lbl.label_name] > 0:
                        dict_train[lbl.label_name] -= 1

                    else:

                        flag = 1

                if flag:

                    flag = 0
                   # print(dict_train)
                #print(lbl.label_name)
            '''

    def create_sequences(self):

        sequences = defaultdict(list)

        for ann in self.annotations:
            seq_path = os.path.dirname(ann.path_file)
            sequences[seq_path].append(ann)

        return sequences

    def count_max_imgs(self, sequences):
        prev_dim =0
        for i, (key,images) in enumerate(sequences.items()):
            auxiliar = 0
            for img in images:
                auxiliar += img.classes_to_vector()
            max_dim = np.max(auxiliar)
            if max_dim > prev_dim:
                prev_dim = max_dim

        return prev_dim


    def histogram_3d(self):
        """
        Creates and shows the histogram with classes of autel.names, sequences of dataset
        and num classes per sequence
        :return:
        """

        # Matrix 3D
        classes = self.load_classes()
        sequences = self.create_sequences()
        max_dim = self.count_max_imgs(sequences)
        new_matrix = np.zeros((len(sequences), len(classes), int(max_dim)))

        for i, (key, images) in enumerate(sequences.items()):
            for idx, img in enumerate(images):
                new_matrix[i,:,idx] = img.classes_to_vector()


        z, x, y = new_matrix.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, edgecolor='none', c='r', marker='o', s=35, label='rndm')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Frequency per class')
        ax.set_zlabel('Sequence')

        plt.show()


    def histogram_2d(self):
        """
        Creates and shows the histogram with classes of autel.names and num classes per sequence
        :return:
        """

        #Matrix 2D
        sequences = self.create_sequences()
        classes = self.load_classes()
        print(classes)
        matrix_seq_cls = np.zeros((len(sequences), len(classes)))

        prev_dim = 0
        for i, (key, images) in enumerate(sequences.items()):
            auxiliar = 0
            for img in images:
                auxiliar += img.classes_to_vector()
            matrix_seq_cls[i] += auxiliar
            max_dim = np.max(auxiliar)
            if max_dim > prev_dim:
                prev_dim = max_dim

        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_xlabel('Sequences')
        ax.set_ylabel('Classes')

        plt.yticks(np.arange(0,len(classes),1),classes)

        plt.imshow(matrix_seq_cls.T,aspect='auto',cmap='hot')
        plt.colorbar()
        plt.tight_layout()

        plt.show()

        return matrix_seq_cls

    def check_labels_folder(self, folder_path):

        dict_labels = dict()
        label_name = 'Unknown'
        for root, dirs, files in os.walk(folder_path, topdown=False):
            files.sort()

            for name_file in files:

                path = os.path.join(root,name_file)
                tree = ET.parse(path)
                new_root = tree.getroot()
                labels = self.parse_labels(new_root)
                self.compute_labels(labels,dict_labels)
                flag = [True for x in labels if x.label_name == "Unknown"]
                if len(flag) > 0:
                    self.generate_dict_wrong_label(labels,path)

                #jpg_id = path.split('/')[-1].replace('.xml','.jpg')
                #self.show_label_image(jpg_id, labels, label_name)
        self.generate_csv_labels()
        print(dict_labels)

    def show_label_image(self,jpg_id,labels,name_wrong):

        for ann in self.annotations:
            if ann.file_name == jpg_id:
                for lbl in labels:
                    if lbl.label_name == name_wrong:
                        coords = (lbl.xmin,lbl.xmax,lbl.ymin,lbl.ymax)
                        self.print_bboxes(coords,lbl.label_name,ann.path_file)

    def generate_dict_wrong_label(self, labels, path):

        self.label_wrong['image_name'].append(path.split('/')[-1].replace('xml','jpg'))
        self.label_wrong['path'].append(os.path.relpath(path,os.path.join(os.path.expanduser('~'),'resources', 'extra_autel')))
        names = [labels[i].get_name() for i in range(len(labels))]
        labels = "/".join(map(str, names))
        self.label_wrong['labels'].append(labels)


    def compute_labels(self,labels,dict_labels):

        for lbl in labels:
            if lbl.label_name in dict_labels:
                dict_labels[lbl.label_name] += 1
            else:
                dict_labels[lbl.label_name] = 1

    def generate_dict_wrong_label(self, labels, path):

        self.label_wrong['xml_name'].append(path.split('/')[-1])
        self.label_wrong['path'].append(os.path.relpath(path, os.path.join(os.path.expanduser('~'),'resources', 'extra_autel')))
        names = [labels[i].get_name() for i in range(len(labels))]
        labels = "/".join(map(str, names))
        self.label_wrong['labels'].append(labels)

    def generate_csv_labels(self):
        """
        Csv with images that do not meet the requirements
        :return:
        """
        if bool(self.label_wrong):
            print("Creating csv with wrong labels...")
            df = pd.DataFrame(self.label_wrong,
                              columns=['xml_name', 'path', 'labels'])
            df.to_csv(os.path.join(self.ROOT_DIR,'auteltools','labels_wrong.csv'))
            print("CSV created in {}".format(os.path.join(self.ROOT_DIR,'auteltools','labels_wrong.csv')))




class Annotation(Autel):
    def __init__(self, file_name, path_file, width, height, depth, labels):
        """
         Constructor of the XML file for read all the parameters related with the image (Folder, Image associated (.jpg),
         dimensions and classes inside)
         :param file_name (str): id of the jpg image
         :param path_file (str): path to jpg image
         :param (width,height,depth) (int,int,int): shape of the image
         :param labels (array of Label): array of labels of the image
         :return:
         """
        Autel.__init__(self,data_location=os.path.join(os.path.expanduser('~')),
                     read_all_data=True, batch_size=1)
        self.file_name = file_name
        self.path_file = path_file
        self.width = width
        self.height = height
        self.depth = depth
        self.labels = labels

    def show_annotation(self):
        """
        Print all the information of the Image and Labels inside the image.
        :return:
        """
        print(self.file_name)
        print(self.path_file)
        print(self.width)
        print(self.height)
        print(self.depth)
        [self.labels[i].show_values() for i in range(len(self.labels))]

    def get_labels_name(self):
        """
        Get the labels name of a given annotation
        :return:
        """
        return [(self.labels[i].get_name()) for i in range(len(self.labels))]

    def classes_to_vector(self):
        """
        Creates a numpy vector of the number of classes indexed with autel.names file
        :return:
        """

        classes = self.load_classes()
        vector_cls = np.zeros(len(classes))

        for label in self.labels:
            idx_class = classes.index(label.label_name)
            vector_cls[idx_class] += int(1)

        return vector_cls

class Label(object):
    def __init__(self, label_name, xmin, ymin, xmax, ymax):
        """
        Constructor of a label of the image.
        :param label_name (str): name of the label
        :param xmin (int): x min bounding box
        :param ymin (int): y min bounding box
        :param xmax (int): x max bounding box
        :param ymax (int): y max bounding box
        :return:
        """
        self.label_name = label_name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def get_name(self):

        return self.label_name

    def show_values(self):
        """
        Show coordinates bbox.
        :param label_name (str): name of the label
        :param (x1,y1) (int): coordinates top left bbox
        :param (x2,y2) (int): coordinates down right bbox
        :return:
        """

        print("Label name: {}".format(self.label_name))
        print("(x1,y1): ({},{})".format(self.xmin, self.ymin))
        print("(x2,y2): ({},{})".format(self.xmax, self.ymax))

#if __name__ == '__main__':

    #dataset.histogram_2d()
    #dataset.histogram_3d()
    #dataset.histogram_sequences()
    #vector_classes = dataset.annotations[7890].get_classes_vector()

    #dataset.create_sequences()
    #dataset.histogram_sequences()
    #dataset.split_train_test_classes()

   # classes_times =dataset.count_classes()

    #dataset.split_train_test_classes()
    #dataset.split_train_test(0.1)
    #print(dataset.__len__())
    #dataset.create_labels_yolo()
    #dataset.show_class('Person',16)
    #dataset.annotations[0].get_classes_vector()

    '''
    font_color = (0,0,0)

    path = '/home/alupotto/projects/PyTorch-YOLOv3/data/coco/images/train2014/COCO_train2014_000000426052.jpg'
    pth2 = '/home/alupotto/resources/AutelData/ObjectRecognitionTrainTest/MB0009/MB0009_000314.jpg'
    label_path = '/home/alupotto/projects/PyTorch-YOLOv3/data/coco/labels/train2014/COCO_train2014_000000426052.txt'
    label_pth2 = '/home/alupotto/resources/AutelData/ObjectRecognitionTrainTest/MB0009/MB0009_000314.txt'
    while True:
        img = cv2.imread(pth2)
        h,w,_ = img.shape
        labels = np.loadtxt(label_pth2).reshape(-1, 5)
        print(labels[1])
        x1 = w * (labels[1, 1] - labels[1, 3] / 2)
        y1 = h * (labels[1, 2] - labels[1, 4] / 2)
        x2 = w * (labels[1, 1] + labels[1, 3] / 2)
        y2 = h * (labels[1, 2] + labels[1, 4] / 2)

        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),font_color,2)
        k = cv2.waitKey(1)


        if k == 27:  # If escape was pressed exit
            cv2.destroyAllWindows()
            sys.exit()
        cv2.imshow('img',img)

    '''