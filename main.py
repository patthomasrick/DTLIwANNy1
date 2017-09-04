#!/usr/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class
"""
Takes an image, converts it to edges, then analyses it in a neural network
to determine the species of the leaf
"""
__author__ = "Patrick Thomas"
__version__ = "1.0"
__date__ = "12/7/15"

import csv
#from jinja2.nodes import Pair
#from libfann import training_data
from swvgsleaf import LeafCollection, create_fann_train_from_csv, Leaf, load_leaf
from neural_net_tools import NeuralNetContainer
from os import getcwd, remove
from os.path import join, exists
#from generate_extraneous_testing_data import generate_run_testing_data
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkMessageBox import showinfo
from time import asctime, localtime
#from skimage.measure import find_contours
#from skimage.feature import canny
#from scipy.ndimage.interpolation import rotate
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.misc import imsave
CWD = getcwd()
BASE_IMAGE_PATH = join(CWD, 'leaf-images')
IMG_PATHS = {'Q. accutissima':   join(BASE_IMAGE_PATH, 'Quercus accutissima'),
            # 'Q. imbricaria':    join(BASE_IMAGE_PATH, 'Quercus imbricaria'),
            # 'Q. laurifolia':    join(BASE_IMAGE_PATH, 'Quercus laurifolia'),
            # 'Q. lyrata':        join(BASE_IMAGE_PATH, 'Quercus lyrata'),
             'Q. nigra':         join(BASE_IMAGE_PATH, 'Quercus nigra'),
            #'Q. palustris':     join(BASE_IMAGE_PATH, 'Quercus palustris'),
            # 'Q. phellos':       join(BASE_IMAGE_PATH, 'Quercus phellos'),
            # 'Q. stellata':      join(BASE_IMAGE_PATH, 'Quercus stellata'),
             'A. saccharum':     join(BASE_IMAGE_PATH, 'Acer saccharum'),
             }
#the species to be tested against
ALL_SPECIES = ['Quercus accutissima',
               #'Quercus imbricaria',
               #'Quercus laurifolia',
               #'Quercus lyrata',
               'Quercus nigra',
               #'Quercus palustris',
               #'Quercus phellos',
               #'Quercus stellata',
               'Acer saccharum']
CSV_DATA_FILE = join(CWD, 'ann', 'leaf_data.csv')#file of just measurements
TRAINING_FILE = join(CWD, 'ann', 'leaf_train.data')
NET_FILE = join(CWD, 'ann', 'leaf_ann.net')
OUT_FILE = join(CWD, 'ann', 'output.txt')
SECTIONS = 70
OUTPUTS = len(ALL_SPECIES)
NUM_HIDDEN_NEURONS = 54

def write_to_output(line):
    with open(OUT_FILE, 'a') as output_file:
        output_file.write('[{0}] {1}\n'.format(asctime(localtime()), line))
        output_file.close()
#for use as custom load function with io.ImageCollection
if __name__ == "__main__":
    #generate new leaf measurements if no measurements  exist
    if not (exists(CSV_DATA_FILE)):
        print 'no leaf data is saved; measuring new data'
        leaf_collection = LeafCollection(sections=SECTIONS)
        for key in IMG_PATHS.keys():
            print'measuring {0}'.format(key)
            leaf_collection.load_leaves_folder(IMG_PATHS[key], num_sections=SECTIONS)
        #export all the data to pkl and csv
        leaf_collection.export_csv(CSV_DATA_FILE)
        print 'created leaf measurement data'
        write_to_output('Saved leaf data to "{0}"'.format(CSV_DATA_FILE))
    if not (exists(TRAINING_FILE)):
        create_fann_train_from_csv(CSV_DATA_FILE,
                                   TRAINING_FILE,
                                   SECTIONS,
                                   ALL_SPECIES)
        print 'training data created'
        write_to_output('Saved leaf training data to "{0}"'.format(TRAINING_FILE))
    if not (exists(NET_FILE)):
        #make the ANN and save it (automatically in training method
        neural_net = NeuralNetContainer()
        neural_net.train(TRAINING_FILE,
                         NET_FILE,
                         SECTIONS,
                         NUM_HIDDEN_NEURONS,
                         OUTPUTS)
        write_to_output('Saved leaf training data to "{0}"'.format(NET_FILE))
    #load an image of a leaf, and run it through all relevant variations of
    #the appropriate ANNs
    target_leaf = askopenfilename()
    write_to_output('User chose "{0}"'.format(target_leaf))
    leaf = Leaf(load_leaf(target_leaf), name='unknown leaf')
    leaf_measurements = leaf.measure_sections(SECTIONS, return_format='fann')
    chances = []
    #ANN stuff
    neural_net = NeuralNetContainer()
    neural_net.load_from_file(NET_FILE)
    results = neural_net.run_numbers(leaf_measurements)
    best_chance = ALL_SPECIES[results.index(max(results))]
    write_to_output('Leaf is most likely "{0}" with {1} confidence'.format(best_chance, int(max(results))))
    print results
    remove(TRAINING_FILE)
    remove(NET_FILE)
    message = 'The leaf is a {0} leaf with a ANN output number of {1}%'.format(best_chance, (max(results)+1)*50)
    showinfo('FANN Leaf Identification', message)
    
