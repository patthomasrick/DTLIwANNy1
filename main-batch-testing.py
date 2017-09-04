#!/usr/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class
from main import BASE_IMAGE_PATH
"""
Takes an image, converts it to edges, then analyses it in a neural network
to determine the species of the leaf
"""
__author__ = "Patrick Thomas"
__version__ = "1.0"
__date__ = "12/7/15"

import csv
from jinja2.nodes import Pair
from libfann import training_data
from swvgsleaf import LeafCollection, create_fann_train_from_csv, Leaf, load_leaf
from neural_net_tools import NeuralNetContainer
from os import getcwd, remove
from os.path import join, exists
from generate_extraneous_testing_data import generate_run_testing_data
#from skimage.measure import find_contours
#from skimage.feature import canny
#from scipy.ndimage.interpolation import rotate
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.misc import imsave

IMPORT_AS_GREY = True
CWD = getcwd()
BASE_IMAGE_PATH = join(CWD, 'leaf-images')
QSTELLATA_PATH = join(BASE_IMAGE_PATH, 'Quercus stellata')
QIMBRICARIA_PATH = join(BASE_IMAGE_PATH, 'Quercus imbricaria')
IMG_PATHS = {'Q. accutissima':  join(BASE_IMAGE_PATH, 'Quercus accutissima'),
             'Q. imbricaria':   join(BASE_IMAGE_PATH, 'Quercus imbricaria'),
             'Q. laurifolia':   join(BASE_IMAGE_PATH, 'Quercus laurifolia'),
             'Q. lyrata':       join(BASE_IMAGE_PATH, 'Quercus lyrata'),
             'Q. nigra':        join(BASE_IMAGE_PATH, 'Quercus nigra'),
             'Q. palustris':    join(BASE_IMAGE_PATH, 'Quercus palustris'),
             'Q. phellos':      join(BASE_IMAGE_PATH, 'Quercus phellos'),
             'Q. stellata':     join(BASE_IMAGE_PATH, 'Quercus stellata'),
             'A. saccharum':    join(BASE_IMAGE_PATH, 'Acer saccharum'),
             'B. lenta':        join(BASE_IMAGE_PATH, 'Betula lenta'),
             'C. dentata':      join(BASE_IMAGE_PATH, 'Castanea dentata'),
             }
TRAINING_FILE = join(CWD, 'ann', 'ann.data')
NET_FILE = join(CWD, 'ann', 'ann.net')
#[3,4,5,6,7,8,9,10]
SECTIONS = [70]
NUM_HIDDEN_NEURONS = range(30,201,20)
#the species to be tested against
ALL_SPECIES = ['Quercus accutissima',
               'Quercus imbricaria',
               'Quercus laurifolia',
               'Quercus lyrata',
               'Quercus nigra',
               'Quercus palustris',
               'Quercus phellos',
               'Quercus stellata',
               'Acer saccharum',
               'Betula lenta',
               'Castanea dentata',
               ]
ALL_SPECIES.sort()
ALL_TEST_LEAVES = [join(BASE_IMAGE_PATH, 'Quercus accutissima (18).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus imbricaria (19).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus laurifolia (18).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus lyrata (12).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus nigra (19).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus palustris (14).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus phellos (18).jpg'),
                   join(BASE_IMAGE_PATH, 'Quercus stellata (28).jpg'),
                   join(BASE_IMAGE_PATH, 'Acer saccharum (4).jpg'),
                   join(BASE_IMAGE_PATH, 'Betula lenta (19).jpg'),
                   join(BASE_IMAGE_PATH, 'Castanea dentata (13).jpg'),
                   ]
ALL_TEST_LEAVES.sort()
if not (len(IMG_PATHS.keys()) == len(ALL_SPECIES) == len(ALL_TEST_LEAVES)):
    raise('check leaves')
#for use as custom load function with io.ImageCollection
if __name__ == "__main__":
    choices = {s: [] for s in ALL_TEST_LEAVES}
    print 'loading leaves...'
    leaf_collection = LeafCollection(sections=SECTIONS[0])
    for key in IMG_PATHS.keys():
        leaf_collection.load_leaves_folder(IMG_PATHS[key], num_sections=SECTIONS[0], measure=True)
    true_leaf_list = [Leaf(load_leaf(ALL_TEST_LEAVES[ALL_SPECIES.index(s)]), name=s) for s in ALL_SPECIES]
    true_leaf_names = [l.name for l in true_leaf_list]
    print 'loaded'
    try:
        for section in SECTIONS:
            new_neurons = (section+1+len(ALL_SPECIES))*2/3
            for i in range(1,1001):
                csv_data_fname = join(CWD,
                                      'ann',
                                      'leaf_measurements_sep{0}.csv'.format(section,
                                                                                       new_neurons))
                if not (exists(csv_data_fname)):
                    print 'no leaf data is saved; measuring new data'
                    leaf_collection.sections = section
                    leaf_collection.measure_all_leaves(section)
                    #export all the data to pkl and csv
                    leaf_collection.export_csv(csv_data_fname)
                    print 'created leaf measurement data'
                imported_data = create_fann_train_from_csv(
                   csv_data_fname,
                   TRAINING_FILE,
                   section,
                   ALL_SPECIES)
                print '{0} sections, {1} hidden neurons, run {2}'.format(section, new_neurons, i)
                neural_net = NeuralNetContainer()
                neural_net.train(TRAINING_FILE,
                                 NET_FILE,
                                 section,
                                 new_neurons,
                                 len(ALL_SPECIES))
                for leaf_fname in ALL_TEST_LEAVES:
                    true_species = true_leaf_list[ALL_TEST_LEAVES.index(leaf_fname)].name
                    leaf = true_leaf_list[ALL_TEST_LEAVES.index(leaf_fname)]
                    leaf_measurements = leaf.measure_sections(section, return_format='fann')
                    neural_net = NeuralNetContainer()
                    neural_net.load_from_file(NET_FILE)
                    results = neural_net.run_numbers(leaf_measurements)
                    best_chance = ALL_SPECIES[results.index(max(results))]
                    choices[leaf_fname].append([section,
                                                new_neurons,
                                                best_chance,
                                                float(max(results)+1)/float(2)])
                remove(NET_FILE)
    except KeyboardInterrupt:
        pass
    results_dict = {}
    csv_lines = [['num sections', 'num hidden neurons', 'right', 'wrong', 'percent']]
    for leaf_fname in choices.keys():
        test_leaf = choices[leaf_fname]
        for trial in test_leaf:
            true_species = ALL_SPECIES[ALL_TEST_LEAVES.index(leaf_fname)]
            #add the specific leaf to the dictionary if it doesnt exist
            if not results_dict.has_key(trial[0]):
                results_dict[trial[0]] = {}
            #nests the next information within
            if not results_dict[trial[0]].has_key(trial[1]):
                results_dict[trial[0]][trial[1]] = [0, 0]#right, wrong
            if trial[2] == true_species:
                results_dict[trial[0]][trial[1]][0] += 1
            elif trial[2] != true_species:
                results_dict[trial[0]][trial[1]][1] += 1
    for num_sections in results_dict.keys():
        for num_hidden_neurons in results_dict[num_sections].keys():
            l = results_dict[num_sections][num_hidden_neurons]
            csv_lines.append(
                [num_sections, num_hidden_neurons, l[0], l[1], float(l[0])/float(l[0]+l[1])])
    with open(join(CWD, 'final_output_tallies.csv'), 'w') as csvfile:
        csv_writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for line in csv_lines:
            csv_writer.writerow(line)
            
    csv_lines = [['true species', 'guessed species', 'correct?']]
    for leaf_fname in choices.keys():
        test_leaf = choices[leaf_fname]
        for trial in test_leaf:
            true_species = ALL_SPECIES[ALL_TEST_LEAVES.index(leaf_fname)]
            correct = 0
            if trial[2] == true_species:
                correct = 1
            #add the specific leaf to the dictionary if it doesnt exist
            csv_lines.append([true_species, trial[2], correct])
    with open(join(CWD, 'final_output_exact.csv'), 'w') as csvfile:
        csv_writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for line in csv_lines:
            csv_writer.writerow(line)