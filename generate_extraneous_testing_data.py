#!/usr/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class
# pylint: disable=too-many-locals
"""
A small script to measure some leaves
that the neural network will not have
seen before
"""
__author__ = "Patrick Thomas"
__version__ = "1.0"
__date__ = "1/5/16"

from swvgsleaf import Leaf, load_leaf
from os.path import join

leaf_imgs = {'Quercus accutissima':join('leaf-images', 'Quercus accutissima (18).jpg'),
             'Quercus imbricaria':join('leaf-images', 'Quercus imbricaria (19).jpg'),
             'Quercus laurifolia':join('leaf-images', 'Quercus laurifolia (18).jpg'),
             'Quercus lyrata':join('leaf-images', 'Quercus lyrata (12).jpg'),
             'Quercus nigra':join('leaf-images', 'Quercus nigra (19).jpg'),
             'Quercus palustris':join('leaf-images', 'Quercus palustris (14).jpg'),
             'Quercus phellos':join('leaf-images', 'Quercus phellos (18).jpg'),
             'Quercus stellata':join('leaf-images', 'Quercus stellata (28).jpg')}

def generate_run_testing_data(sections):
    leaf_measurements = {}
    for key in leaf_imgs.keys():
        leaf_path = leaf_imgs[key]
        leaf = Leaf(load_leaf(leaf_path), name=key)
        leaf_measurements[key] = leaf.measure_sections(sections, return_format='fann')
    return leaf_measurements