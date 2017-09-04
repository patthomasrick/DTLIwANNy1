#!/usr/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class
# pylint: disable=too-many-locals
"""
A class containing a leaf that can display
the edges of a leaf, find the endpoints of
the leaf, and measure the leaf's width at
several intervals
"""
__author__ = "Patrick Thomas"
__version__ = "1.0"
__date__ = "12/7/15"

import csv
from math import sqrt, acos, degrees
from os import listdir
from os.path import isfile, join
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.morphology import binary_dilation, binary_erosion, skeletonize, remove_small_objects, binary_closing
from scipy.ndimage.interpolation import rotate
# from scipy import ndimage as ndi
# from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt


# from skimage.transform import resize
# from skimage.measure import find_contours

def load_leaf(file_path):
    """
    Opens an image from a path, imports the image in greyscale, and resizes
    the image to the desired dimensions
    Mainly for use with the ImageCollection's custom load function
    """
    img = imread(file_path, as_grey=True)
    return img  # resize(img, [240, 480])


class Leaf():
    """
    A class to contain functions and measurements on a single leaf
    """

    def __init__(self, leaf_array, canny_sigma=2.5,
                 name='leaf'):
        """
        Initializes the leaf and its variables.
        The array is cropped slightly so avoid issues with
        the image's poor borders changing things.
        Also measures the leaf on initialization unless otherwise
        specified.
        No contours are gotten, since that are currently
        of no use.
        """
        self.name = name
        self.array = leaf_array[2:-2, 2:-2].copy()
        self.canny_sigma = canny_sigma
        self.edges_old = canny(self.array, self.canny_sigma)
        thresh = threshold_otsu(self.array)
        binary = -(self.array > thresh)
        binary = binary_erosion(binary_closing(binary_dilation(binary_dilation(binary))))
        self.edges = binary
        self.endpoints = self.find_extrema()
        # some useful parameters about the images themselves
        self.img_dim = self.array.shape[::-1]
        self.area = self.edges.sum()
        self.perimeter = canny(self.edges, 0.1).sum()
        veins = remove_small_objects(
            (self.edges_old - np.logical_and(self.edges_old, binary)),
            3,
            connectivity=1,
            in_place=True)
        veins = binary_dilation(binary_dilation(binary_dilation(veins)))
        veins = skeletonize(veins)
        self.vein_length = veins.sum()
        y_diff = self.endpoints[0][1] - self.endpoints[1][1]
        if (y_diff > 5) or (y_diff < -5):
            self.level_leaf()
        print
        'initialized leaf', self.name

    def find_extrema(self):
        """
        This will find the most extreme points on the leaf's edge (Leaf.edges)
        and then formulate the equation of a line between the two points.
        This'll start searching from the leaf side, top-to-down,
        then repeat the process from the right side.
        Assumes the leaf is oriented from stem-to-tip going left-to-right.
        """
        left_extrema = []  # stem
        right_extrema = []  # tip of leaf
        # Create an array of index values
        array_length = len(self.edges[0])
        left_to_right = range(0, array_length)
        for i in left_to_right:
            column = self.edges[:, i]
            # xvalue = np.argmax(self.edges[:, i])
            if column.any():
                left_extrema = [i, np.argmax(column)]
                break
        right_to_left = reversed(left_to_right)
        for i in right_to_left:
            column = self.edges[:, i]
            if column.any():
                right_extrema = [i, np.argmax(column)]
                break
        return [left_extrema, right_extrema]

    def level_leaf(self):
        """
        takes the extrema of the leaf, finds the
        hypotenuse, finds the angle between the base
        and the hypotenuse, rotates the image by that
        number, and reinitializes the leaf with the
        new image
        """
        left_endpoint, right_endpoint = self.endpoints
        # find the distance (length) between the two points (hypotenuse)
        diff_x = right_endpoint[0] - left_endpoint[0]
        diff_y = right_endpoint[1] - left_endpoint[1]
        hypot = sqrt((diff_x) ** 2 + (diff_y) ** 2)
        # get the angle between the endpoints to rotate the image by
        angle_radians = acos(diff_x / hypot)
        angle_degrees = degrees(angle_radians)
        array = self.array.copy()
        # rotate the image, preserving size
        if diff_y < 0:
            array = rotate(array, -angle_degrees, reshape=True, mode='nearest')
        else:
            array = rotate(array, angle_degrees, reshape=True, mode='nearest')
        # reinitzialize the image again
        self.__init__(leaf_array=array,
                      canny_sigma=self.canny_sigma,
                      name=self.name)

    def measure_sections(self, num_of_sections, return_format='point'):
        """
        split the leaf into <sections> number of
        sections and measure the width of each
        section
        """
        left_endpoint, right_endpoint = self.endpoints
        num_of_sections += 1
        # Find the difference in the y values between the
        # extrema and get the section length.
        # There are lots of floats so eliminate the
        # inaccuracy of dividing integers.
        length = float(right_endpoint[0]) - float(left_endpoint[0])
        section_length = float(length) / float(num_of_sections - 1)
        # get the sections from a range (free range sections lol)
        sections = [
            i * section_length + left_endpoint[0]
            for i in range(1, num_of_sections)]
        # iterate the sections and individually find the lengths
        section_results = []
        for j in sections:
            # gets the "slice" of the section and its reverse since
            # np.argmax(a) returns the FIRST maximum
            section_slice = self.edges[:, j]
            section_slice_rev = section_slice[::-1]
            # get the y-values for the slices
            top_y = np.argmax(section_slice)
            bot_y = self.img_dim[1] - np.argmax(section_slice_rev)
            # append the numbers to the lists for collection
            data = None
            if return_format == 'point':
                data = [[j, top_y], [j, bot_y]]
            elif return_format == 'xy':
                data = [[j, j], [top_y, bot_y]]
            elif return_format == 'width':
                data = [(bot_y - top_y), j]
            elif return_format == 'csv':
                if section_results == []:
                    data = self.name
                else:
                    width = (bot_y - top_y)
                    length = float(self.endpoints[1][0] - self.endpoints[0][0])
                    # the ratio of the difference to the length
                    data = width / length
            elif return_format == 'pickle':
                data = {
                    'name': self.name,
                    'index': sections.index(j),
                    'ratio': (bot_y - top_y) / length,
                    'width': (bot_y - top_y) / length,
                    'length': length,
                    'area': self.area / length,
                    'perimeter': self.perimeter / length,
                    'area/perimeter': self.area / self.perimeter,
                    'vein length': self.vein_length,
                    'top_point': [j, top_y],
                    'bot_point': [j, bot_y],
                    'xy_form': [[j, j], [top_y, bot_y]]
                }
            elif return_format == 'fann':
                if j == sections[-1]:
                    section_results.append(bot_y - top_y)
                    section_results.append(length)
                    section_results.append(self.area)
                    data = self.perimeter
                else:
                    data = bot_y - top_y
            else:
                raise ValueError(
                    'Nonexistant format for returning data')
            section_results.append(data)
        print
        "measured {0} leaf".format(self.name)
        return section_results

    def show_all(self):
        """
        Displays the image, edges, and boolean edges
        (essentially same as edges) in 2 rows in a window
        """
        section_xy = self.measure_sections(
            num_of_sections=20,
            return_format='xy')
        (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(14, 4))[1]
        ax1.set_title('Original image')
        ax1.imshow(self.array, cmap='gray')
        ax2.set_title('Edges, endpoints, and sections')
        ax2.imshow(np.invert(self.edges), cmap='gray')
        # plot endpoints
        ax2.plot(self.endpoints[0][0], self.endpoints[0][1], 'ro',
                 self.endpoints[1][0], self.endpoints[1][1], 'ro')
        ax2.plot([self.endpoints[0][0], self.endpoints[1][0]],
                 [self.endpoints[0][1], self.endpoints[1][1]],
                 'r-',
                 linewidth=2)
        # plot sections
        for sec in section_xy:
            ax2.plot(sec[0],
                     sec[1],
                     'g-',
                     linewidth=2,
                     label=str(sec[0][1] - sec[1][1]))
            plt.annotate(s=str(sec[1][1] - sec[1][0]),
                         xy=(sec[0][1], sec[1][1]),
                         xytext=(sec[0][1], sec[1][1] + 5))
        # limit edge window
        ax2.set_xlim([0, self.img_dim[0]])
        ax2.set_ylim([0, self.img_dim[1]])
        plt.grid(True)
        plt.ylabel('Height in pixels')
        plt.xlabel('Length in pixels')
        plt.show()

    def show_section_measure(self, sections):
        """
        Plots the section measure from the xy format along
        with its endpoints
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title('Width by sections')
        section_xy = self.measure_sections(
            num_of_sections=sections,
            return_format='xy')
        for sec in section_xy:
            plt.plot(sec[0], sec[1], 'g-', linewidth=1)
        ax1.plot(self.endpoints[0][0], self.endpoints[0][1], 'ro',
                 self.endpoints[1][0], self.endpoints[1][1], 'ro')
        ax1.set_xlim([0, self.img_dim[0]])
        ax1.set_ylim([0, self.img_dim[1]])
        plt.show()


class LeafCollection():
    """
    A class to contain and do batch processes to leaves
    """

    def __init__(self, sections=25):
        """
        Initializes the leaf collection
        """
        self.leaf_dict = {}
        self.leaf_dict_data = {}
        self.sections = sections

    def __getitem__(self, key):
        try:
            return self.leaf_dict[key]
        except KeyError:
            print
            'No genus "{0}" in self.leaf_dict.'.format(key)

    def __str__(self):
        return str(self.leaf_dict)

    def load_leaves_folder(self, folder_path, num_sections=25, measure=True):
        """
        Loads a folder of leaves, assuming the name of the
        folder is the binomial nomenclature name and each
        image within the folder is of the same species as the
        title.
        """
        # get the name of the leaf from the path
        directories = folder_path.split('/')
        fullname = directories[-1]
        genus_species = fullname.split()
        genus, species = genus_species
        # add the names to the dictionary
        self.add_genus(genus)
        self.add_species(genus, species)
        # load the ImageCollection
        # gets all the images from inside the folder, assuming all are .jpg
        leaf_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        leaf_imgs = []
        for leaf_file in leaf_files:
            fname = join(folder_path, leaf_file)
            try:
                leaf_imgs.append(load_leaf(join(folder_path, leaf_file)))
            except IOError:
                print
                fname, 'is not an image of a leaf.'
        """
        leaf_imagecollection = ImageCollection(
            join(folder_path, '*.jpg'),
            load_func=load_leaf)
            """
        for leaf_img in leaf_imgs:
            new_leaf = Leaf(leaf_img,
                            name='{0} {1}'.format(genus, species))
            self.leaf_dict[genus][species].append(new_leaf)
        if measure:
            self.measure_all_leaves(num_sections)
        else:
            self.sections = num_sections
            for single_genus in self.leaf_dict.keys():
                genus = self.leaf_dict[single_genus]
                for single_species in genus.keys():
                    species = genus[single_species]
                    for single_leaf in species:
                        self.leaf_dict_data[single_leaf] = None

    def add_genus(self, genus_name):
        """
        Adds the genus, genus_name, to the leaf_dict.
        If the genus already exists, then it refuses.
        """
        if not self.leaf_dict.has_key(genus_name):
            self.leaf_dict[genus_name] = {}

    def add_species(self, genus_name, species_name):
        """
        Adds the species, species_name, to the leaf_dict
        under the genus, genus name.
        If the species already exists, then it refuses.
        If the genus does not exist, then it refuses.
        """
        if self.leaf_dict.has_key(genus_name):
            if not self.leaf_dict[genus_name].has_key(species_name):
                self.leaf_dict[genus_name][species_name] = []

    def measure_all_leaves(self, num_sections=5):
        """
        Measures all the leaves in the leaf_data_dict.
        Could possibly be improved by iterating through
        the main leaf_dict instead of just loooking at
        the data dict.
        """
        self.sections = num_sections
        for single_genus in self.leaf_dict.keys():
            genus = self.leaf_dict[single_genus]
            for single_species in genus.keys():
                species = genus[single_species]
                for single_leaf in species:
                    measurements = single_leaf.measure_sections(
                        num_of_sections=num_sections,
                        return_format='pickle')
                    self.leaf_dict_data[single_leaf] = measurements

    def export_csv(self, fname):
        """
        gets the measurements of all leaves from the
        leaf_data_dict dictionary and puts the data into line,
        which then gets written to a .csv file
        the data dictionary is in this format:
        {leaf:[{data_1}...{data_n}]}
        """
        if None in self.leaf_dict_data.values():
            self.measure_all_leaves(self.sections)
        all_lines = [
            ['LEAF MEASUREMENTS'],
            ['']]
        # gets all the leaves from the leaf_dict, then goes back
        # through the data_dict
        for single_genus in self.leaf_dict.keys():
            genus = self.leaf_dict[single_genus]
            for single_species in genus.keys():
                species = genus[single_species]
                for single_leaf in species:
                    measurements_list = self.leaf_dict_data[single_leaf]
                    leaf_name = measurements_list[0]['name']
                    line_buffer = [leaf_name]
                    for measurement in measurements_list:
                        # assume the leaf is in order
                        line_buffer.append(measurement['width'])
                        print
                        measurement['width']
                    line_buffer.append(measurements_list[0]['length'])
                    line_buffer.append(measurements_list[0]['area'])
                    line_buffer.append(measurements_list[0]['perimeter'])
                    line_buffer.append(measurements_list[0]['area/perimeter'])
                    line_buffer.append(measurements_list[0]['vein length'])
                    all_lines.append(line_buffer)
                all_lines.append([])
        with open(fname, 'w') as csvfile:
            csv_writer = csv.writer(csvfile,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            for line in all_lines:
                csv_writer.writerow(line)

    def export_fann_train(self, fname, des_genus, des_species):
        """
        Gets all the measurements for all the leaves, then exports
        it to a ANN friendly training data set for a specific leaf.
        """
        total_leaves = 0
        all_leaf_data = []
        for genus_name in self.leaf_dict.keys():
            genus = self.leaf_dict[genus_name]
            for species_name in genus.keys():
                leaves = genus[species_name]
                total_leaves += len(leaves)
                for leaf in leaves:
                    leaf_data = self.leaf_dict_data[leaf]
                    data_string = ''
                    for section in leaf_data:
                        data_string = data_string + ' ' + str(section['width'])
                    all_leaf_data.append(data_string.strip() + '\n')
                    if self.leaf_dict[des_genus][des_species] == leaves:
                        all_leaf_data.append('1\n')
                    else:
                        all_leaf_data.append('-1\n')
        train_data_header = '{0} {1} {2}\n'.format(total_leaves, self.sections, 1)
        all_lines = [train_data_header]
        for line in all_leaf_data:
            all_lines.append(line)
        with open(fname, 'w') as train_file:
            for line in all_lines:
                train_file.write(line)

    def export_leaf_ratios(self, leaf):
        leaf_data = self.leaf_dict_data[leaf]
        ratios = []
        for section in leaf_data:
            ratios.append(section['ratio'])
        return ratios


def create_fann_train_from_csv(csv_fname, train_fname, sections, all_species):
    """
    takes the csv file saved by the leaf collection and creates FANN training
    data from it.
    """
    # format 'binomial nomenclature: [leaf measurements]
    imported_data = {}
    total_leaves = 0
    # read and parse the csv file
    with open(csv_fname, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            if (len(row) == sections + 6):
                if not (row[0] in imported_data.keys()):
                    imported_data[row[0]] = []
                imported_data[row[0]].append(row[1:])
    # get the total number of leaves. this is needed for the
    # fann training data
    for key in imported_data.keys():
        total_leaves += len(imported_data[key])
    # make the "header" for the training data
    # format (total data trainings), inputs, outputs

    # CHANCE THE INT ADDED TO SECTIONS WHEN MORE DATA IS ADDED#############################
    data_list = ['{0} {1} {2}'.format(total_leaves, sections + 5, len(all_species))]
    for key in imported_data.keys():
        for leaf_data in imported_data[key]:
            data_line = ' '.join(leaf_data)
            data_list.append(data_line)
            output_list = []
            for lname in all_species:
                if lname == key:
                    output_list.append(1)
                else:
                    output_list.append(-1)
            output_line = ' '.join([str(i) for i in output_list])
            data_list.append(output_line)
    # write the data to a file
    with open(train_fname, 'w') as train_file:
        for line in data_list:
            train_file.write((line + '\n'))
    return imported_data
