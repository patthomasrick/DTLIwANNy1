def load_leaf_img(file_path):
    """
    Opens an image from a path, imports the image in greyscale, and resizes
    the image to the desired dimensions
    """
    desired_dim = [240, 480]
    _img = io.imread(file_path, as_grey=IMPORT_AS_GREY)
    return transform.resize(_img, desired_dim)
def load_img_collection(path):
    """
    Gets all the images in a directory in an
    ImageCollection object and returns
    the ImageCollection
    """
    image_path_list = join(path, '*.jpg')
    image_list = io.ImageCollection(image_path_list, load_func=load_leaf_img)
    return image_list
def save_to_csv(data, fname='leaf'):
    """
    Takes an image collection class,
    measures all the leaves, and then
    saves it all to a csv file
    """
    with open(join(CWD,
                   '{0}_data.csv'.format(fname)),
                   'w') as csvfile:
        csv_writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for line in data:
            csv_writer.writerow(line)
def pickle_leaves(data, fname):
    """
    Saves a leaf in pickle format from the
    default Python library, pickle
    """
    pickle.dump(data, open(join(CWD, fname), 'w'))
    
    
    
    
    
    """
    #load the leaf collection from the pickled files
    imported_data = create_fann_train_from_csv(CSV_DATA_FILE,
                                               TRAINING_FILE,
                                               SECTIONS,
                                               'Quercus imbricaria')
    print 'imported leave data from pickled files'
    #creates the training data for the ANN
    neural_net = NeuralNetContainer()
    neural_net.train(TRAINING_FILE, NET_FILE)
    #run numbers for every species in the imported_data dict and save it to
    #a csv file
    print 'trained neural network'
    ran_numbers = {}
    csv_list = []
    for key in imported_data.keys():
        if not (key in ran_numbers.keys()):
            ran_numbers[key] = []
        for data_str in imported_data[key]:
            numbers = [float(i) for i in data_str]
            ran_numbers[key].append(neural_net.run_numbers(numbers)[0])
    print 'Q. imbricaria data compared against itself'
    for key in ran_numbers.keys():
        csv_line = [str((i+1)/2) for i in ran_numbers[key]]
        csv_line.insert(0, key)
        csv_str = ' '.join(csv_line)
        csv_list.append(csv_line)
    with open(OUTPUT_FILE, 'w') as csvfile:
        csv_writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for line in csv_list:
            csv_writer.writerow(line)
    """