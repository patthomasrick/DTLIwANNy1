for key in IMG_PATHS.keys():
        start_time = time.time()
        print"importing {0}".format(key)
        LEAF_COLLECTION.load_leaves_folder(IMG_PATHS[key], num_sections=SECTIONS)
        print """done importing and measuring all {0} with
{1} sections per leaf in {2} seconds""".format(key,
                                               str(SECTIONS),
                                               str(time.time()-start_time))