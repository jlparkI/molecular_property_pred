#We will use sys.argv to import arguments from pipeline. Python getopts is redundant
#here since pipeline.sh already parses user arguments.
import numpy as np, os, pickle, sys
import gen_props_gc


#We will use a pickled dictionary of all of the molecules in the
#dataset so we can quickly access the appropriate molecule for a
#given datapoint as we read them from the training file. If
#the pickled dictionary does not already exist, we will create it.
#Otherwise, load it. Then call the train_extraction function
#to process the data.
def main():
    if 'moldict' not in os.listdir():
        moldict = dict()
        os.chdir('sdfs')
        for i, filename in enumerate(os.listdir()):
            try:
                suppl = Chem.SDMolSupplier(filename, removeHs=False)
                mol = suppl[0]
                moldict[filename.split('.sdf')[0]] = mol
            except:
                print('Error for file %s'%filename)
            if i % 10000 == 0:
                print('%s complete'%i)
        os.chdir('..')
        with open('moldict', 'wb') as out:
            pickle.dump(moldict, out)
    else:
        with open('moldict', 'rb') as inp:
            moldict = pickle.load(inp)

    numbonds = int(sys.argv[1][0])
    #If we are working with the 2JHC dataset, the number of
    #datapoints for this dataset is so large we can't build the features in
    #memory. In that case, train_extraction will build then
    #merge separate files using numpy memory-mapping.
    #Also, before generating features we check to make sure they weren't already generated...
    if 'train_%sx.npy'%(sys.argv[1]) in os.listdir():
        print('features already generated')
    else:
        print('Generating features for %s, dataset is %s. '
              'This may take a minute... will print updates'
              ' every 200000 datapoints.'%(sys.argv[1], sys.argv[2]))
        if sys.argv[2] == 'large':
            train_extraction(sys.argv[1], numbonds, moldict, large=True)
        else:
            train_extraction(sys.argv[1], numbonds, moldict)
            


#########Generate features for training set data
def train_extraction(coupling_type, numbonds, moldict, large=False):
    featuremats, groundtruths = [], []
    counter, filecounter = 0, 0
    with open('train.csv') as input_file:
        input_file.readline()
        for line in input_file:
            segments = line.strip().split(',')
            mol = moldict[segments[1]]
            if segments[4] == coupling_type:
                #Make sure the molecule associated with this datapoint
                #has an entry in the molecule dictionary from above.
                if mol is not None:
                    #Skipping lines 37 and 38 because those two
                    #have associated errors (need to come up with a more
                    #permanent fix)
                    if counter != 37 and counter != 38:
                        featuremats.append(gen_props_gc.
                                        gen_props_charges(mol, int(segments[2]),
                                        int(segments[3]), numbonds=numbonds))
                        groundtruths.append(float(segments[5]))
                    counter += 1
                    #If dealing with a large dataset (e.g. 2JHC), create
                    #separate files with subsets of the dataset and
                    #merge them later.
                    if counter % 200000 == 0:
                        print('200000 lines parsed')
                        if large == True:
                            dump_data(featuremats, groundtruths, filecounter, coupling_type)
                            filecounter += 1
                            featuremats, groundtruths = [], []
    print('Writing features to file...')                
    dump_data(featuremats, groundtruths, filecounter, coupling_type, large)
    if large == False:
        print('features generated: dimensions of trainset tensor are: %s x %s x %s'%(len(featuremats),
                                                                      featuremats[0].shape[0],
                                                                      featuremats[0].shape[1]))
    else:
        filecounter += 1
        merge_files(coupling_type, filecounter)
    del featuremats, groundtruths

#Convenience function that dumps the assembled data into a numpy file
#for loading during model training.
def dump_data(featuremats, groundtruths, filecounter, coupling_type, large=False):
    groundtruths = np.asarray(groundtruths)
    featuremats = np.stack(featuremats)
    indices = np.random.choice(groundtruths.shape[0], size=groundtruths.shape[0],
                               replace=False)
    featuremats = featuremats[indices,:,:]
    groundtruths = groundtruths[indices]
    if large == True:
        np.save('train_%sx_%s.npy'%(coupling_type, filecounter), featuremats)
        np.save('train_%sy_%s.npy'%(coupling_type, filecounter), groundtruths)
    else:
        np.save('train_%sx.npy'%(coupling_type), featuremats)
        np.save('train_%sy.npy'%(coupling_type), groundtruths)

#This function is only used for large datasets (e.g. 2JHC), where
#subsets of the data are temporarily stored in separate temporary files
#which are then memory-mapped and merged.
def merge_files(coupling_type, file_counter):
    x_files, y_files = [], []
    rows = 0
    for i in range(0, file_counter):
        x_files.append('train_%sx_%s.npy'%(coupling_type, i))
        y_files.append('train_%sy_%s.npy'%(coupling_type, i))
    for data_file in x_files:
        print('checking %s'%data_file)
        current = np.load(data_file)
        rows += current.shape[0]
    mergedx = np.memmap('train_%sx.npy'%coupling_type, dtype='float', mode='w+', shape=(rows, 29, 58))
    mergedy = np.memmap('train_%sy.npy'%coupling_type, dtype='float', mode='w+', shape=(rows))
    rows = 0
    for i in range(0, file_counter):
        print('merging %s'%x_files[i])
        current_x = np.load(x_files[i])
        current_y = np.load(y_files[i])
        mergedx[rows:rows+current_x.shape[0],:,:] = current_x
        mergedy[rows:rows + current_y.shape[0]] = current_y
        rows += current_x.shape[0]
        del current_x, current_y
    del mergedx, mergedy
    for i in range(0, file_counter):
        os.remove(x_files[i])
        os.remove(y_files[i])


#########
main()
