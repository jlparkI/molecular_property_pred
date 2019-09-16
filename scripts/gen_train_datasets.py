import numpy as np, os, pickle, sys
import gen_props_gc


if len(sys.argv) == 3 and sys.argv[2] == 'new':
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
            


#########Generate features for training set data
def train_extraction(coupling_type, numbonds, large=False):
    featuremats, groundtruths = [], []
    counter, filecounter = 0, 0
    with open('train.csv') as input_file:
        input_file.readline()
        for line in input_file:
            segments = line.strip().split(',')
            mol = moldict[segments[1]]
            if segments[4] == coupling_type:
                if mol is not None:
                    if counter != 37 and counter != 38:
                        featuremats.append(gen_props_gc.
                                        gen_props_charges(mol, int(segments[2]),
                                        int(segments[3]), numbonds=numbonds))
                        groundtruths.append(float(segments[5]))
                    counter += 1
                    if counter % 200000 == 0:
                        if large == True:
                            dump_data(featuremats, groundtruths, filecounter, coupling_type)
                            filecounter += 1
                            featuremats, groundtruths = [], []
                        
    dump_data(featuremats, groundtruths, filecounter, coupling_type)
    del featuremats, groundtruths
    if large == True:
        filecounter += 1
        merge_files(coupling_type, filecounter)

def dump_data(featuremats, groundtruths, filecounter, coupling_type):
    groundtruths = np.asarray(groundtruths)
    featuremats = np.stack(featuremats)
    print(featuremats.shape)
    indices = np.random.choice(groundtruths.shape[0], size=groundtruths.shape[0],
                               replace=False)
    featuremats = featuremats[indices,:,:]
    groundtruths = groundtruths[indices]
    np.save('train_%sx_%s.npy'%(coupling_type, filecounter), featuremats)
    np.save('train_%sy_%s.npy'%(coupling_type, filecounter), groundtruths)


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
    for i in range(0, file_counter):
        os.remove(x_files[i])
        os.remove(y_files[i])

numbonds = int(sys.argv[1][0])
if sys.argv[1] == '2JHC':
    train_extraction(sys.argv[1], numbonds, large=True)
else:
    train_extraction(sys.argv[1], numbonds)
