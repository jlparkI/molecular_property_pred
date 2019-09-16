import numpy as np, os, pickle, gen_props_gc
import convnet_gc
from convnet_gc import *

with open('moldict', 'rb') as inp:
    moldict = pickle.load(inp)

numbondsdict = {'1JHC':1, '1JHN':1, '2JHH':2,
          '2JHN':2, '2JHC':2, '3JHH':3,
          '3JHC':3, '3JHN':3}
use_rd2_dict = {'1JHC':True, '1JHN':True, '2JHH':True,
          '2JHN':True, '2JHC':True, '3JHH':True,
          '3JHC':False, '3JHN':True}

rerun_list = ['2JHC']
error_file = open('duds.csv', 'w+')

def read_couplings(coupling_type, exp=False):
    featuremats, ids = [], []
    with open('test.csv') as input_file:
        input_file.readline()
        for counter, line in enumerate(input_file):
            segments = line.strip().split(',')
            mol = moldict[segments[1]]
            if segments[4] == coupling_type:
                if mol is not None:
                    numbonds = numbondsdict[coupling_type]
                    featuremats.append(
                    gen_props_gc.gen_props_charges(mol, int(segments[2]),
                                int(segments[3]), numbonds=numbonds))
                    ids.append(segments[0])
                else:
                    print(line)
                    _ = error_file.write(line)
            if counter % 100000 ==0:
                print(counter)
    featuremats = np.stack(featuremats)
    return np.stack(featuremats), ids

def make_preds(coupling_type, featuremats, ids):
    output_sheet = open('%s_preds.csv'%coupling_type, 'w+')
    os.chdir('complete_models')
    with open('%s_model'%coupling_type, 'rb') as pickled_model:
        model = pickle.load(pickled_model)
    preds = model.predict(featuremats)
    for i in range(0, len(ids)):
        output_sheet.write(''.join([ids[i], ',', str(preds[i]), '\n']))
    os.chdir('..')
    output_sheet.close()
    del model



    
for key in rerun_list:
    features, ids = read_couplings(key, use_rd2_dict[key])
    make_preds(key, features, ids)
    print('Success for %s'%key)

error_file.close()
