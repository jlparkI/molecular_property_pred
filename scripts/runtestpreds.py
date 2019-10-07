import numpy as np, os, pickle, gen_props_gc, sys
import convnet_gc
from convnet_gc import *


def main():
    #Before we do anything else, check to make sure a model was
    #already generated. If not, the user needs to run pipeline.sh
    #without -o first to train a model.
    if '%s_model'%sys.argv[1] not in os.listdir():
        print('Error: Attempting to generate predictions for the test set '
              'without first training a model. Please run pipeline.sh without '
              'the -o flag first to train a model, then run again with -o '
              'to generate test set predictions.')
        exit()
    
    #Load the molecule dictionary generated under gen_train_datasets.
    with open('moldict', 'rb') as inp:
        moldict = pickle.load(inp)

    numbondsdict = {'1JHC':1, '1JHN':1, '2JHH':2,
          '2JHN':2, '2JHC':2, '3JHH':3,
          '3JHC':3, '3JHN':3}
    #If there ARE any problem datapoints (shouldn't be, I fixed them all),
    #they will be written to duds.
    error_file = open('duds.csv', 'w+')
    print('Now parsing test.csv. This may take a minute...')
    features, ids = read_couplings(sys.argv[1], error_file,
                                   moldict, numbondsdict)
    make_preds(sys.argv[1], features, ids)
    print('**********\n\nTest set predictions generated for %s'%sys.argv[1])
    print('Predictions saved to file %s_preds.csv'%sys.argv[1])
    print('Problem datapoints (if any) are saved in the file duds.csv')
    print('To merge test set predictions for multiple coupling constant '
          'types, use the merge_preds script.')
    error_file.close()

#Read couplings calls the appropriate feature generation function in
#gen_props_gc and generates features for all test set datapoints using
#the molecule dictionary moldict to find the appropriate molecule
#for each datapoint. The features and ids are returned and passed to
#make_preds.
def read_couplings(coupling_type, error_file, moldict, numbondsdict):
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
    featuremats = np.stack(featuremats)
    print('Features generated. There are %s datapoints for this cc type.'
          ' Now making predictions.'%(featuremats.shape[0]))
    return np.stack(featuremats), ids

#Make preds uses the pickled model to make predictions for
#the test set and saves them to a csv file in the format
#requested by Kaggle. The .predict function is from
#convnet_gc, which stores all of the model details.
def make_preds(coupling_type, featuremats, ids):
    output_sheet = open('%s_preds.csv'%coupling_type, 'w+')
    with open('%s_model'%coupling_type, 'rb') as pickled_model:
        model = pickle.load(pickled_model)
    preds = model.predict(featuremats)
    for i in range(0, len(ids)):
        output_sheet.write(''.join([ids[i], ',', str(preds[i]), '\n']))
    output_sheet.close()
    del model

#####################
main()
