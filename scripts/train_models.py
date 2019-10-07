import numpy as np, pickle,time, sys
from sklearn.model_selection import train_test_split
import convnet_gc
from convnet_gc import *


def main():
    #If large is passed, read data from memmapped numpy file.
    if sys.argv[2] == 'large':
        y = np.memmap('train_%sy.npy'%sys.argv[1], dtype='float', mode='r')
        x = np.memmap('train_%sx.npy'%sys.argv[1], dtype='float', mode='r',
		shape=(y.shape[0], 29, 58))
    else:
        #Otherwise, it fits into memory because we built it in memory.
        x = np.load('train_%sx.npy'%sys.argv[1])
        y = np.load('train_%sy.npy'%sys.argv[1])
    #When first building this pipeline, I used the next line of code
    #to make train validation splits to evaluate hyperparameters.
    #At this point, however, it is just being used to shuffle the
    #data. This is crucial, otherwise a minibatch may consist of datapoints
    #belonging to just a couple molecules, resulting in very slow
    #convergence.
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.0)
    print('\n*************\nNow training model for 35 - 50 epochs (depending on coupling '
          'constant type). Loss is calculated as mean sum square of deviations.'
          'Loss on current minibatch is printed every 50 minibatches, together with a '
          'notification at the end of each epoch. The script will '
          'pause for 30 seconds after every 10 epochs so you have time to review.\n*********************')
    time.sleep(15)
    #Create the model object (see convnet_gc for details). We are just
    #using default parameters & number of epochs for all of these, could get better
    #results by fine-tuning a little -- time pressure prevailed here
    #though.
    nmr_model = fcn_gc()
    nmr_model.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
    time.sleep(30)
    nmr_model.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
    time.sleep(30)
    nmr_model.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
    time.sleep(30)
    if sys.argv[1] not in ['3JHN', '3JHH', '1JHN', '2JHN']:
        nmr_model.train(xtrain, ytrain, minibatch=100, epochs=10, lr=0.002)
    else:
        nmr_model.train(xtrain, ytrain, minibatch=100, epochs=5, lr=0.002)
    #The largest dataset needs to train for another 5 epochs.
    if sys.argv[1] == '2JHC':
        time.sleep(30)
        nmr_model.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
    #Pickle the model for later retrieval
    print('****************\n\ntraining complete. saving model to file. '
          'Re-run the pipeline '
          'script with -o to generate predictions for the test set. Or, '
          're-run without -o to rebuild / retrain model.')
    with open('%s_model'%sys.argv[1], 'wb') as out:
        pickle.dump(nmr_model, out)


################
main()
