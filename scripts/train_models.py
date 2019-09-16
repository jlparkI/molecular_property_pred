import numpy as np, pickle,time
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from importlib import reload
import convnet_gc
from convnet_gc import *


####JHC1 model
x = np.load('train_1JHCx.npy')
y = np.load('train_1JHCy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25)
monika = fcn_gc()
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(200)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(200)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(200)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(200)
with open('1JHC_model', 'wb') as out:
	pickle.dump(monika, out)

del x,y,xtrain,ytrain,xtest,ytest
######################################

################jhh2

x = np.load('train_2JHHx.npy')
y = np.load('train_2JHHy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.0)
monika = fcn_gc()
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(120)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(120)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(120)
monika.train(xtrain, ytrain, minibatch=200, epochs=5, lr=0.002)
time.sleep(120)
with open('2JHH_model', 'wb') as out:
	pickle.dump(monika, out)

############jhh3
x = np.load('train_3JHHx.npy')
y = np.load('train_3JHHy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.0)
monika = fcn_gc()
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=300, epochs=5, lr=0.002)
time.sleep(60)
with open('3JHH_model', 'wb') as out:
	pickle.dump(monika, out)
del x, y, xtrain, ytrain



##################1JHN

x = np.load('train_1JHNx.npy')
y = np.load('train_1JHNy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
monika = fcn_gc(exp_dim1=100, exp_dim2=200, exp_dim3=100)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(60)


with open('1JHN_model', 'wb') as out:
	pickle.dump(monika, out)
del x,y,xtrain,ytrain,xtest,ytest









###########JHC2 model
y = np.memmap('train_2JHCy.npy', dtype='float', mode='r')
x = np.memmap('train_2JHCx.npy', dtype='float', mode='r',
		shape=(y.shape[0], 29, 58))
monika = fcn_gc()
monika.train(x, y, minibatch=300, epochs=10, lr=0.002)
time.sleep(120)
monika.train(x, y, minibatch=300, epochs=10, lr=0.002)
time.sleep(120)
monika.train(x, y, minibatch=300, epochs=10, lr=0.002)
time.sleep(120)
monika.train(x, y, minibatch=300, epochs=10, lr=0.002)
time.sleep(120)
monika.train(x, y, minibatch=300, epochs=10, lr=0.002)
time.sleep(120)
with open('2JHC_model_new', 'wb') as out:
	pickle.dump(monika, out)
del x,y,xtrain,ytrain,xtest,ytest




############3JHC
x = np.load('train_3JHCx.npy')
y = np.load('train_3JHCy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.0)
monika = fcn(input_dim=25, exp_dim1=400, exp_dim2 = 200)
monika.train(xtrain, ytrain, minibatch=200, epochs=10, lr=0.002)
time.sleep(240)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(240)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(240)
monika.train(xtrain, ytrain, minibatch=300, epochs=10, lr=0.002)
time.sleep(60)
with open('3JHC_model', 'wb') as out:
	pickle.dump(monika, out)
del x, y, xtrain, ytrain


############3JHN
x = np.load('train_3JHNx.npy')
y = np.load('train_3JHNy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
monika = fcn3b(input_dim=26)
monika.train(xtrain, ytrain, minibatch=200, epochs=20, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=100, epochs=20, lr=0.002)
with open('3JHN_model', 'wb') as out:
	pickle.dump(monika, out)
del x, y, xtrain, ytrain



############jhn2
x = np.load('train_2JHNx.npy')
y = np.load('train_2JHNy.npy')
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
monika = fcn_gc()
monika.train(xtrain, ytrain, minibatch=200, epochs=20, lr=0.002)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=100, epochs=10, lr=0.001)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=100, epochs=10, lr=0.001)
time.sleep(60)
monika.train(xtrain, ytrain, minibatch=100, epochs=10, lr=0.001)
time.sleep(60)
with open('2JHN_model', 'wb') as out:
	pickle.dump(monika, out)
del x, y, xtrain, ytrain



