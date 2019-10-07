import os, time
import torch
import numpy as np
import torch.nn.functional as F

#This is set up to run on GPU. I would set it up to run on CPU as well,
#but to be honest, I suspect training time on CPU would be long enough
#it just wouldn't be worth it.
#device = torch.device('cpu')
device = torch.device('cuda')


#This first part is pretty self-explanatory. We create a model object, use
#super so it inherits attributes common to pytorch models and create
#the appropriate layers (these are described in the ReadMe). The
#normalization could use some work -- if I had time would revisit that part.
class fcn_gc(torch.nn.Module):
    def __init__(self, l2 = 0, dropout = 0.0,
                 input_dim=24, exp_dim1=100, exp_dim2=100, exp_dim3=200):
        super(fcn_gc, self).__init__()
        self.l2 = l2
        self.dropout = dropout
        self.inner_dim1 = int(0.5*exp_dim1)
        self.inner_dim2 = int(0.5*exp_dim2)
        self.inner_dim3 = int(0.5*exp_dim3)
        self.lnorm1 = torch.nn.LayerNorm(self.inner_dim1)
        self.lnorm2 = torch.nn.LayerNorm(self.inner_dim1)
        
        self.conv1 = torch.nn.Linear(input_dim, exp_dim1)
        self.conv2 = torch.nn.Linear(self.inner_dim1, exp_dim1)
        self.conv3 = torch.nn.Linear(self.inner_dim1, exp_dim2)
        self.clayer = torch.nn.Linear(int(4*self.inner_dim2), exp_dim3)
        self.lnorm3 = torch.nn.LayerNorm(int(4*self.inner_dim2))

        self.lnorm4 = torch.nn.LayerNorm(self.inner_dim3)
        self.compile_layer1 = torch.nn.Linear(self.inner_dim3,
                                              2*self.inner_dim3)
        self.lnorm5 = torch.nn.LayerNorm(self.inner_dim3)
        self.o_layer = torch.nn.Linear(self.inner_dim3, 1)


    #Forward pass function. For details of why we are doing what we are
    #doing in here, see the readme.
    def forward(self, x, training=False):
        categories = x[:,:,24:28]
        adjmat = x[:,:,28:-1]
        x = self.conv1(x[:,:,:24])
        x = x[:,:,0:self.inner_dim1] * torch.sigmoid(x[:,:,self.inner_dim1:])
        x = torch.matmul(adjmat, x)
        x = self.conv2(x)
        x = x[:,:,0:self.inner_dim1] * torch.sigmoid(x[:,:,self.inner_dim1:])
        x = self.conv3(torch.matmul(adjmat,x))
        x = x[:,:,0:self.inner_dim2] * torch.sigmoid(x[:,:,self.inner_dim2:])
        x = torch.matmul(torch.transpose(categories, 1, 2), x)
        x = torch.cat((x[:,0,:], x[:,1,:], x[:,2,:],
                       x[:,3,:]),-1).squeeze()
        
        x = self.clayer(self.lnorm3(x))
        x = x[:,0:self.inner_dim3] * torch.sigmoid(x[:,self.inner_dim3:])
        x = self.lnorm4(x)
        x = self.compile_layer1(x)
        x = x[:,0:self.inner_dim3] * torch.sigmoid(x[:,self.inner_dim3:])
        x = self.lnorm5(x)
        return self.o_layer(x).squeeze()


    def train(self, x, y, epochs=1, minibatch=100, track_loss = True,
              lr=0.005):
        self.cuda()
        #Using the Adam optimizer. You can in theory get slightly better results with
        #Nestorov SGD, but Adam is at present hard to outperform substantially.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #Standard regression loss function.
        loss_fn = torch.nn.MSELoss()
        for i in range(0, epochs):
            num_iters, current_position = 0, 0
            next_epoch = False
            #Loop through the training set on each epoch and extract a minibatch.
            while(next_epoch == False):
                if (current_position + minibatch) >= x.shape[0]:
                    xbatch = x[current_position:x.shape[0], :,:]
                    ybatch = y[current_position:y.shape[0]]
                    current_position = x.shape[0]
                else:
                    xbatch = x[current_position:(current_position+minibatch),:,:]
                    ybatch = y[current_position:(current_position+minibatch)]
                    current_position += minibatch
                if xbatch.shape[0] < 2:
                    next_epoch=True
                    print('an epoch has ended')
                    break
                #Convert to torch tensors. (Should really have saved the
                #data as torch tensors instead of numpy...this should be fixed,
                #would improve efficiency slightly by eliminating this conversion).
                #On the other hand, numpy files are much easier to use with sklearn,
                #which doesn't work with torch tensors.
                xbatch = torch.from_numpy(xbatch).float()
                ybatch = torch.from_numpy(ybatch).float()
                y_pred = self.forward(xbatch.cuda(), training=True)
                loss = loss_fn(input = y_pred, target = ybatch.cuda())
                #Backprop and...
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #Update user.
                if track_loss == True and num_iters % 50 == 0:
                    print('Current loss: %s'%loss.item())
                num_iters += 1
                #I like to slow things down very slightly to make life easier
                #for my cpu (and its cooling fan). If you are not also
                #so inclined, get rid of this line.
                time.sleep(0.0075)

    #This function is used to generate predicted values for a validation or
    #test set.
    def predict(self, x):
        current_position = 0
        #Step through the data in minibatches of size 500, because sending
        #the whole test set down to the GPU all at once is too much.
        minibatch=500
        y_pred = []
        next_epoch = False
        with torch.no_grad():
            #"Epoch" isn't really the right term, we're just looping through the whole
            #dataset essentially...
            while(next_epoch == False):
                if current_position >= x.shape[0]:
                    break
                if (current_position + minibatch) >= x.shape[0]:
                    xbatch = x[current_position:x.shape[0], :,:]
                    current_position = x.shape[0]
                    next_epoch = True
                else:
                    xbatch = x[current_position:(current_position+minibatch),:,:]
                    current_position += minibatch
                xbatch = torch.from_numpy(xbatch).float()
                y_pred.append(self.forward(xbatch.cuda(), training=False).cpu().numpy())
            y_pred = np.concatenate(y_pred,0)
            #Return as numpy array for use with sklearn routines (MAE etc).
            return y_pred
