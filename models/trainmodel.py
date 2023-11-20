import torch
from torchvision import transforms, datasets, io
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import PIL
import tqdm

sys.path.append('/..')
import BasicCNN
import PoseNet
import liamfuncs
np.random.seed(100)
device = torch.device("cuda")

dir = os.getcwd()+"\\cache"
# import data
x = liamfuncs.data_import(dir)#(2000, 3, 64, 64)

labeldata = np.genfromtxt(dir+"\\dataset_index.csv", skip_header=1)
# print(labeldata)
cpux = x.cpu().numpy()
n_samples_raw = np.shape(cpux)[0]
redbluediff = (np.sum(cpux[:,0,:,:].reshape(n_samples_raw, 1, 64**2) - cpux[:,2,:,:].reshape(n_samples_raw, 1, 64**2), axis=-1).astype('bool')).flatten()
positive = np.where(redbluediff==True)
negative = np.where(redbluediff==False)


labelconfidence = redbluediff.astype('float')
labeloffset = labeldata[:,1]

forward_sample = []
left_sample = []
right_sample = []
y = np.zeros((n_samples_raw, 4))
for i in range(n_samples_raw):
    if not labelconfidence[i]:
        y[i,:] = np.array([0, 0, 0, 1])
    else:
        if labeloffset[i] > -3 and labeloffset[i] < 3:
            y[i,:] = np.array([1, 0, 0, 0])
            forward_sample.append(i)
        elif labeloffset[i] < -3:
            y[i,:] = np.array([0, 0, 1, 0])
            right_sample.append(i)
        elif labeloffset[i] > 3:
            y[i,:] = np.array([0, 1, 0, 0])
            left_sample.append(i)
        else:
            raise ValueError("odd angle")

samples = np.concatenate((cpux[negative[0][0:100]],
                    cpux[forward_sample[0:100]],
                    cpux[right_sample[0:100]],
                    cpux[left_sample[0:100]]))/255
labels = np.concatenate((y[negative[0][0:100]],
                    y[forward_sample[0:100]],
                    y[right_sample[0:100]],
                    y[left_sample[0:100]]))

p = np.random.permutation(len(samples))
x = torch.from_numpy(samples[p]).cuda()
y = torch.from_numpy(labels[p]).cuda()

x_train = x[0:300]
y_train = y[0:300]
x_valid = x[300:350]
y_valid = y[300:350]
x_test = x[350:400]
y_test = y[350:400]

# Training parameters
epochs = 1000 # 1000)

# Loss function
# criterion = torch.nn.MSELoss()
# crit_name = "MSE"
criterion = torch.nn.CrossEntropyLoss()
crit_name = "CE"

# model
# model = BasicCNN.CNN().cuda()
# modelname = "CameraCNN"
model = PoseNet.NN().cuda()
modelname="Pose_Net"

# learning rate
lr = 0.01

# Reset optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optim_name = "SGD"
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optim_name = "Adam"

# name for files
run_name = modelname+"_LR"+str(lr)+"_Ep"+str(epochs)+"_Opt-"+optim_name+"_Loss"+crit_name

# Init list
epoch_hist = []
loss_hist = []
valid_hist = []

# Enable training mode (activates dropout, etc.)
model.train()

num_samples = x_train.size()[0]
batch_size = num_samples
i = 0

with tqdm.tqdm(total=epochs*int(num_samples/batch_size)) as pbar:
    for epoch in range(epochs):
        #print("Epoch "+str(epoch+1))break
        # randomize samples
        indices = torch.randperm(x_valid.size()[0])
        x_valid = x_valid[indices]
        y_valid = y_valid[indices]
        for batch in range(int(num_samples/batch_size)):
            #print('batch '+str(batch+1))
            epoch_hist.append(i)
            i += 1

            optimizer.zero_grad() # no cumulative gradients
            
            rnge1 = batch_size*batch
            rnge2 = batch_size*(batch+1)

            # Predict
            outputs = model(x_train[rnge1:rnge2])

            # Calculate Loss
            loss = criterion(outputs, y_train[rnge1:rnge2])
            loss_hist.append(loss.cpu().detach())
            # print(loss.cpu().detach())
            loss.backward() # gradient of loss

            optimizer.step() # update weights

            # Validation step
            with torch.no_grad(): # Do not calculate gradients for validation
                outputs = model(x_valid[0:batch_size]) # prediction

            # Calculate performance metrics and save to history
                loss = criterion(outputs, y_valid[0:batch_size])
                valid_hist.append(loss.cpu().detach())
                #print(loss.cpu().detach())

            pbar.update()
        

     

# Save Model
save = True
if save: torch.save(model.state_dict(), os.getcwd()+"/models/saved/"+run_name+".pt")

# Activate evaluation mode
model.eval()

print("Evaluation")
with torch.no_grad(): # Do not calculate gradients for testing
    for i in range(10):
        y_hat = model(x_test[i:i+1,:,:,:]) # prediction
        print("Offset Predicted:", y_hat[0])
        print("Offset label:", y_test[i])


plt.figure(2)
# plot loss vs. epoch
plot_data = np.array(loss_hist)
plot_data2 = np.array(valid_hist)
plt.plot(np.arange(1,len(plot_data)+1,1), plot_data, label="Training Loss")
plt.plot(np.arange(1,len(plot_data)+1,1), plot_data2, label="validation Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title('Training Loss: '+run_name)
plt.legend()
#plt.savefig(os.getcwd()+"/ref/Loss"+modelname+"LR"+str(lr)+".png")
# plt.savefig("ref/loss_"+run_name+".png")
plt.show()