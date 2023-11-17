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
redbluediff = (np.sum(cpux[:,0,:,:].reshape(2000, 1, 64**2) - cpux[:,2,:,:].reshape(2000, 1, 64**2), axis=-1).astype('bool')).flatten()
positive = np.where(redbluediff==True)
x = x[positive]
labelconfidence = redbluediff.astype('float')
labeloffset = labeldata[:,1]
y = torch.from_numpy(labeloffset).type(torch.float).view(len(labeloffset), 1).cuda()# torch.from_numpy(np.stack((labeloffset, labelconfidence), axis=1)).type(torch.float).cuda()
y = y[positive]
min1 = -60 #torch.min(y)
max1 = 60 #torch.max(y)
print(min1)
print(max1)
y = (y - min1) / (max1 - min1)
x_train = x[0:400,:,:,:]
y_train = y[0:400]
x_valid = x[400:425,:,:,:]
y_valid = y[400:425]
x_test = x[425:450,:,:,:]
y_test = y[425:450]

# Training parameters
epochs = 1000 # 1000)

# Loss function
criterion = torch.nn.MSELoss()
crit_name = "MSE"

# model
# model = BasicCNN.CNN().cuda()
# modelname = "CameraCNN"
model = PoseNet.NN().cuda()
modelname="Pose_Net"

# learning rate
lr = 0.001

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

# print("Evaluation")
# with torch.no_grad(): # Do not calculate gradients for testing
#     for i in range(10):
#         y_hat = model(x_test[i:i+1,:,:,:]) # prediction
#         print("Offset Predicted:", y_hat[0])
#         print("Offset label:", y_test[i])


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

ax = plt.figure(3).add_subplot(projection='3d')
# plot pred vs true
y_hat = model(x_train).cpu().detach().numpy()
y_true = y_train.cpu().detach().numpy()
depthdata = labeldata[:,2]
depthpositive = depthdata[positive]
depth = depthpositive[0:400].reshape(400, 1)
print(np.shape(depth))
ax.plot(y_true, y_hat, depth, 'g.')
plt.xlabel("true")
plt.ylabel("pred")
ax.set_zlabel("Sample Depth")
plt.title('Model Prediction')
plt.axis([0, 1, 0, 1])
plt.show()