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
import liamfuncs
np.random.seed(100)
device = torch.device("cuda")

dir = os.getcwd()+"\\data"
# import data
x_train, y_train, x_valid, y_valid, x_test, y_test = liamfuncs.data_import(dir)
# print(x_train.size()) #(73, 3, 64, 64)


# Training parameters
epochs = 100 # 1000)



# Loss function
#criterion = torch.nn.BCELoss()
criterion = liamfuncs.dice_loss
crit_name = "dice"

# Model
# model = models.U_net_2.U_net().cuda()
# modelname = "U_net_2"
# model = models.U_net_3.U_net().cuda()
# modelname = "U_net_3"
# model = models.U_net_4.U_net().cuda()
# modelname = "U_net_4"
# model = models.U_net_2_norm.U_net().cuda()
# modelname = "U_net_2_norm"
# model = models.U_net_3_norm.U_net().cuda()
# modelname = "U_net_3_norm"
model = models.U_net_4_norm.U_net().cuda()
modelname = "U_net_4_norm"

# learning rate
lr = 0.05

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
batch_size = 4
num_samples = x_train.size()[0]
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

            # # show training images (debug)
            # fig, axs = plt.subplots(1, 2)
            # fig.suptitle('Multiple images')
            # axs[0].imshow(torch.movedim(x_train[rnge1].cpu(), 0,2))
            # axs[1].imshow(y_train[rnge1].cpu())
            # plt.show()

            pbar.update()
        

     

# Save Model
save = True
if save: torch.save(model.state_dict(), "models/saved/"+run_name+".pt")

# Activate evaluation mode
model.eval()

print("Evaluation")
with torch.no_grad(): # Do not calculate gradients for testing
    y_hat = model(x_test[0:4]) # prediction

    # Test Image
    fig, axs = plt.subplots(1, 3, num=1)
    fig.suptitle('Predict '+run_name)
    axs[0].imshow(torch.movedim(x_test[0].cpu(), 0,2))
    axs[0].set_title("Input")
    axs[1].imshow(y_test[0].cpu())
    axs[1].set_title("Label")
    axs[2].imshow(y_hat[0].view(512,512).cpu().detach().numpy())
    axs[2].set_title("Prediction")
    plt.savefig("ref/pred_"+run_name+".png")


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
plt.savefig("ref/loss_"+run_name+".png")
plt.show()