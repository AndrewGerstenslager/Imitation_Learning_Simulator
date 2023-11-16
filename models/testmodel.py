import sys
sys.path.append('.')

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import PIL

sys.path.append('/..')
import BasicCNN
np.random.seed(100)
device = torch.device("cuda")

def test_model(model, run_name, x_test, y_test):
    #---------------------Test All Models--------------------------
    print("Testing")
    # Activate evaluation mode
    model.eval()

    print("Evaluation")

    with torch.no_grad(): # Do not calculate gradients for testing

        dice_hist = []
        IoU_hist = []
        # predict all test samples
        with tqdm.tqdm(total=int(x_test.size()[0])) as pbar:
            for i in range(x_test.size()[0]):
                y_hat = model(x_test[i].view(1, 3, 512, 512)) # prediction

                # calculate score
                dice = liamfuncs.dice_loss(y_hat[0], y_test[i])
                IoU = liamfuncs.IoU_loss(y_hat[0], y_test[i])

                dice_hist.append(dice)
                IoU_hist.append(IoU)
                pbar.update()

        # find average scores
        dice_total = float(sum(dice_hist)/x_test.size()[0])
        IoU_total = sum(IoU_hist)/x_test.size()[0]

        # plot results
        fig, axs = plt.subplots(2, 3, num=1)
        fig.suptitle('Predict '+run_name)
        axs[0, 0].imshow(torch.movedim(x_test[-1], 0,2))
        axs[0, 0].set_title("Input")
        axs[0, 1].imshow(y_test[-1])
        axs[0, 1].set_title("Label")
        axs[1, 1].imshow(torch.ceil(y_test[-1]))
        axs[1, 1].set_title("Label (rounded)")
        axs[0, 2].imshow(y_hat[-1].view(512,512))
        axs[0, 2].set_title("Prediction")
        tol = 0.1
        axs[1, 2].imshow(torch.ceil(y_hat[-1].view(512,512)-tol))
        axs[1, 2].set_title("Prediction (rounded)")

        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])
        # plt.savefig("ref/pred_"+run_name+".png")
        print("Dice: "+str(round(dice_total, 4)))
        print("IoU: "+str(round(IoU_total, 4)))
        plt.show()

print("Loading Data...")
# import data
dir = os.getcwd()+"\\data"
x_test, y_test = liamfuncs.data_import(dir, False, True)

# DO NOT USE
# Model = models.U_net_2.U_net()
# Modelname = "U_net_2"
# Model = models.U_net_3.U_net()
# Modelname = "U_net_3"
# Model = models.U_net_4.U_net()
# Modelname = "U_net_4"
# Model = models.U_net_2_norm.U_net()
# Modelname = "U_net_2_norm"
# Model = models.U_net_3_norm.U_net()
# Modelname = "U_net_3_norm"
# Model = models.U_net_4_norm.U_net()
# Modelname = "U_net_4_norm"
# Model = models.U_net_2_norm.U_net()
# Modelname = "U_net_2_norm_drop_dice"

# SELECT MODEL
Model = models.U_net_2.U_net()
Modelname = "U_net_2_LR0.01_Ep30_Opt-SGD_Lossdice.pt"               # 1
# Model = models.U_net_3.U_net()
# Modelname = "U_net_3_LR0.01_Ep30_Opt-SGD_Lossdice.pt"             # 2
# Model = models.U_net_4.U_net()
# Modelname = "pred_U_net_4_LR0.01_Ep50_Opt-SGD_Lossdice.png"       # 3
# Model = models.U_net_2_norm.U_net()
# Modelname = "U_net_2_norm_drop_dice_LR0.01_Ep30_Opt-SGD.pt"       # 4
# Model = models.U_net_3_norm.U_net()
# Modelname = "U_net_3_norm_LR0.01_Ep50_Opt-SGD_Lossdice.pt"        # 5
# Model = models.U_net_4_norm.U_net()
# Modelname = "U_net_4_norm_LR0.05_Ep100_Opt-SGD_Lossdice.pt"       # 6

# lr = 0.05
# epochs = 100
# opt = "SGD"
# crit_name = "dice"
# run_name = Modelname+"_LR"+str(lr)+"_Ep"+str(epochs)+"_Opt-"+opt+"_Loss"+crit_name
Model.load_state_dict(torch.load("models/saved/"+Modelname, map_location=torch.device('cpu')))

test_model(Model, Modelname, x_test, y_test)
