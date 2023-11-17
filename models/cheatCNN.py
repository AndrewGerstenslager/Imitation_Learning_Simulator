import numpy as np

def readImg(img):
    img = np.moveaxis(img, 2, 0)/255
    redbluediff = (img[0,:,:] - img[2,:,:]).astype('bool')
    positive = np.where(redbluediff==True)
    offset = np.mean(positive[0])-32
    if offset < -3:
        ctrl = [0, 1, 0]
    elif offset > 3:
        ctrl = [0, 0, 1]
    elif offset > -3 and offset < 3:
        ctrl = [1, 0, 0]
    else:
        ctrl = [0, 0, 0]
    return ctrl