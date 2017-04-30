import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize



def createBigPicture(nbPic,nbRow,nbCol):

    data_path = ""

    image_size = 32

    imgs = glob.glob(data_path+"*.jpg")

    res =np.zeros((nbRow*image_size,nbCol*image_size,3), dtype = "uint8")

    for i, img_path in enumerate(imgs):
        img = np.array(Image.open(img_path))

        i1 = image_size*( i/2)
        i2 = (image_size)+image_size*(i/2)
        j1 = (i%2)*image_size
        j2 = (image_size)*(1 +i%2)
        res[ j1 : j2, i1 :i2,:] = img[:,:,:]

    img = Image.fromarray(res)
    img.save("result.jpg")


createBigPicture(6,2,3)
