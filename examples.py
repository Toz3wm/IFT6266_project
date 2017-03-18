import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize


import theano
import theano.tensor as T
from theano import shared
import random


def resize_mscoco():
    '''
    function used to create the dataset,
    Resize original MS_COCO Image into 64x64 images
    '''

    ### PATH need to be fixed
    data_path="inpainting/train2014"
    save_dir = "inpainting/train2014cropped"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preserve_ratio = True
    image_size = (64, 64)
    #crop_size = (32, 32)

    imgs = glob.glob(data_path+"/*.jpg")

    timer = 0
    for i, img_path in enumerate(imgs):
        if timer > 10:
            break
        else:
            timer += 1
        img = Image.open(img_path)
        print i, len(img), img_path

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

                ### Crop the 64/64 center
                tocrop = np.array(img)
                center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print tocrop.shape, center, (center[0]-32,center[0]+32), (center[1]-32,center[1]+32)
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                else:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))




def show_examples(batch_idx, batch_size,
                  ### PATH need to be fixed
                  mscoco="inpainting", split="val2014", caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    print(data_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    print("imgs size :"+str(len(imgs)))
    #batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]
    batch_imgs = imgs
    print "starting preprocessing"
    for i, img_path in enumerate(batch_imgs):
        #print(img_path)
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

        imgInput = Image.fromarray(input)
        imgTarget = Image.fromarray(target)
        imgInput.save("inpainting/val2014cropped/" + os.path.basename(img_path))
        imgTarget.save("inpainting/val2014target/" + os.path.basename(img_path))
        if i%100 == 0:
            print str(i)+ "pictures processed out of" + str(len(batch_imgs))
        #Image.fromarray(img_array).show()
        #Image.fromarray(input).show()
        #Image.fromarray(target).show()
        #print i, caption_dict[cap_id]

def reconstruct_image(border_pic, center_pic):
    input = np.copy(border_pic)
    center = (int(np.floor(input.shape[1] / 2.)), int(np.floor(input.shape[2] / 2.)))
    input[:,center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = center_pic
    return input







if __name__ == '__main__':
    #resize_mscoco()
    show_examples(0, 10)
